"""
paper_discovery_hpc.py — BFS paper discovery + LLM relevance filtering on HPC.

Mirrors the infrastructure of function_extraction_from_paper.py. Designed to
run as a single SLURM job (not an array — BFS is depth-sequential, and one
llama-server with -np 16 parallel slots saturates a single A100 for a 4B
relevance-filtering model).

Workflow:
    1. Locally: write all_seeds list to seeds.txt (one DOI/URL per line)
    2. rsync paper_discovery/ to HPC
    3. (Optional) rsync local eval_cache.json into --output-dir to seed the cache
    4. sbatch paper_discovery_hpc.sh
    5. rsync discovered_papers.csv back to laptop
    6. Run local PDF-matching cell in notebook against the downloaded CSV

Usage:
    python paper_discovery_hpc.py \\
        --seeds /path/to/seeds.txt \\
        --output-dir /path/to/output \\
        --gguf /path/to/qwen3.5-4b.gguf \\
        --max-depth 2 \\
        --required-keywords "Drosophila|fruit fly|melanogaster"

Outputs (in --output-dir):
    discovered_papers.csv       — relevant papers with metadata + depth
    eval_cache.json             — LLM relevance decisions (resumable)
    s2_cache/                   — Semantic Scholar HTTP cache (resumable)
    depth_N_candidates.csv      — per-depth checkpoint for inspection
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests_cache
from openai import OpenAI
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs, unquote
import requests  # for CrossRef PII lookup

from llm_core import (
    find_free_port, start_llama_server, wait_for_llama_server,
)


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True,
                   help="Text file with one seed DOI or URL per line. # comments allowed.")
    p.add_argument("--output-dir", default="./discovery_output")
    p.add_argument("--gguf", required=True,
                   help="Path to GGUF. 4B model recommended for relevance filtering.")
    p.add_argument("--llama-bin",
                   default=os.path.expanduser("~/llama.cpp/build/bin/llama-server"))
    p.add_argument("--llama-port", type=int, default=None)
    p.add_argument("--ctx-size", type=int, default=4096,
                   help="llama-server context window. Title+abstract is ~1k tokens.")
    p.add_argument("--n-parallel-slots", type=int, default=16,
                   help="llama-server -np (parallel decoding slots)")
    p.add_argument("--n-llm-workers", type=int, default=16,
                   help="Concurrent client threads making LLM requests")
    p.add_argument("--n-fetch-workers", type=int, default=8,
                   help="Concurrent S2 fetch threads (still globally rate-limited)")
    p.add_argument("--s2-rate", type=float, default=1.0,
                   help="S2 requests per second (1.0 without API key)")
    p.add_argument("--s2-api-key",
                   default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
                   help="Optional S2 API key (also reads $SEMANTIC_SCHOLAR_API_KEY)")
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--max-papers-per-seed", type=int, default=50,
                   help="Max citations + references collected per relevant paper")
    p.add_argument("--required-keywords", default="",
                   help="Pipe-separated alternatives within a group; '||' between groups. "
                        "E.g., 'Drosophila|fruit fly||neuron|circuit' = "
                        "(Drosophila or fruit fly) AND (neuron or circuit).")
    p.add_argument("--save-every", type=int, default=10,
                   help="Persist eval_cache every N completed evaluations")
    return p.parse_args()


# ── Relevance prompt & eval ────────────────────────────────────────────────
FILTER_SYSTEM_PROMPT = """You are a research assistant helping curate a database of Drosophila neuroscience papers.

The database contains papers that experimentally test the function of specific neurons or cell types in Drosophila, typically using optogenetics, calcium imaging, electrophysiology, or behavioural assays paired with neural manipulation.

For each paper, decide if it is RELEVANT based on its title and abstract.

A paper is RELEVANT if it:
- Studies Drosophila (any species, larval or adult)
- Experimentally investigates the function of specific identified neurons or cell types
- Uses techniques like optogenetics, calcium imaging, electrophysiology, behavioural genetics

A paper is NOT RELEVANT if it:
- Is about non-Drosophila species (unless directly comparative)
- Is purely computational/theoretical with no Drosophila experiments
- Focuses on molecular biology or development without functional neural experiments
- Is purely anatomical with no functional testing

Respond ONLY with a JSON object:
{"relevant": bool, "confidence": "high"|"medium"|"low", "reason": "brief justification"}"""

_eval_cache_lock = threading.Lock()
_save_counter = {"n": 0}


def _save_cache_atomic(cache, path):
    """Write to .tmp then rename. Safe against mid-write kill."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    tmp.replace(path)


def evaluate_paper_relevance(title, abstract, llama_port, eval_cache,
                              save_path, save_every):
    if not abstract or len(abstract.strip()) < 50:
        return {"relevant": False, "confidence": "high", "reason": "No abstract"}

    cache_key = f"{title}|{abstract[:100]}"
    with _eval_cache_lock:
        if cache_key in eval_cache:
            return eval_cache[cache_key]

    user_prompt = f"Title: {title}\n\nAbstract: {abstract}\n\nIs this paper relevant?"
    try:
        client = OpenAI(base_url=f"http://localhost:{llama_port}/v1",
                        api_key="not-needed")
        resp = client.chat.completions.create(
            model="",
            messages=[
                {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                {"role": "user",   "content": "/no_think\n" + user_prompt},
            ],
            max_tokens=512,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        result = json.loads(text)
        if "relevant" not in result:
            return {"relevant": False, "confidence": "low",
                    "reason": "Malformed LLM output"}
    except Exception as e:
        return {"relevant": False, "confidence": "low",
                "reason": f"LLM error: {e}"}

    with _eval_cache_lock:
        eval_cache[cache_key] = result
        _save_counter["n"] += 1
        if _save_counter["n"] >= save_every:
            _save_cache_atomic(eval_cache, save_path)
            _save_counter["n"] = 0
    return result


# ── Keyword filter ─────────────────────────────────────────────────────────
def matches_keywords(text, kw_spec):
    """'A|B||C|D' = (A or B) AND (C or D). Empty → pass-through."""
    if not kw_spec:
        return True
    text = text.lower()
    for group in kw_spec.split("||"):
        alts = [a.strip().lower() for a in group.split("|") if a.strip()]
        if alts and not any(a in text for a in alts):
            return False
    return True


# ── Semantic Scholar (cached + rate-limited) ───────────────────────────────
class RateLimiter:
    def __init__(self, cps):
        self.min_interval = 1.0 / cps
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()


class SemanticScholarClient:
    def __init__(self, session, limiter, api_key=None):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.session = session
        self.limiter = limiter
        self.headers = {"x-api-key": api_key} if api_key else {}

    def _get(self, url):
        # Probe cache cheaply; cache hits skip rate limiting entirely
        try:
            cached = self.session.get(url, only_if_cached=True,
                                       headers=self.headers, timeout=5)
            if getattr(cached, "from_cache", False) and cached.status_code == 200:
                return cached
        except Exception:
            pass
        self.limiter.wait()
        return self.session.get(url, headers=self.headers, timeout=15)

    def get_paper_by_doi(self, doi):
        fields = "paperId,title,abstract,year,authors,citationCount,openAccessPdf,externalIds"
        r = self._get(f"{self.base_url}/paper/DOI:{doi}?fields={fields}")
        return r.json() if r.status_code == 200 else None

    def get_paper_by_id(self, paper_id):
        fields = "paperId,title,abstract,year,authors,citationCount,openAccessPdf,externalIds"
        r = self._get(f"{self.base_url}/paper/{paper_id}?fields={fields}")
        return r.json() if r.status_code == 200 else None

    def get_citations(self, paper_id, limit=100):
        fields = "paperId,title,abstract,year,openAccessPdf,externalIds"
        r = self._get(f"{self.base_url}/paper/{paper_id}/citations?fields={fields}&limit={limit}")
        if r.status_code == 200:
            data = (r.json() or {}).get("data") or []
            return [it["citingPaper"] for it in data if it and it.get("citingPaper")]
        return []

    def get_references(self, paper_id, limit=100):
        fields = "paperId,title,abstract,year,openAccessPdf,externalIds"
        r = self._get(f"{self.base_url}/paper/{paper_id}/references?fields={fields}&limit={limit}")
        if r.status_code == 200:
            data = (r.json() or {}).get("data") or []
            return [it["citedPaper"] for it in data if it and it.get("citedPaper")]
        return []


DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>?#]+')
_pii_doi_cache = {}

def url_to_doi(s):
    if not isinstance(s, str):
        return None
    s = s.strip().strip('"').strip()
    if not s:
        return None

    m = re.fullmatch(r"(?:https?://)?(?:dx\.)?doi\.org/(.+)", s, re.I)
    if m:
        return unquote(m.group(1)).rstrip(".")
    if s.startswith("10."):
        m = DOI_RE.match(s)
        if m:
            return m.group(0).rstrip(".")

    try:
        u = urlparse(s if "://" in s else "https://" + s)
    except Exception:
        return None
    host = (u.netloc or "").lower().removeprefix("www.")
    path = unquote(u.path or "")

    if host.endswith("nature.com"):
        m = re.search(r"/articles/([^/?#]+)", path)
        if m:
            return f"10.1038/{m.group(1)}"

    if host.endswith("elifesciences.org"):
        m = re.search(r"/(?:articles|reviewed-preprints)/(\d+)", path)
        if m:
            return f"10.7554/eLife.{m.group(1)}"

    if host.endswith("biorxiv.org") or host.endswith("medrxiv.org"):
        m = re.search(r"/content/(10\.\d{4,9}/[\d.]+)", path)
        if m:
            return m.group(1)

    if "plos.org" in host:
        qs = parse_qs(u.query)
        if "id" in qs:
            return qs["id"][0]

    if host.endswith("frontiersin.org"):
        m = re.search(r"/articles/(10\.\d{4,9}/[^/?#]+)", path)
        if m:
            return m.group(1)

    m = DOI_RE.search(path + "?" + (u.query or ""))
    if m:
        return m.group(0).rstrip(".&/")

    return None


def pii_to_doi(pii, timeout=10):
    pii = re.sub(r"[^A-Za-z0-9]", "", pii)
    if not pii:
        return None
    if pii in _pii_doi_cache:
        return _pii_doi_cache[pii]
    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params={"filter": f"alternative-id:{pii}", "rows": 1},
            headers={"User-Agent": "paper-discovery/0.1 (mailto:yy432@cam.ac.uk)"},
            timeout=timeout,
        )
        if r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            doi = items[0].get("DOI") if items else None
        else:
            doi = None
    except Exception as e:
        print(f"CrossRef lookup failed for PII {pii}: {e}")
        doi = None
    _pii_doi_cache[pii] = doi
    return doi


def extract_pii(url):
    m = re.search(r"/pii/(S?[\w()-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"cell\.com/[^/?#]+/[^/?#]+/(S\d[\d()-]+[\dX])", url, re.I)
    if m:
        return m.group(1)
    return None


def url_to_ss_id(s, lookup_pii=True):
    """Returns bare DOI ('10.x/y'), or 'PMID:...'/'PMCID:PMC...', or None."""
    if not isinstance(s, str):
        return None
    s = s.strip().strip('"').strip()
    if not s:
        return None

    m = re.search(r"/PMC(\d+)", s, re.I)
    if m:
        return f"PMCID:PMC{m.group(1)}"
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", s)
    if m:
        return f"PMID:{m.group(1)}"

    doi = url_to_doi(s)
    if doi:
        return doi

    if lookup_pii:
        pii = extract_pii(s)
        if pii:
            return pii_to_doi(pii)

    return None


def fetch_seed(ss_client, ident_or_url):
    ident = url_to_ss_id(ident_or_url)
    if not ident:
        return None
    if ident.startswith(("PMID:", "PMCID:", "URL:", "ARXIV:", "MAG:", "CorpusId:")):
        return ss_client.get_paper_by_id(ident)
    return ss_client.get_paper_by_doi(ident)


# ── BFS discovery ──────────────────────────────────────────────────────────
def _paper_record(p, depth, confidence, reason):
    return {
        "paper_id": p["paperId"],
        "title":    p.get("title", ""),
        "abstract": p.get("abstract", ""),
        "year":     p.get("year"),
        "doi":      (p.get("externalIds") or {}).get("DOI"),
        "open_access_pdf":
            (p.get("openAccessPdf") or {}).get("url") if p.get("openAccessPdf") else None,
        "depth":    depth,
        "relevance_confidence": confidence,
        "relevance_reason":     reason,
    }


def discover(args, llama_port, eval_cache, eval_cache_path):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session = requests_cache.CachedSession(
        str(out / "s2_cache"),
        backend="sqlite",
        expire_after=60 * 60 * 24 * 30,
    )
    limiter = RateLimiter(args.s2_rate)
    ss = SemanticScholarClient(session, limiter, api_key=args.s2_api_key)

    raw = Path(args.seeds).read_text().splitlines()
    seeds = [l.strip() for l in raw if l.strip() and not l.strip().startswith("#")]
    print(f"Loaded {len(seeds)} seed lines from {args.seeds}")

    print("Fetching seed metadata from Semantic Scholar...")
    with ThreadPoolExecutor(max_workers=args.n_fetch_workers) as ex:
        seed_results = list(tqdm(
            ex.map(lambda s: fetch_seed(ss, s), seeds),
            total=len(seeds), desc="Seeds",
        ))

    seen, all_papers, seed_ids = set(), [], set()
    current = []
    for p in seed_results:
        if p and p.get("paperId") and p["paperId"] not in seen:
            seen.add(p["paperId"])
            current.append(p)
            seed_ids.add(p["paperId"])
    print(f"Resolved {len(current)}/{len(seeds)} seeds to S2 records")

    for depth in range(args.max_depth + 1):
        if not current:
            break
        print(f"\n=== Depth {depth}: {len(current)} candidates ===")

        # Per-depth checkpoint for inspection
        pd.DataFrame([{
            "paper_id": p["paperId"],
            "title":    p.get("title", ""),
            "doi":      (p.get("externalIds") or {}).get("DOI"),
            "year":     p.get("year"),
            "depth":    depth,
        } for p in current]).to_csv(out / f"depth_{depth}_candidates.csv", index=False)

        # Keyword filter; seeds always pass
        passed_kw = []
        for p in current:
            if p["paperId"] in seed_ids:
                passed_kw.append(p)
                continue
            ab = p.get("abstract") or ""
            if len(ab.strip()) < 50:
                continue
            if matches_keywords(f"{p.get('title','')} {ab}", args.required_keywords):
                passed_kw.append(p)
        n_seeds_through = sum(1 for p in passed_kw if p["paperId"] in seed_ids)
        print(f"After keyword filter: {len(passed_kw)} ({n_seeds_through} seeds whitelisted)")

        # Seeds: auto-include without LLM
        relevant = []
        for p in passed_kw:
            if p["paperId"] in seed_ids:
                relevant.append(p)
                all_papers.append(_paper_record(p, depth, "high", "seed (auto-included)"))

        # Non-seeds: LLM eval in parallel
        to_eval = [p for p in passed_kw if p["paperId"] not in seed_ids]
        if to_eval:
            with ThreadPoolExecutor(max_workers=args.n_llm_workers) as ex:
                futs = {
                    ex.submit(
                        evaluate_paper_relevance,
                        p.get("title", ""), p.get("abstract", ""),
                        llama_port, eval_cache, eval_cache_path, args.save_every,
                    ): p
                    for p in to_eval
                }
                for fut in tqdm(as_completed(futs), total=len(futs),
                                desc=f"LLM d={depth}"):
                    p = futs[fut]
                    try:
                        ev = fut.result()
                    except Exception:
                        continue
                    if ev.get("relevant"):
                        relevant.append(p)
                        all_papers.append(_paper_record(
                            p, depth, ev.get("confidence"), ev.get("reason")))

        print(f"Relevant at depth {depth}: {len(relevant)}")

        # Final save at end of each depth
        with _eval_cache_lock:
            _save_cache_atomic(eval_cache, eval_cache_path)

        # Always-current discovered_papers.csv
        pd.DataFrame(all_papers).to_csv(out / "discovered_papers.csv", index=False)

        if depth >= args.max_depth:
            break

        # Expand citations + references (parallel; globally rate-limited)
        def expand(p):
            half = max(1, args.max_papers_per_seed // 2)
            return (ss.get_citations(p["paperId"], limit=half)
                    + ss.get_references(p["paperId"], limit=half))

        nxt = []
        if relevant:
            with ThreadPoolExecutor(max_workers=args.n_fetch_workers) as ex:
                expand_results = list(tqdm(
                    ex.map(expand, relevant), total=len(relevant),
                    desc=f"Expand d={depth}",
                ))
            for related in expand_results:
                for rp in related:
                    if rp and rp.get("paperId") and rp["paperId"] not in seen:
                        seen.add(rp["paperId"])
                        nxt.append(rp)
        current = nxt
        print(f"Queued for depth {depth+1}: {len(current)}")

    pd.DataFrame(all_papers).to_csv(out / "discovered_papers.csv", index=False)
    print(f"\nDone. {len(all_papers)} relevant papers → {out / 'discovered_papers.csv'}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_cache_path = out / "llm_evaluation_cache.json"

    # Resume from existing eval cache if present
    eval_cache = {}
    if eval_cache_path.exists():
        try:
            eval_cache = json.loads(eval_cache_path.read_text())
            print(f"Loaded {len(eval_cache)} cached evaluations from {eval_cache_path}")
        except Exception as e:
            print(f"Could not load existing eval_cache, starting fresh: {e}")

    port = args.llama_port or find_free_port()
    proc = start_llama_server(args.llama_bin, args.gguf, port,
                              args.ctx_size, args.n_parallel_slots)
    try:
        wait_for_llama_server(port)
        # Warmup
        print("[llama-server] Warming up...")
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")
        warmup_deadline = time.time() + 120
        warmed = False
        while time.time() < warmup_deadline:
            try:
                client.chat.completions.create(
                    model="",
                    messages=[{"role": "user", "content": "/no_think\nSay OK."}],
                    max_tokens=16,
                )
                warmed = True
                break
            except Exception as e:
                print(f"[llama-server] Warmup retry ({e})")
                time.sleep(10)
        if not warmed:
            raise RuntimeError("llama-server failed warmup")
        print("[llama-server] Ready.")

        discover(args, port, eval_cache, eval_cache_path)
    finally:
        print("[llama-server] Shutting down...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        with _eval_cache_lock:
            _save_cache_atomic(eval_cache, eval_cache_path)


if __name__ == "__main__":
    main()