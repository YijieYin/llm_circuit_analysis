"""
normalize_cell_types.py — Map extracted cell_type strings to connectome-canonical names.
 
Pipeline per unique cell_type string:
  1. Exact lookup in synonym table (with normalisation variants).
  2. If no match: LLM call with top-N candidates retrieved by token overlap.
     LLM outputs exactly one of: MATCH: <n> | MULTI | UNMAPPABLE
  3. Build alias→canonical dict, join back onto original CSV.
     Adds columns: canonical_ct, ct_source ('exact'|'llm'|'null')
 
Usage:
    python normalize_cell_types.py \\
        --input      extracted.csv \\
        --connectome connectome_names.txt \\
        --output     normalized_extracted.csv \\
        --provider   llama \\
        --gguf       /path/to/model.gguf \\
        --llama-bin  ~/llama.cpp/build/bin/llama-server
 
    # Local (Ollama, assumes `ollama serve` is already running):
    python normalize_cell_types.py \\
        --input      extracted.csv \\
        --connectome connectome_names.txt \\
        --output     normalized_extracted.csv \\
        --provider   ollama \\
        --model      qwen3.5:35b
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from synonyms import SYNONYMS


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument(
        "--connectome",
        default=None,
        help="Flat text file, one connectome name per line. "
        "If omitted, names are fetched from the four connectome "
        "GitHub CSVs (requires internet access).",
    )
    p.add_argument("--output", required=True)
    p.add_argument(
        "--provider",
        default="llama",
        choices=["llama", "anthropic", "openai", "ollama"],
    )
    p.add_argument("--gguf", default=None)
    p.add_argument(
        "--llama-bin", default=os.path.expanduser("~/llama.cpp/build/bin/llama-server")
    )
    p.add_argument(
        "--llama-port",
        type=int,
        default=None,
        help="Port for llama-server (auto-assigned if unset)",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--n-candidates", type=int, default=30)
    p.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds between calls (cloud APIs only)",
    )
    return p.parse_args()


# ── llama-server management ───────────────────────────────────────────────────
def _find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_server(llama_bin, gguf, port):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/public/gcc/12_2_0/lib64:" + env.get(
        "LD_LIBRARY_PATH", ""
    )
    cmd = [
        llama_bin,
        "-m",
        gguf,
        "--port",
        str(port),
        "-ngl",
        "99",
        "--ctx-size",
        "4096",
        "--no-mmap",
        "-np",
        "1",
    ]
    print(f"[llama-server] Starting on port {port}")
    return subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
    )


def _wait_for_server(port, timeout=300):
    import urllib.request

    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    time.sleep(30)  # allow model weights to fully load
                    print("[llama-server] Ready")
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("llama-server did not become ready in time")


# ── normalisation helpers ─────────────────────────────────────────────────────
_GREEK = {
    "α": "a",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "ε": "e",
    "α′": "a'",
    "α'": "a'",
    "β′": "b'",
    "β'": "b'",
    "′": "'",
}


def _norm(s: str) -> str:
    s = s.strip().lower()
    for g, a in _GREEK.items():
        s = s.replace(g, a)
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)  # strip trailing "(LHON)" etc.
    return re.sub(r"\s+", " ", s)


_NORM_SYNONYMS: dict[str, str] = {_norm(k): v for k, v in SYNONYMS.items()}

_STOP = re.compile(
    r"\s*(neurons?|cells?|interneurons?|projections?\s*neurons?|"
    r"dopaminergic|serotonergic|cholinergic|glutamatergic|gabaergic|"
    r"dans?|mbons?|kcs?|orns?|pns?|lhns?|lhons?|lhlns?|"
    r"\(dans?\)|\(mbons?\)|\(kcs?\)|\(orns?\)|\(pns?\))\s*$",
    re.IGNORECASE,
)


def exact_lookup(ct: str) -> str | None:
    if ct in SYNONYMS:
        return SYNONYMS[ct]
    n = _norm(ct)
    if n in _NORM_SYNONYMS:
        return _NORM_SYNONYMS[n]
    # iteratively strip trailing descriptor words
    stripped = n
    for _ in range(4):
        new = _STOP.sub("", stripped).strip()
        if new == stripped:
            break
        stripped = new
        if stripped in _NORM_SYNONYMS:
            return _NORM_SYNONYMS[stripped]
    return None


# ── LaTeX cleaning ────────────────────────────────────────────────────────────
_LATEX_GREEK = {
    r"\\alpha": "α",
    r"\\beta": "β",
    r"\\gamma": "γ",
    r"\\delta": "δ",
    r"\\epsilon": "ε",
    r"\\Alpha": "Α",
    r"\\Beta": "Β",
    r"\\Gamma": "Γ",
    r"\\Delta": "Δ",
}


def _clean_latex(s: str) -> str:
    """Strip LaTeX math delimiters and replace Greek letter commands."""
    for cmd, ch in _LATEX_GREEK.items():
        s = s.replace(cmd, ch)
    # Remove remaining $ delimiters
    s = s.replace("$", "")
    # Remove remaining backslash-letter commands (e.g. \prime → ')
    s = re.sub(r"\\prime", "'", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return s.strip()


# ── candidate retrieval ───────────────────────────────────────────────────────
def _tokens(s: str) -> set[str]:
    toks = set(re.findall(r"[a-zA-Zα-ωΑ-Ω0-9']+", s.lower()))
    # add singular forms so e.g. 'tpgrns' matches 'tpgrn'
    toks |= {t[:-1] for t in toks if t.endswith("s") and len(t) > 2}
    return toks


# Words common in paper descriptions but absent from connectome canonical names.
# Stripped from the query before scoring to avoid spurious matches
# (e.g. 'neuron' matching the connectome name 'AC neuron').
_QUERY_STOPWORDS = {
    "neuron",
    "neurons",
    "Neuron",
    "Neurons",
    "cell",
    "cells",
    "interneuron",
    "interneurons",
    "projection",
    "projections",
    "dopaminergic",
    "serotonergic",
    "cholinergic",
    "glutamatergic",
    "gabaergic",
    "receptor",
    "sensory",
    "motor",
    "output",
    "input",
    "inhibitory",
    "excitatory",
    "expressing",
    "labeled",
    "positive",
    "negative",
    "type",
    "types",
    "subtype",
    "subtypes",
    "cluster",
    "clusters",
    "population",
    "subset",
    "like",
    "general",
    "intrinsic",
    "extrinsic",
    "body",
    "lobe",
    "lobes",
    "antennal",
    "brain",
    "central",
    "lateral",
    "dorsal",
    "ventral",
    "anterior",
    "posterior",
    "medial",
    "bilateral",
    "ascending",
    "descending",
}


def retrieve_candidates(ct: str, connectome: list[str], n: int) -> list[str]:
    qt = _tokens(ct) - _QUERY_STOPWORDS
    if not qt:
        return []
    scored = sorted(
        [(len(qt & _tokens(nm)), nm) for nm in connectome if qt & _tokens(nm)],
        key=lambda x: -x[0],
    )
    prefix = re.split(r"[-_\s]", ct)[0].lower()
    prefix_matches = [nm for nm in connectome if nm.lower().startswith(prefix)][:15]
    seen, out = set(), []
    for nm in prefix_matches + [nm for _, nm in scored[:n]]:
        if nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out[:n]  # empty list → caller will skip LLM for this ct


# ── LLM ──────────────────────────────────────────────────────────────────────
SYSTEM = (
    "You have good knowledge of Drosophila neuroscience, and map Drosophila neuron names from papers to connectome-canonical names.\n"
    "Respond with EXACTLY one of:\n"
    "  MATCH: <canonical_name>\n"
    "  MULTI\n"
    "  UNMAPPABLE\n"
    "No other text."
)


def _prompt(ct: str, candidates: list[str]) -> str:
    cands = "\n".join(f"  {c}" for c in candidates)
    return (
        f'Neuron string from paper: "{ct}"\n\n'
        f"Connectome candidate names:\n{cands}\n\n"
        "Rules:\n"
        "- MATCH: <name>  if the string clearly and specifically refers to exactly ONE\n"
        "  candidate. Copy the name verbatim from the list. Only match if unambiguous.\n"
        "- MULTI: <n1>, <n2>, ...  if the string explicitly names multiple resolvable\n"
        "  cell types (e.g. 'oviDNa and oviDNb'). Names must appear in the candidate list.\n"
        "  If individually unresolvable, output MULTI with no names.\n"
        "- UNMAPPABLE  if the string is generic, a class/population rather than a specific\n"
        "  cell type, a GAL4 driver line, or no single candidate clearly fits.\n"
        "  When in doubt, output UNMAPPABLE.\n"
        "\n"
        "Generic strings that should be UNMAPPABLE (not matched to any single candidate):\n"
        "  'Olfactory Sensory Neurons (ORNs)'  — class of many types, not one cell type\n"
        "  'PAM dopaminergic neurons'           — a cluster, not a specific cell type\n"
        "  'Kenyon Cells (KCs)'                 — whole population\n"
        "  'Mushroom Body Output Neurons'       — class of many types\n"
        "  'GABAergic local interneurons'       — functional class, not a cell type\n"
        "  'Local Neurons (LNs)'              — class of many types\n"
        "  'Lateral Accessory Lobe (LAL) neurons' — neurons in a brain region, not a cell type\n"
        "  'Lateral Horn (LH) Neurons'         — neurons in a brain region, not a cell type\n"
        "\n"
        "Strings with specific ID that SHOULD match:\n"
        "  'CB0233 neurons (FDA-II)'  →  MATCH: CB0233  (ID present verbatim)\n"
        "  'AV1a1 (LHON)'            →  MATCH: LHAV1a1  (LH prefix omitted in literature)\n"
        "  'oviDNa and oviDNb'       →  MULTI: oviDNa, oviDNb\n"
        "\n"
        "Respond with exactly one line, no extra text."
    )


def call_llm(ct, candidates, provider, model, port, max_tokens):
    prompt = _prompt(ct, candidates)
    if provider == "llama":
        from openai import OpenAI

        r = OpenAI(
            base_url=f"http://localhost:{port}/v1", api_key="x"
        ).chat.completions.create(
            model="",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "/no_think\n" + prompt},
            ],
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()
    elif provider == "anthropic":
        from anthropic import Anthropic

        r = Anthropic().messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in r.content if hasattr(b, "text")).strip()
    elif provider == "ollama":
        from ollama import Client as OllamaClient

        client = OllamaClient(host="http://localhost:11434")
        r = client.chat(
            model=model or "qwen3.5:35b",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            think=False,
            options={"num_predict": max_tokens},
        )
        return (r.message.content or "").strip()
    elif provider == "openai":
        from openai import OpenAI

        r = OpenAI().chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()


def parse_llm_output(text: str, candidates: list[str]) -> tuple[str | list[str] | None, str]:
    """Returns (canonical, source) where canonical may be a list for MULTI.

    Validation applied to MATCH results:
      (a) Multiple MATCH: lines → treated as MULTI.
      (b) Returned name must appear in the candidate list.
      (c) Returned name must be a normalised-substring of the original cell_type
          string (checked by the caller after this function returns).
    """
    text = text.strip()

    # (a) Detect multiple MATCH: lines masquerading as a single response
    match_lines = [l.strip() for l in text.splitlines() if l.strip().startswith("MATCH:")]
    if len(match_lines) > 1:
        names = [l[len("MATCH:"):].strip() for l in match_lines]
        # Keep only names that are in the candidate list
        valid = [n for n in names if n in candidates]
        return (valid if valid else None), "multi"

    if text.startswith("MATCH:"):
        name = text[len("MATCH:"):].strip()
        # (b) Validate against candidate list
        if name not in candidates:
            return None, "null"
        return name, "llm"

    if text.startswith("MULTI:"):
        names = [n.strip() for n in text[len("MULTI:"):].split(",") if n.strip()]
        return (names if names else None), "multi"
    if text.strip() == "MULTI":
        return None, "multi"
    return None, "null"


# ── substring validation ──────────────────────────────────────────────────────
def _norm_for_substr(s: str) -> str:
    """Normalise a string for substring containment check:
    lowercase, greek→ascii, strip punctuation/spaces."""
    s = s.lower()
    for g, a in _GREEK.items():
        s = s.replace(g, a)
    # collapse non-alphanumeric runs to empty so e.g. "KCab-p" matches "KCab p"
    s = re.sub(r"[^a-z0-9']", "", s)
    return s


def llm_match_is_substring(ct: str, canonical: str) -> bool:
    """Return True if canonical (normalised) appears within ct (normalised)."""
    return _norm_for_substr(canonical) in _norm_for_substr(ct)


# ── connectome name fetching ──────────────────────────────────────────────────
_CONNECTOME_SOURCES = [
    (
        "https://raw.githubusercontent.com/YijieYin/connectome_data_prep/refs/heads/main/data/fafb_all_neuron/fafb_all_neuron_meta.csv",
        "cell_type",
    ),
    (
        "https://raw.githubusercontent.com/YijieYin/connectome_data_prep/refs/heads/main/data/maleCNS/mcns_all_neuron_meta.csv",
        "cell_type",
    ),
    (
        "https://raw.githubusercontent.com/YijieYin/connectome_data_prep/refs/heads/main/data/hemibrain/hemibrain_neuron_meta.csv",
        "cell_type",
    ),
    (
        "https://raw.githubusercontent.com/YijieYin/connectome_data_prep/refs/heads/main/data/MANC/manc_meta.csv",
        "combined_type",
    ),
]


def fetch_connectome_names() -> list[str]:
    """Fetch and union cell type names from the four connectome datasets on GitHub."""
    import io
    import urllib.request

    names: set[str] = set()
    for url, col in _CONNECTOME_SOURCES:
        print(f"  Fetching {url.split('/')[-1]} ...")
        with urllib.request.urlopen(url) as r:
            df = pd.read_csv(io.BytesIO(r.read()))
        vals = df[col][~df[col].str.isdigit()].dropna().unique()
        names.update(vals)
        print(f"    {len(vals)} unique values, running total: {len(names)}")
    return sorted(names)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    # Clean LaTeX encoding from cell_type column before any processing
    df["cell_type"] = df["cell_type"].dropna().apply(_clean_latex).reindex(df.index)

    if args.connectome:
        connectome = [
            l.strip()
            for l in Path(args.connectome).read_text().splitlines()
            if l.strip()
        ]
    else:
        print("No --connectome file given; fetching from GitHub ...")
        connectome = fetch_connectome_names()
    print(f"Loaded {len(df)} rows, {len(connectome)} connectome names.")

    unique_cts = df["cell_type"].dropna().unique().tolist()
    print(f"Unique cell_type strings: {len(unique_cts)}")

    # Stage 1: synonym / passthrough lookup
    connectome_set = set(connectome)
    mapping: dict[str, tuple[str | None, str]] = {}
    need_llm = []
    for ct in unique_cts:
        if ct in connectome_set:
            mapping[ct] = (ct, "exact")
        elif canon := exact_lookup(ct):
            mapping[ct] = (canon, "exact")
        else:
            need_llm.append(ct)
    print(f"Exact matches: {len(mapping)}  |  Sending to LLM: {len(need_llm)}")

    # Stage 2: LLM normalisation
    llama_proc = None
    llama_port = args.llama_port
    if need_llm and args.provider == "llama" and args.gguf:
        if not args.gguf:
            raise ValueError("--gguf is required for the llama provider")
        llama_port = llama_port or _find_free_port()
        llama_proc = _start_server(args.llama_bin, args.gguf, llama_port)
        _wait_for_server(llama_port)
        try:  # warmup
            call_llm("T4", ["T4"], "llama", None, llama_port, 16)
        except Exception:
            pass

    try:
        for i, ct in tqdm(enumerate(need_llm), total=len(need_llm)):
            candidates = retrieve_candidates(ct, connectome, args.n_candidates)
            if not candidates:
                mapping[ct] = (None, "null")
                continue
            try:
                raw = call_llm(
                    ct,
                    candidates,
                    args.provider,
                    args.model,
                    llama_port,
                    args.max_tokens,
                )
                canon, src = parse_llm_output(raw, candidates)
                # (c) Substring check for MATCH and MULTI results
                if src == "llm" and isinstance(canon, str):
                    if not llm_match_is_substring(ct, canon):
                        canon, src = None, "null"
                elif src == "multi" and isinstance(canon, list):
                    canon = [n for n in canon if llm_match_is_substring(ct, n)]
                    if not canon:
                        canon, src = None, "null"
            except Exception as e:
                print(f"  [WARN] {ct!r}: {e}")
                canon, src = None, "null"
            mapping[ct] = (canon, src)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(need_llm)} done")
            if args.provider != "llama":
                time.sleep(args.delay)
    finally:
        if llama_proc:
            print("[llama-server] Shutting down")
            llama_proc.terminate()
            llama_proc.wait()

    # Join back + expand MULTI rows
    rows = []
    for _, row in df.iterrows():
        ct = row["cell_type"] if pd.notna(row.get("cell_type")) else None
        canon, src = mapping.get(ct, (None, "null")) if ct else (None, "null")
        if src == "multi" and isinstance(canon, list) and len(canon) > 1:
            for name in canon:
                new_row = row.copy()
                new_row["canonical_ct"] = name
                new_row["ct_source"] = "multi"
                rows.append(new_row)
        else:
            row = row.copy()
            row["canonical_ct"] = (
                canon if not isinstance(canon, list) else (canon[0] if canon else None)
            )
            row["ct_source"] = src
            rows.append(row)
    out_df = pd.DataFrame(rows).reset_index(drop=True)

    out_df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output} ({len(out_df)} rows, {len(df)} original)")
    print(out_df["ct_source"].value_counts().to_string())
    n = out_df["canonical_ct"].notna().sum()
    print(f"Rows with canonical_ct: {n}/{len(out_df)} ({100*n/len(out_df):.1f}%)")


if __name__ == "__main__":
    main()