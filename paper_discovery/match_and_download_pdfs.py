"""
match_and_download_pdfs.py — Match discovered papers against local PDFs on HPC
and download missing open-access PDFs.

Designed to run AFTER paper_discovery_hpc.py. Reads discovered_papers.csv,
scans an existing PDF directory, matches by DOI then by fuzzy title, then
downloads any unmatched papers that have an open_access_pdf URL.

Usage:
    python match_and_download_pdfs.py \\
        --discovered-csv output/discovered_papers.csv \\
        --pdfs-dir /cephfs2/yyin/llm_circuit_analysis/pdfs \\
        --output-csv output/discovered_papers_with_local.csv

Outputs:
    discovered_papers_with_local.csv  — input + local_pdf_path/match_method/download_status
    pdf_scan_cache.json               — cached PDF metadata (mtime-keyed, fast re-runs)
    download_log.csv                  — compact per-paper outcome log
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import requests
import requests_cache
from rapidfuzz import fuzz, process
from tqdm import tqdm


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)



UNPAYWALL_EMAIL = "yy432@cam.ac.uk"

# Cached session for Unpaywall lookups — re-runs free
_unpaywall_session = None


def get_unpaywall_session(cache_path):
    global _unpaywall_session
    if _unpaywall_session is None:
        _unpaywall_session = requests_cache.CachedSession(
            str(cache_path), backend="sqlite",
            expire_after=60 * 60 * 24 * 30,
        )
    return _unpaywall_session


def get_unpaywall_url(doi, cache_path):
    """Query Unpaywall for an OA PDF URL. Returns URL or None."""
    if not doi or pd.isna(doi):
        return None
    sess = get_unpaywall_session(cache_path)
    try:
        r = sess.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": UNPAYWALL_EMAIL},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        loc = data.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        return None


def normalize_pdf_url(url):
    """Transform known landing-page URLs into direct PDF URLs."""
    if not url:
        return url
    # bioRxiv / medRxiv: append .full.pdf if not already a PDF
    if ("biorxiv.org" in url or "medrxiv.org" in url) and "/content/" in url:
        if not url.endswith(".pdf"):
            return url.rstrip("/") + ".full.pdf"
    # arXiv: /abs/X → /pdf/X
    if "arxiv.org/abs/" in url:
        return url.replace("/abs/", "/pdf/")
    return url

# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--discovered-csv", required=True,
                   help="Output CSV from paper_discovery_hpc.py")
    p.add_argument("--pdfs-dir", required=True,
                   help="Directory to scan for existing local PDFs (recursive)")
    p.add_argument("--download-dir", default=None,
               help="Where new downloads go (default: same as --pdfs-dir)")
    p.add_argument("--output-csv", required=True,
                   help="Output CSV with match + download columns")
    p.add_argument("--scan-cache", default=None,
                   help="Path for PDF scan cache (default: alongside output)")
    p.add_argument("--similarity-threshold", type=int, default=85,
                   help="rapidfuzz threshold 0-100 for title matching (default 85)")
    p.add_argument("--n-workers", type=int, default=8,
                   help="Concurrent PDF download threads (default 8)")
    p.add_argument("--skip-download", action="store_true",
                   help="Match only; don't download missing PDFs")
    p.add_argument("--max-downloads", type=int, default=None,
                   help="Limit number of downloads (for testing)")
    return p.parse_args()


# ── PDF scanning (cached by mtime) ──────────────────────────────────────────
def extract_pdf_metadata(path):
    """Return {'path', 'title', 'doi'} or None on failure."""
    try:
        doc = fitz.open(path)
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip()

        doi = None
        if doc.page_count > 0:
            text = doc[0].get_text()[:5000]
            m = DOI_RE.search(text)
            if m:
                doi = m.group(0).rstrip(".,);")

        doc.close()
        if not title:
            title = Path(path).stem.replace("_", " ").replace("-", " ")
        return {"path": str(path), "title": title, "doi": doi}
    except Exception as e:
        print(f"  [scan] Failed: {path}: {e}")
        return None


def scan_pdfs_dir(pdfs_dir, cache_path):
    """Recursively scan; reuse cached metadata for unchanged files (by mtime)."""
    pdfs_dir = Path(pdfs_dir)
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    cached_files = cache.get("files", {})

    all_pdfs = sorted(pdfs_dir.rglob("*.pdf"))
    print(f"Found {len(all_pdfs)} PDFs in {pdfs_dir}")

    fresh = {}
    to_process = []
    for p in all_pdfs:
        key = str(p)
        mtime = p.stat().st_mtime
        prev = cached_files.get(key)
        if prev and prev.get("mtime") == mtime:
            fresh[key] = prev
        else:
            to_process.append((p, mtime))

    print(f"Re-scanning {len(to_process)} new/changed PDFs (cached: {len(fresh)})")
    for p, mtime in tqdm(to_process, desc="Scanning PDFs"):
        meta = extract_pdf_metadata(p)
        if meta:
            fresh[str(p)] = {**meta, "mtime": mtime}

    cache["files"] = fresh
    cache["scanned_at"] = time.time()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(cache_path)

    return pd.DataFrame([
        {"path": k, "title": v.get("title", ""), "doi": v.get("doi")}
        for k, v in fresh.items()
    ])


# ── Matching ────────────────────────────────────────────────────────────────
def _norm_doi(s):
    return str(s).upper().rstrip(".,);").strip()


def match_papers(discovered_df, local_pdfs_df, threshold=85):
    matched = discovered_df.copy()
    matched["local_pdf_path"] = None
    matched["match_method"] = None

    # DOI index (built once)
    doi_idx = {}
    for _, r in local_pdfs_df.iterrows():
        if pd.notna(r.get("doi")):
            doi_idx[_norm_doi(r["doi"])] = r["path"]

    local_titles = local_pdfs_df["title"].fillna("").tolist()
    local_paths = local_pdfs_df["path"].tolist()

    n_doi = n_title = 0
    for idx, paper in tqdm(matched.iterrows(), total=len(matched), desc="Matching"):
        if pd.notna(paper.get("doi")):
            k = _norm_doi(paper["doi"])
            if k in doi_idx:
                matched.at[idx, "local_pdf_path"] = doi_idx[k]
                matched.at[idx, "match_method"] = "doi"
                n_doi += 1
                continue
        if paper.get("title"):
            hit = process.extractOne(
                paper["title"], local_titles,
                scorer=fuzz.ratio, score_cutoff=threshold,
            )
            if hit:
                _, score, i = hit
                matched.at[idx, "local_pdf_path"] = local_paths[i]
                matched.at[idx, "match_method"] = f"title_sim_{score:.0f}"
                n_title += 1

    n_matched = matched["local_pdf_path"].notna().sum()
    print(f"Matched {n_matched}/{len(matched)} ({n_doi} by DOI, {n_title} by fuzzy title)")
    return matched


# ── Downloads ───────────────────────────────────────────────────────────────
def create_filename_from_title(title, paper_id, max_len=120):
    base = title or paper_id or "untitled"
    slug = re.sub(r"[^\w\s-]", "", base).strip().lower()
    slug = re.sub(r"[-\s]+", "_", slug)[:max_len]
    suffix = (paper_id or "")[:8]
    return f"{slug}_{suffix}.pdf".replace("__", "_")


def download_pdf(url, save_path, timeout=30, max_size_mb=50):
    """Returns (True, status_str) or (False, reason)."""
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": USER_AGENT},
                         allow_redirects=True, stream=True)
        if r.status_code != 200:
            return False, f"http_{r.status_code}"

        # Read with size cap
        chunks = []
        total = 0
        cap = max_size_mb * 1024 * 1024
        for chunk in r.iter_content(chunk_size=64 * 1024):
            chunks.append(chunk)
            total += len(chunk)
            if total > cap:
                return False, "too_large"
        content = b"".join(chunks)

        # Validate: %PDF- magic bytes anywhere in the first 1024 bytes
        # (some servers prefix valid PDFs with whitespace/BOM)
        if b"%PDF-" not in content[:1024]:
            # Save for debugging — first 200 bytes
            head = content[:200].decode("utf-8", errors="replace")
            return False, f"not_pdf:{head[:100]!r}"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(content)
        return True, "ok"
    except requests.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, f"err:{type(e).__name__}"


def download_missing(matched_df, download_dir, n_workers=8,
                      max_downloads=None, unpaywall_cache_path=None):
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    matched_df = matched_df.copy()
    matched_df["download_status"] = pd.Series(dtype="object")
    matched_df["download_source"] = pd.Series(dtype="object")

    work = []
    n_queued = 0
    for idx, row in matched_df.iterrows():
        if pd.notna(row.get("local_pdf_path")):
            matched_df.at[idx, "download_status"] = "already_local"
            continue
        s2_url = row.get("open_access_pdf")
        doi    = row.get("doi")
        if (pd.isna(s2_url) or not s2_url) and (pd.isna(doi) or not doi):
            matched_df.at[idx, "download_status"] = "no_oa_url"
            continue
        if max_downloads is not None and n_queued >= max_downloads:
            matched_df.at[idx, "download_status"] = "skipped_limit"
            continue
        work.append((idx, dict(row)))
        n_queued += 1

    print(f"Queued {len(work)} downloads")

    def task(idx_row):
        idx, row = idx_row
        fname = create_filename_from_title(row.get("title", ""), str(row.get("paper_id", "")))
        out = download_dir / fname
        if out.exists():
            return idx, str(out), "already_downloaded", "cache"

        # Build ordered list of candidate URLs to try
        candidates = []
        s2 = row.get("open_access_pdf")
        if pd.notna(s2) and s2:
            candidates.append(("s2", normalize_pdf_url(s2)))
        doi = row.get("doi")
        if pd.notna(doi) and doi:
            up = get_unpaywall_url(doi, unpaywall_cache_path)
            if up and up != s2:
                candidates.append(("unpaywall", normalize_pdf_url(up)))

        if not candidates:
            return idx, None, "no_oa_url", None

        last_reason = "no_candidates"
        for source, url in candidates:
            time.sleep(0.05)
            ok, reason = download_pdf(url, out)
            if ok:
                return idx, str(out), "downloaded", source
            last_reason = f"{source}:{reason}"

        return idx, None, f"download_failed:{last_reason}", None

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for idx, path, status, source in tqdm(
            ex.map(task, work), total=len(work), desc="Downloading"
        ):
            if path is not None:
                matched_df.at[idx, "local_pdf_path"] = path
            matched_df.at[idx, "download_status"] = status
            matched_df.at[idx, "download_source"] = source

    counts = matched_df["download_status"].value_counts().head(15).to_dict()
    print(f"Download outcomes (top 15): {counts}")
    return matched_df


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    scan_cache = Path(args.scan_cache) if args.scan_cache else out_csv.with_name("pdf_scan_cache.json")

    print(f"Loading {args.discovered_csv}")
    discovered = pd.read_csv(args.discovered_csv)
    print(f"  {len(discovered)} discovered papers\n")

    print(f"== Step 1: Scan {args.pdfs_dir} ==")
    local_pdfs = scan_pdfs_dir(args.pdfs_dir, scan_cache)
    print(f"  {len(local_pdfs)} local PDFs cataloged\n")

    print(f"== Step 2: Match ==")
    matched = match_papers(discovered, local_pdfs, threshold=args.similarity_threshold)
    print()

    if args.skip_download:
        matched.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} (matched only, no downloads)")
        return

    print(f"== Step 3: Download missing OA PDFs ==")
    final = download_missing(
        matched, args.download_dir or args.pdfs_dir,
        n_workers=args.n_workers,
        max_downloads=args.max_downloads,
        unpaywall_cache_path=out_csv.with_name("unpaywall_cache.sqlite"),
    )

    final.to_csv(out_csv, index=False)
    log_cols = ["paper_id", "title", "doi", "open_access_pdf",
                "local_pdf_path", "match_method", "download_status"]
    log_cols = [c for c in log_cols if c in final.columns]
    final[log_cols].to_csv(out_csv.with_name("download_log.csv"), index=False)

    total = len(final)
    have_pdf = final["local_pdf_path"].notna().sum()
    print(f"\n== Coverage ==")
    print(f"  Total papers:  {total}")
    print(f"  Have PDF:      {have_pdf} ({100*have_pdf/total:.1f}%)")
    print(f"  Output:        {out_csv}")


if __name__ == "__main__":
    main()