"""
merge_extraction_results.py — Merge chunk JSONL outputs into a single file/dataframe.

Usage:
    python merge_extraction_results.py --results-dir ./results --output extracted.jsonl
    python merge_extraction_results.py --results-dir ./results --output extracted.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--output", default="extracted.jsonl",
                   help="Output file (.jsonl or .csv)")
    p.add_argument("--n-chunks", type=int, default=None,
                   help="Expected number of chunks (for completeness check)")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    chunk_files = sorted(results_dir.glob("results_chunk_*.jsonl"))
    if not chunk_files:
        print(f"No chunk files found in {results_dir}")
        return

    print(f"Found {len(chunk_files)} chunk files:")
    for f in chunk_files:
        print(f"  {f}")

    if args.n_chunks and len(chunk_files) < args.n_chunks:
        print(f"WARNING: Expected {args.n_chunks} chunks but only found {len(chunk_files)}")

    # Load all records
    records = []   # valid cell-type extraction entries
    empty   = []   # papers that were processed but yielded 0 cell types
    errors  = []   # records with an error key or failed JSON parse

    for f in chunk_files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  Bad line in {f}: {e}")
                    continue

                if "error" in rec:
                    errors.append(rec)
                elif "cell_types_found" in rec:
                    # Logged by process_paper when a paper returned 0 cell types
                    empty.append(rec)
                else:
                    records.append(rec)

    # ---- Summary ----
    print(f"\nTotal cell-type entries: {len(records)}")
    print(f"Papers with 0 cell types found: {len(empty)}")
    print(f"Errors/failures: {len(errors)}")

    if errors:
        failed_papers = [e.get("paper_title") or e.get("paper_id", "?") for e in errors]
        print("Failed papers:")
        for p in failed_papers:
            print(f"  {p}")
        error_file = results_dir / "errors.jsonl"
        with open(error_file, "w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
        print(f"Errors saved to {error_file}")

    if not records:
        print("No valid records to merge.")
        return

    # ---- Deduplicate by (paper_id, cell_type) — keep last ----
    # Unlike the neuron script (one result per cell type globally), extraction
    # can legitimately produce the same cell type from different papers.
    seen = {}
    for r in records:
        key = (r.get("paper_id", ""), r.get("cell_type", ""))
        seen[key] = r
    records = list(seen.values())
    print(f"Unique (paper, cell_type) entries after dedup: {len(records)}")
    print(f"Unique papers represented: {len({r.get('paper_id') for r in records})}")
    print(f"Unique cell types (across all papers): {len({r.get('cell_type') for r in records})}")

    # ---- Save ----
    out = Path(args.output)
    if out.suffix == ".csv":
        df = pd.json_normalize(records)
        df.to_csv(out, index=False)
        print(f"\nSaved CSV  → {out}  ({len(df)} rows × {len(df.columns)} cols)")
    else:
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\nSaved JSONL → {out}  ({len(records)} records)")


if __name__ == "__main__":
    main()