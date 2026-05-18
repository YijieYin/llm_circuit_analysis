"""
merge_chunks.py — Merge chunk JSONL outputs from any pipeline step.

Replaces:
  - function_extraction_from_papers/merge_extraction_results.py
  - neuron_interpretation/neuron_interpretation_merge_results.py

Usage:
  python merge_chunks.py --kind extraction --results-dir ./results --output extracted.jsonl
  python merge_chunks.py --kind neuron     --results-dir ./results --output hypotheses.csv

  # Check that all expected chunks landed:
  python merge_chunks.py --kind extraction --results-dir ./results --output out.csv --n-chunks 8
"""

import argparse
import json
from pathlib import Path

import pandas as pd


# Per-kind configuration. To add a new pipeline step (e.g. circuit step 1
# pairwise), append an entry here.
KIND_CONFIG = {
    "extraction": {
        # function_extraction_from_paper.py emits one record per (paper, cell_type)
        # so dedup on the pair. Empty-paper records are tagged with cell_types_found.
        "dedup_keys":   ("paper_id", "cell_type"),
        "empty_marker": "cell_types_found",
        "id_field":     "paper_id",
    },
    "neuron": {
        # neuron_interpretation.py emits one record per (cell_type, add_function setting)
        "dedup_keys":   ("cell_type", "add_function"),
        "empty_marker": None,
        "id_field":     "cell_type",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=sorted(KIND_CONFIG),
                   help="Which pipeline step's results to merge")
    p.add_argument("--results-dir", default="./results",
                   help="Directory containing results_chunk_*.jsonl files")
    p.add_argument("--output", required=True,
                   help="Output path. .jsonl writes JSONL; .csv writes a flattened CSV.")
    p.add_argument("--n-chunks", type=int, default=None,
                   help="If set, warn when fewer chunk files are present than expected")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = KIND_CONFIG[args.kind]
    results_dir = Path(args.results_dir)
    chunk_files = sorted(results_dir.glob("results_chunk_*.jsonl"))

    if not chunk_files:
        print(f"No chunk files found in {results_dir}")
        return
    print(f"Found {len(chunk_files)} chunk files in {results_dir}")

    if args.n_chunks and len(chunk_files) < args.n_chunks:
        print(f"WARNING: expected {args.n_chunks} chunks, found {len(chunk_files)}")

    records, empty, errors = [], [], []
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
                elif cfg["empty_marker"] and cfg["empty_marker"] in rec:
                    empty.append(rec)
                else:
                    records.append(rec)

    print(f"\nValid records: {len(records)}")
    if cfg["empty_marker"]:
        print(f"Empty (processed, 0 results): {len(empty)}")
    print(f"Errors: {len(errors)}")

    if errors:
        ids = [e.get(cfg["id_field"], "?") for e in errors]
        preview = ids[:10] + (["..."] if len(ids) > 10 else [])
        print(f"Failed: {preview}")
        error_file = results_dir / "errors.jsonl"
        with open(error_file, "w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
        print(f"Errors saved to {error_file}")

    if not records:
        print("No valid records to merge.")
        return

    # Dedup by configured key; keep last (later chunks override earlier on collision)
    seen = {}
    for r in records:
        key = tuple(r.get(k, "") for k in cfg["dedup_keys"])
        seen[key] = r
    records = list(seen.values())
    print(f"Unique records after dedup by {cfg['dedup_keys']}: {len(records)}")

    # Report on uniqueness of the main id field for context
    if cfg["id_field"]:
        unique_ids = {r.get(cfg["id_field"]) for r in records}
        print(f"Unique {cfg['id_field']}: {len(unique_ids)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df = pd.json_normalize(records)
        df.to_csv(out, index=False)
        print(f"Saved CSV  → {out}  ({len(df)} rows × {len(df.columns)} cols)")
    else:
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved JSONL → {out}  ({len(records)} records)")


if __name__ == "__main__":
    main()