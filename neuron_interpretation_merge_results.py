"""
merge_results.py — Merge chunk JSONL outputs into a single file/dataframe.

Usage:
    python merge_results.py --results-dir ./results --output hypotheses.jsonl
    python merge_results.py --results-dir ./results --output hypotheses.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--output", default="hypotheses.jsonl",
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
    records = []
    errors = []
    for f in chunk_files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "error" in rec and "inferred_computation" not in rec:
                        errors.append(rec)
                    else:
                        records.append(rec)
                except json.JSONDecodeError as e:
                    print(f"  Bad line in {f}: {e}")

    print(f"\nTotal records: {len(records)}")
    print(f"Errors/failures: {len(errors)}")

    if errors:
        error_cells = [e.get("cell_type", "?") for e in errors]
        print(f"Failed cells: {error_cells}")
        # Save errors separately
        error_file = results_dir / "errors.jsonl"
        with open(error_file, "w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
        print(f"Errors saved to {error_file}")

    if not records:
        print("No valid records to merge.")
        return

    # Deduplicate by cell_type (keep last)
    seen = {}
    for r in records:
        seen[r.get("cell_type", "")] = r
    records = list(seen.values())
    print(f"Unique cell types: {len(records)}")

    out = Path(args.output)
    if out.suffix == ".csv":
        df = pd.json_normalize(records)
        df.to_csv(out, index=False)
        print(f"Saved CSV to {out}")
    else:
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved JSONL to {out}")


if __name__ == "__main__":
    main()