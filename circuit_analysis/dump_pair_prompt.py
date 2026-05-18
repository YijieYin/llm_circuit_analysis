"""
dump_pair_prompt.py — Dump the circuit-step-1 prompt for a single (source, target)
pair, for use in a Colab notebook or pasted into Claude.ai / ChatGPT manually.

This is the manual-friendly counterpart to circuit_step1_pairwise.py. It loads
the same connectome data, builds the same prompt the full pipeline would build
for one pair, and writes it to a markdown file. No LLM call is made.

Usage:
    python dump_pair_prompt.py \\
        --source LC4 --target DNp03 \\
        --base-path ~/interpret_connectome \\
        --known-types-csv ~/known_types_snapshots/known_types_140526.csv \\
        --hypotheses-csv ~/llm_circuit_analysis/neuron_interpretation/hypotheses.csv \\
        --output prompts/LC4__DNp03.md
"""

import argparse
import sys
from pathlib import Path

import dotenv

from circuit_utils import (
    add_connectome_args,
    load_connectome_data,
    load_hypotheses,
)
from circuit_step1_pairwise import (
    build_prompts_for_pair,   # ← added in the edit below
    REQUIRED_KEYS,
)
from llm_core import write_prompt_file, safe_filename


def main():
    parser = argparse.ArgumentParser(description="Dump one circuit-step-1 prompt")
    parser.add_argument("--source", required=True, help="Source cell type")
    parser.add_argument("--target", required=True, help="Target cell type")
    parser.add_argument("--output", default=None,
                        help="Output .md path (default: prompts/{source}__{target}.md)")
    parser.add_argument("--dataset", type=str, default="FAFB",
                        choices=["FAFB", "maleCNS"])
    add_connectome_args(parser)
    args = parser.parse_args()

    dotenv.load_dotenv()

    print(f"Loading connectome data ({args.dataset})...")
    data = load_connectome_data(args.base_path, args.known_types_csv, args.dataset)
    data["type_to_hypothesis"] = load_hypotheses(args.hypotheses_csv)

    print(f"Building prompt for {args.source} → {args.target}...")
    prompts = build_prompts_for_pair(args.source, args.target, data, args)
    if prompts is None:
        print(f"No paths found from {args.source} to {args.target} within "
              f"n_steps={args.n_steps}, threshold={args.threshold}.")
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path(
        "prompts") / f"{safe_filename(args.source)}__{safe_filename(args.target)}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    notes = (
        f"Generated for pair: **{args.source} → {args.target}** "
        f"({args.side} hemisphere, n_steps={args.n_steps}, "
        f"threshold={args.threshold}).\n\n"
        f"Net signed effect (raw data): {prompts.get('net_effect_line', 'n/a')}"
    )
    write_prompt_file(
        out_path,
        item_id=f"{args.source} → {args.target}",
        system_prompt=prompts["system"],
        user_prompt=prompts["user"],
        required_keys=REQUIRED_KEYS,
        extra_notes=notes,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()