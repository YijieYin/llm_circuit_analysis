"""
select_types.py — Select source and target cell types for circuit analysis.

Edit the filtering logic below to match your circuit of interest,
then run:
    python select_types.py

This produces circuit_types.json, which is read by the pipeline scripts.

Example filters:
  - By known function:  df.known_function.str.contains("turning")
  - By super class:     meta.super_class == "descending"
  - By cell type name:  meta.cell_type.isin(["DNa01", "DNa03", "DNa04"])
  - Combinations:       function filter & super_class filter
"""

import json
import os
import pandas as pd

# ---- Configuration: edit these paths to match your setup ----
KNOWN_TYPES_CSV = os.path.expanduser(
    "../../known_types_snapshots/known_types_140326.csv"
)
META_CSV = os.path.join(
    "../../interpret_connectome/data/fafb_all_neuron/fafb_all_neuron_meta.csv"
)
OUTPUT_FILE = "circuit_types.json"


def main():
    # Load data
    cell_type_to_function_df = pd.read_csv(KNOWN_TYPES_CSV)
    meta = pd.read_csv(META_CSV, index_col=0, low_memory=False)

    # ================================================================
    # EDIT BELOW: define your source and target cell types
    # ================================================================

    # Example: turning-related descending neurons
    turning_types = cell_type_to_function_df.cell_type[
        cell_type_to_function_df.known_function.str.contains("turning", na=False)
        & cell_type_to_function_df.cell_type.isin(
            meta.cell_type[meta.super_class == "descending"]
        )
    ].values.tolist()

    # Sources and targets can be different sets
    sources = turning_types
    targets = turning_types

    # Alternative: explicit lists
    # sources = ["DNa03", "DNa04"]
    # targets = ["DNa01", "DNa02"]

    # Alternative: different source/target classes
    # sources = meta.cell_type[meta.super_class == "sensory"].unique().tolist()
    # targets = meta.cell_type[meta.super_class == "descending"].unique().tolist()

    # ================================================================
    # END EDIT
    # ================================================================

    # Remove duplicates, sort for reproducibility
    sources = sorted(set(sources))
    targets = sorted(set(targets))

    print(f"Sources ({len(sources)}): {sources}")
    print(f"Targets ({len(targets)}): {targets}")

    # Count valid (source, target) pairs (excluding self-pairs)
    pairs = [(s, t) for s in sources for t in targets if s != t]
    print(f"Total (source, target) pairs: {len(pairs)}")

    output = {"sources": sources, "targets": targets}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()