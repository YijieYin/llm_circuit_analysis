"""
circuit_utils.py — Connectome-specific utilities for the circuit analysis pipeline.

Provider-agnostic LLM utilities (call_llm, parse_llm_response, llama-server
lifecycle, JSON validation, retry logic) live in the `llm_core` package and
are re-exported here for backward compatibility — circuit_step{1,2,3}.py
keep their existing `from circuit_utils import ...` imports unchanged.

This module owns the connectome-specific code:
  - load_connectome_data (FAFB / maleCNS)
  - load_hypotheses
  - find_paths_for_pair
  - compute_net_signed_effect
  - format_paths_by_recipient
  - format_known_to_target
  - compute_shared_intermediates
  - format_net_effect_table
  - build_source_target_matrix
  - add_connectome_args (CLI)
  - add_common_args (CLI: LLM + connectome combined)
  - load_types_file
"""

import argparse
import os
from functools import reduce
from pathlib import Path
import json

import pandas as pd
import scipy as sp

# ---------------------------------------------------------------------------
# Re-export from llm_core so existing scripts (circuit_step*.py) still work
# unchanged. Anyone writing new code is encouraged to import from llm_core
# directly.
# ---------------------------------------------------------------------------

from llm_core import (
    # providers
    PROVIDER_DEFAULTS,
    MODEL_DEFAULTS,
    RESPONSES_API_MODELS,
    resolve_model_config,
    call_llama,
    call_openai,
    call_anthropic,
    call_llm,
    # llama-server
    find_free_port,
    start_llama_server,
    wait_for_llama_server,
    is_llama_server_healthy,
    restart_llama_server,
    setup_llama_server,
    # parsing
    parse_llm_response,
    validate_response,
    # retry
    call_llm_with_retry,
    MAX_RETRIES,
    # CLI
    add_llm_args,
)


# ---------------------------------------------------------------------------
# CLI: connectome-specific args
# ---------------------------------------------------------------------------

def add_connectome_args(parser):
    """Add connectome-specific CLI args (paths, hemisphere, path-search params)."""
    parser.add_argument("--base-path", type=str, default="../../interpret_connectome/")
    parser.add_argument("--known-types-csv", type=str,
                        default="~/Downloads/known_types_snapshots/known_types_100226.csv")
    parser.add_argument("--hypotheses-csv", type=str, default="hypotheses.csv",
                        help="CSV from neuron_interpretation_merge_results.py")
    parser.add_argument("--types-file", type=str, default="circuit_types.json",
                        help="JSON file with 'sources' and 'targets' lists")
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Minimum weight threshold for path filtering")
    return parser


def add_common_args(parser):
    """Add all CLI args used by the circuit_step* scripts (LLM + connectome).

    Backward-compatible wrapper around add_llm_args + add_connectome_args.
    New code can call them separately if only one set is needed.
    """
    add_llm_args(parser)
    add_connectome_args(parser)
    return parser


# ---------------------------------------------------------------------------
# Connectome-specific code below this line is unchanged from the original
# circuit_utils.py — pasted verbatim. Edit here only if the connectome logic
# itself changes.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_connectome_data(base_path, known_types_csv, dataset = 'FAFB'):
    """Load connectome data and build all lookup dicts.
    
    Returns a dict with all the data structures needed by the pipeline.
    """
    from connectome_interpreter import find_paths_of_length, group_paths, filter_paths

    cell_type_to_function_df = pd.read_csv(os.path.expanduser(known_types_csv))
    cell_type_to_function = dict(
        zip(cell_type_to_function_df.cell_type, cell_type_to_function_df.known_function)
    )

    if dataset == 'FAFB':
        inprop = sp.sparse.load_npz(
            os.path.join(base_path, "data", "fafb_all_neuron", "fafb_ad_inprop_all_neuron.npz")
        )
        meta = pd.read_csv(
            os.path.join(base_path, "data", "fafb_all_neuron", "fafb_all_neuron_meta.csv"),
            index_col=0, low_memory=False,
        )
        meta["type_side"] = meta.cell_type + "_" + meta.side

        # Apply known functions
        meta.loc[meta.cell_type.isin(cell_type_to_function), "known_function"] = (
            meta[meta.cell_type.isin(cell_type_to_function)].cell_type.map(cell_type_to_function)
        )
        meta.loc[meta.super_class == "motor", "known_function"] = meta[
            meta.super_class == "motor"
        ].agg(
            lambda x: x["cell_sub_class"]
            if x["cell_type"] == x["known_function"] else x["known_function"],
            axis=1,
        )
        meta["known_function"] = meta["known_function"].fillna(meta.cell_type)
        cell_type_to_function = dict(zip(meta.cell_type, meta.known_function))

        meta["side_known_function"] = meta.side + " " + meta.known_function
        type_side_to_side_function = dict(
            zip(meta.cell_type + "_" + meta.side, meta.side_known_function)
        )
        type_side_to_function = dict(zip(meta.type_side, meta.known_function))
        side_function_to_type = dict(zip(meta.side_known_function, meta.cell_type))
    elif dataset == 'maleCNS': 
        inprop = sp.sparse.load_npz(
            os.path.join(base_path, "data", "maleCNS", "mcns_ad_inprop_all_neuron.npz")
        )
        meta = pd.read_csv(
            os.path.join(base_path, "data", "maleCNS", "mcns_all_neuron_meta.csv"),
            index_col=0, low_memory=False,
        )
        meta['side'] = meta.somaSide.replace({'L': 'left', 'R': 'right', 'M': 'center'})
        meta['type_side'] = meta.cell_type + '_' + meta.side

        # Apply known functions
        meta.loc[meta.cell_type.isin(cell_type_to_function), "known_function"] = (
            meta[meta.cell_type.isin(cell_type_to_function)].cell_type.map(cell_type_to_function)
        )
        # TODO motor neuron functions 
        meta["known_function"] = meta["known_function"].fillna(meta.cell_type)
        cell_type_to_function = dict(zip(meta.cell_type, meta.known_function))
        meta["side_known_function"] = meta.side + " " + meta.known_function
        type_side_to_side_function = dict(
            zip(meta.cell_type + "_" + meta.side, meta.side_known_function)
        )
        type_side_to_function = dict(zip(meta.type_side, meta.known_function))
        side_function_to_type = dict(zip(meta.side_known_function, meta.cell_type))
    
    idx_to_type_side = dict(zip(meta.idx, meta.type_side))
    idx_to_type = dict(zip(meta.idx, meta.cell_type))
    type_to_sign = dict(zip(meta.cell_type, meta.sign))
    type_side_to_sign = dict(zip(meta.type_side, meta.sign))
    type_side_to_type = dict(zip(meta.type_side, meta.cell_type))
    

    return {
        "inprop": inprop,
        "meta": meta,
        "cell_type_to_function_df": cell_type_to_function_df,
        "cell_type_to_function": cell_type_to_function,
        "idx_to_type_side": idx_to_type_side,
        "idx_to_type": idx_to_type,
        "type_to_sign": type_to_sign,
        "type_side_to_sign": type_side_to_sign,
        "type_side_to_type": type_side_to_type,
        "type_side_to_side_function": type_side_to_side_function,
        "type_side_to_function": type_side_to_function,
        "side_function_to_type": side_function_to_type,
    }


def load_hypotheses(hypotheses_csv, add_function=True):
    """Load hypotheses from CSV and return type_to_hypothesis dict."""
    hypotheses = pd.read_csv(hypotheses_csv)
    mask = hypotheses.add_function == add_function
    return dict(zip(
        hypotheses[mask].cell_type,
        hypotheses[mask].inferred_computation,
    ))


# ---------------------------------------------------------------------------
# Path finding & formatting
# ---------------------------------------------------------------------------

def find_paths_for_pair(source, target, data, n_steps, threshold, side):
    """Find all paths from source to target within n_steps.
    
    Returns a concatenated edgelist DataFrame with columns:
        layer, pre, post, weight, pre_type, post_type, pre_sign,
        pre_hypothesis, post_hypothesis
    
    Returns None if no paths found.
    """
    from connectome_interpreter import find_paths_of_length, group_paths, filter_paths

    meta = data["meta"]
    idx_to_type_side = data["idx_to_type_side"]
    type_side_to_sign = data["type_side_to_sign"]
    type_side_to_type = data["type_side_to_type"]
    type_to_hypothesis = data.get("type_to_hypothesis", {})
    # Use ad_inprop if available (adjacency-based), else inprop
    inprop = data.get("ad_inprop", data["inprop"])

    all_paths = []
    for plen in range(1, n_steps + 1):
        p = find_paths_of_length(
            inprop,
            meta.idx[meta.cell_type == source],
            meta.idx[(meta.cell_type == target) & (meta.side == side)],
            plen,
            quiet=True,
        )
        if p is None or p.shape[0] == 0:
            continue
        p = group_paths(p, idx_to_type_side, idx_to_type_side)
        p = filter_paths(p, threshold, quiet=True)
        if p is None or p.shape[0] == 0:
            continue
        p["pre_type"] = p.pre.map(type_side_to_type)
        p["post_type"] = p.post.map(type_side_to_type)
        p["pre_sign"] = p.pre.map(type_side_to_sign)
        p["pre_hypothesis"] = p.pre_type.map(type_to_hypothesis)
        p["post_hypothesis"] = p.post_type.map(type_to_hypothesis)
        all_paths.append(p)

    if not all_paths:
        return None
    return pd.concat(all_paths, ignore_index=True)


def compute_net_signed_effect(paths_df, type_side_to_sign):
    """Compute net signed effect from source to target using
    signed_effective_conn_from_paths.
    
    Returns (excitatory_strength, inhibitory_strength) or None.
    """
    from connectome_interpreter import signed_effective_conn_from_paths

    try:
        result_e, result_i = signed_effective_conn_from_paths(
            paths_df[["layer", "pre", "post", "weight"]].copy(),
            wide=False,
            idx_to_nt=type_side_to_sign,
        )
        e_total = result_e.weight.sum() if len(result_e) > 0 else 0.0
        i_total = result_i.weight.sum() if len(result_i) > 0 else 0.0
        return e_total, i_total
    except Exception as e:
        print(f"  Warning: signed_effective_conn_from_paths failed: {e}")
        return None


def format_paths_by_recipient(paths_df, source, target, type_to_hypothesis):
    """Format path edgelist grouped by layer, then by recipient type_side.
    
    For each recipient at each layer, lists all inputs with weights and signs,
    plus the recipient's hypothesis.
    """
    lines = []
    layers = sorted(paths_df.layer.unique())

    for layer in layers:
        layer_df = paths_df[paths_df.layer == layer]

        # Contextual label
        if layer == min(layers):
            lines.append(f"Layer {layer} (from source {source}):")
        elif layer == max(layers):
            lines.append(f"Layer {layer} (to target {target}):")
        else:
            lines.append(f"Layer {layer}:")

        # Group by recipient (post = type_side)
        for post_ts, grp in layer_df.groupby("post"):
            post_type = grp.iloc[0]["post_type"]
            post_hyp = type_to_hypothesis.get(post_type, "No hypothesis available")
            # Truncate hypothesis for readability
            if isinstance(post_hyp, str) and len(post_hyp) > 500:
                post_hyp = post_hyp[:500] + "..."

            sign_label = ""
            # Get recipient's own sign (E/I identity) if available
            if "post_type" in grp.columns:
                # The recipient's sign as a neuron (its output type)
                # This is in the *next* layer's pre_sign, but we can look it up
                pass  # sign is shown per-input below

            lines.append(f"  → {post_ts} (hypothesis: \"{post_hyp}\"):")

            for _, row in grp.iterrows():
                sign = "Excit" if row["pre_sign"] == 1 else "Inhib"
                lines.append(f"    - from {row['pre']}: w={row['weight']:.5f}, {sign}")

        lines.append("")  # blank line between layers

    return "\n".join(lines)


def format_known_to_target(data, target, side, n_steps, top_n=5):
    """Compute and format the top-N known functional inputs to a target.
    
    Returns a formatted string. Uses signed_conn_by_path_length_data for
    indirect connectivity.
    """
    from connectome_interpreter import signed_conn_by_path_length_data

    meta = data["meta"]
    inprop = data["inprop"]
    type_side_to_sign = data["type_side_to_sign"]
    idx_to_type_side = data["idx_to_type_side"]
    type_side_to_side_function = data["type_side_to_side_function"]

    try:
        ekct, ikct = signed_conn_by_path_length_data(
            inprop,
            meta.idx[(meta.cell_type != meta.known_function) & (meta.cell_type != target)],
            meta.idx[(meta.cell_type == target) & (meta.side == side)],
            n_steps,
            type_side_to_sign,
            idx_to_type_side,
            idx_to_type_side,
        )
        ekct_sum = reduce(lambda a, b: a.add(b, fill_value=0), ekct)
        ikct_sum = reduce(lambda a, b: a.add(b, fill_value=0), ikct)
        diff = ekct_sum.subtract(ikct_sum, fill_value=0)
        diff.index = diff.index.map(type_side_to_side_function)

        # Top N by absolute value
        top = diff.iloc[:, 0].sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)

        lines = [f"Top {top_n} known functional inputs to {target} (indirect connectivity, showing only top {top_n}):"]
        for name, val in top.items():
            sign = "Excitatory" if val > 0 else "Inhibitory"
            lines.append(f"  - {name}: {val:.5f} ({sign})")
        return "\n".join(lines)
    except Exception as e:
        return f"Known inputs to {target}: could not compute ({e})"


def compute_shared_intermediates(data, sources, target, side, n_steps, threshold,
                                  top_k=10):
    """Find intermediate neurons shared across multiple source→target pathways,
    and compute signed effective connectivity from sources to intermediates and
    from intermediates to target using signed_conn_by_path_length_data.
    
    Returns a formatted string, or None if nothing found.
    """
    from connectome_interpreter import el_within_n_steps, signed_conn_by_path_length_data
    from collections import Counter

    meta = data["meta"]
    inprop = data.get("ad_inprop", data["inprop"])
    idx_to_type_side = data["idx_to_type_side"]
    type_side_to_sign = data["type_side_to_sign"]
    type_side_to_type = data["type_side_to_type"]
    type_to_hypothesis = data.get("type_to_hypothesis", {})

    try:
        target_idx = meta.idx[(meta.cell_type == target) & (meta.side == side)]
        if len(target_idx) == 0:
            return None

        target_type_sides = set(
            meta.type_side[(meta.cell_type == target) & (meta.side == side)]
        )

        # Step 1: collect intermediates per source using el_within_n_steps
        source_to_intermediates = {}
        for source in sources:
            if source == target:
                continue
            source_type_sides = set(meta.type_side[meta.cell_type == source])
            el = el_within_n_steps(
                inprop,
                meta.idx[meta.cell_type == source],
                target_idx,
                n_steps,
                threshold,
                idx_to_type_side,
                idx_to_type_side,
                quiet=True,
            )
            if el is None or len(el) == 0:
                continue
            intermediates = set(el.pre) - target_type_sides - source_type_sides
            source_to_intermediates[source] = intermediates

        if not source_to_intermediates:
            return None

        # Step 2: find intermediates shared by 2+ sources
        all_intermediates = Counter()
        for source, intms in source_to_intermediates.items():
            for intm in intms:
                all_intermediates[intm] += 1

        shared = {intm for intm, count in all_intermediates.items() if count >= 2}
        if not shared:
            return None

        # Take top_k by number of sharing sources
        shared_sorted = sorted(shared, key=lambda x: all_intermediates[x], reverse=True)
        shared_top = set(shared_sorted[:top_k])

        # Which sources are relevant (connect to at least one shared intermediate)?
        relevant_sources = [
            s for s, intms in source_to_intermediates.items()
            if intms & shared_top
        ]

        # Step 3: signed effective connectivity from sources to shared intermediates
        # Uses n_steps - 1 because intermediates are between source and target.
        # Note: an intermediate may be 1 hop from one source but 2 from another;
        # n_steps - 1 covers both cases but may also capture indirect paths not
        # strictly in the source→target pathway.
        source_idx = meta.idx[meta.cell_type.isin(relevant_sources)]
        intm_idx = meta.idx[meta.type_side.isin(shared_top)]

        e_s2i_list, i_s2i_list = signed_conn_by_path_length_data(
            inprop,
            source_idx,
            intm_idx,
            n_steps - 1,
            type_side_to_sign,
            idx_to_type_side,
            idx_to_type_side,
        )
        # Aggregate across path lengths
        e_s2i = reduce(lambda a, b: a.add(b, fill_value=0), e_s2i_list) if e_s2i_list else pd.DataFrame()
        i_s2i = reduce(lambda a, b: a.add(b, fill_value=0), i_s2i_list) if i_s2i_list else pd.DataFrame()

        # Step 4: signed effective connectivity from shared intermediates to target
        e_i2t_list, i_i2t_list = signed_conn_by_path_length_data(
            inprop,
            intm_idx,
            target_idx,
            n_steps - 1,
            type_side_to_sign,
            idx_to_type_side,
            idx_to_type_side,
        )
        e_i2t = reduce(lambda a, b: a.add(b, fill_value=0), e_i2t_list) if e_i2t_list else pd.DataFrame()
        i_i2t = reduce(lambda a, b: a.add(b, fill_value=0), i_i2t_list) if i_i2t_list else pd.DataFrame()

        # Step 5: format output
        lines = [
            f"Shared intermediates (neurons in pathways from 2+ sources to {target}, "
            f"showing top {top_k} of {len(shared)} found).",
            f"NOTE: This is a SUBSET of the circuit for additional context. Do not focus "
            f"exclusively on these intermediates — the pairwise pathway analyses above are "
            f"the primary evidence.",
            "",
        ]

        for intm_ts in shared_sorted[:top_k]:
            intm_type = type_side_to_type.get(intm_ts, intm_ts)
            n_sharing = all_intermediates[intm_ts]
            sharing_sources = [
                s for s, intms in source_to_intermediates.items()
                if intm_ts in intms
            ]
            hyp = type_to_hypothesis.get(intm_type, "")
            if isinstance(hyp, str) and len(hyp) > 300:
                hyp = hyp[:300] + "..."

            lines.append(f"  {intm_ts} (shared by {n_sharing} sources):")
            if hyp:
                lines.append(f"    Hypothesis: {hyp}")

            # Source → intermediate connectivity (sign-specific)
            excit_inputs = []
            inhib_inputs = []
            for source in sharing_sources:
                # Find source type_sides and look up in the matrices
                source_tss = list(meta.type_side[meta.cell_type == source].unique())
                for s_ts in source_tss:
                    e_val = 0.0
                    i_val = 0.0
                    if not e_s2i.empty and s_ts in e_s2i.index and intm_ts in e_s2i.columns:
                        e_val = e_s2i.loc[s_ts, intm_ts]
                    if not i_s2i.empty and s_ts in i_s2i.index and intm_ts in i_s2i.columns:
                        i_val = i_s2i.loc[s_ts, intm_ts]
                    if e_val > 0:
                        excit_inputs.append(f"{s_ts} ({e_val:.5f})")
                    if i_val > 0:
                        inhib_inputs.append(f"{s_ts} ({i_val:.5f})")

            lines.append(f"    Signed effective connectivity from sources to {intm_ts}:")
            if excit_inputs:
                lines.append(f"      Effective excitation: {', '.join(excit_inputs)}")
            if inhib_inputs:
                lines.append(f"      Effective inhibition: {', '.join(inhib_inputs)}")
            if not excit_inputs and not inhib_inputs:
                lines.append(f"      (no significant effective connectivity found)")

            # Intermediate → target connectivity (sign-specific)
            e_to_target = 0.0
            i_to_target = 0.0
            # Target might have multiple type_sides but we filter to one side
            for t_ts in target_type_sides:
                if not e_i2t.empty and intm_ts in e_i2t.index and t_ts in e_i2t.columns:
                    e_to_target += e_i2t.loc[intm_ts, t_ts]
                if not i_i2t.empty and intm_ts in i_i2t.index and t_ts in i_i2t.columns:
                    i_to_target += i_i2t.loc[intm_ts, t_ts]

            lines.append(f"    Signed effective connectivity from {intm_ts} to {target}:")
            parts = []
            if e_to_target > 0:
                parts.append(f"Effective excitation: {e_to_target:.5f}")
            if i_to_target > 0:
                parts.append(f"Effective inhibition: {i_to_target:.5f}")
            if parts:
                lines.append(f"      {'; '.join(parts)}")
            else:
                lines.append(f"      (no significant effective connectivity found)")

            lines.append("")  # blank between intermediates

        return "\n".join(lines)

    except Exception as e:
        print(f"  Warning: shared intermediates computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_net_effect_table(step1_results):
    """Build a compact (source, net_sign, strength) table from step 1 results.
    
    Extracts the net_effect metadata stored in step 1 records.
    Returns a formatted string for inclusion in the stage 2 prompt.
    """
    lines = ["Raw data summary (source → target net signed effects):"]
    for rec in step1_results:
        source = rec.get("source", "?")
        net_excit = rec.get("net_excit")
        net_inhib = rec.get("net_inhib")
        critic_flag = rec.get("critic_flag", "")

        if net_excit is not None and net_inhib is not None:
            net = net_excit - net_inhib
            sign = "Excitatory" if net > 0 else "Inhibitory"
            line = f"  {source}: {sign} (excit={net_excit:.5f}, inhib={net_inhib:.5f}, net={net:.5f})"
        else:
            line = f"  {source}: net effect not computed"

        if critic_flag:
            line += f"  [CRITIC FLAG: {critic_flag}]"
        lines.append(line)
    return "\n".join(lines)


def build_source_target_matrix(step1_dir):
    """Build a source×target matrix of net signed effects from step 1 results.
    
    Returns a formatted string showing the matrix.
    """
    step1_dir = Path(step1_dir)
    records = []
    for f in sorted(step1_dir.glob("pairwise_chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "source" in rec and "target" in rec:
                    records.append(rec)

    if not records:
        return "Source-target matrix: no data available"

    # Build matrix
    sources = sorted(set(r["source"] for r in records))
    targets = sorted(set(r["target"] for r in records))

    lines = ["Source → Target net signed effect matrix:"]
    # Header
    header = "  " + " | ".join([f"{'':>10}"] + [f"{t:>10}" for t in targets])
    lines.append(header)
    lines.append("  " + "-" * len(header))

    for source in sources:
        row = [f"{source:>10}"]
        for target in targets:
            # Find the record
            matching = [r for r in records
                        if r["source"] == source and r["target"] == target]
            if matching:
                rec = matching[0]
                net_e = rec.get("net_excit")
                net_i = rec.get("net_inhib")
                if net_e is not None and net_i is not None:
                    net = net_e - net_i
                    sign_char = "+" if net > 0 else "-"
                    row.append(f"{sign_char}{abs(net):.4f}".rjust(10))
                else:
                    row.append("?".rjust(10))
            elif source == target:
                row.append("—".rjust(10))
            else:
                row.append("no path".rjust(10))
        lines.append("  " + " | ".join(row))

    return "\n".join(lines)



def load_types_file(path):
    """Load the circuit_types.json file."""
    with open(path) as f:
        types = json.load(f)
    sources = types["sources"]
    targets = types["targets"]
    print(f"Sources ({len(sources)}): {sources}")
    print(f"Targets ({len(targets)}): {targets}")
    return sources, targets