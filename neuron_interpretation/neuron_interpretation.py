"""
interpret.py — Neuron interpretation script for HPC SLURM array jobs.

Usage:
    python interpret.py --chunk-id 0 --n-chunks 4 [options]

The script:
  1. Loads connectome data
  2. Splits the dns list into N equal chunks
  3. Processes chunk --chunk-id
  4. Optionally starts a local llama-server (llama.cpp) per job
  5. Saves results incrementally to results/results_chunk_{chunk_id}.jsonl

Provider options:
  --provider llama   : local llama-server (llama.cpp OpenAI-compatible API)
  --provider openai  : OpenAI API (needs OPENAI_API_KEY)
  --provider anthropic: Anthropic API (needs ANTHROPIC_API_KEY)
"""

import argparse
import json
import os
from functools import reduce
from pathlib import Path

import dotenv
import pandas as pd
import scipy as sp
from tqdm import tqdm

from llm_core import (
    PROVIDER_DEFAULTS, resolve_model_config,
    call_llm, call_llm_with_retry,
    setup_llama_server, restart_llama_server, is_llama_server_healthy,
    parse_llm_response, validate_response,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Neuron interpretation batch job")
    p.add_argument("--chunk-id", type=int, required=True,
                   help="Index of this chunk (0-based)")
    p.add_argument("--n-chunks", type=int, required=True,
                   help="Total number of chunks")
    p.add_argument("--provider", type=str, default="llama",
                   choices=["llama", "openai", "anthropic"],
                   help="LLM provider to use")
    p.add_argument("--model", type=str, default=None,
                   help="Model name (overrides provider default)")
    p.add_argument("--gguf", type=str,
                   default="/cephfs2/yyin/huggingface/hub/qwen35_gguf/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf",
                   help="Path to GGUF file (llama provider only)")
    p.add_argument("--llama-bin", type=str,
                   default=os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
                   help="Path to llama-server binary")
    p.add_argument("--base-path", type=str,
                   default="../../interpret_connectome/",
                   help="Path to interpret_connectome data directory")
    p.add_argument("--known-types-csv", type=str,
                   default="~/Downloads/known_types_snapshots/known_types_100226.csv",
                   help="Path to known_types CSV")
    p.add_argument("--output-dir", type=str, default="./results",
                   help="Directory to save output JSONL files")
    p.add_argument("--n-steps", type=int, default=4,
                   help="Number of synaptic steps for circuit extraction")
    p.add_argument("--top-n", type=int, default=15,
                   help="Top N connections to include")
    p.add_argument("--add-function", action=argparse.BooleanOptionalAction, default=True,
                   help="Map cell types to known functions (--no-add-function to disable)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="Anthropic extended thinking budget (tokens)")
    p.add_argument("--reasoning-effort", type=str, default=None,
                   choices=["low", "medium", "high"],
                   help="OpenAI reasoning effort (overrides model default)")
    p.add_argument("--verbosity", type=str, default=None,
                   choices=["low", "medium", "high"],
                   help="OpenAI Responses API verbosity (gpt-5/gpt-5.4 only)")
    p.add_argument("--llama-port", type=int, default=None,
                   help="Port for llama-server (auto-assigned if None)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip already-processed cell types (--no-resume to disable)")
    p.add_argument('--side', type=str, default='left', choices=['left', 'right'],
                   help="Which side of the brain to analyze (left or right)")
    p.add_argument("--dump-prompts", type=str, default=None,
               help="Instead of calling the LLM, write one .md prompt file "
                    "per cell type to this directory and exit.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Prompts (copied from notebook)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a neuroscience expert specializing in Drosophila neural circuits, computational neuroscience, and circuit mechanisms.

Your task is to infer the COMPUTATIONAL ROLE and ADAPTIVE, PROBLEM-SOLVING FUNCTION of a cell type by analyzing its contextual receptive field—
how it integrates inputs from functionally characterized neurons to solve a specific behavioral or sensorimotor problem.

KEY PRINCIPLE: Move beyond connectivity description to mechanistic inference of what problem this cell type solves.
Examples of the shift:
  ❌ "Receives visual motion + motor feedback"
  ✓ "Integrates visual optic flow with motor command to distinguish self-motion from external disturbance"
  ❌ "Intermediate neuron receives excitation and inhibition"
  ✓ "Intermediate filters sensory input based on behavioral state—activated during behavior to suppress irrelevant inputs"

You will be given:
1. The cell type name
2. Signed effective connectivity: sensory neurons → cell type
3. Signed effective connectivity: functionally characterized neurons → cell type
4. Intermediate neurons directly upstream:
   - Their connectivity strength to the cell type
   - Signed effective connectivity from characterized inputs to these intermediates
   - Sign of each intermediate (excitatory/inhibitory)
5. Signed effective connectivity: cell type → functionally characterized neurons

Your analysis should:
1. **Infer intermediate computations**: For each intermediate neuron, what role does it play in this specific circuit?
   What transformation or operation does it perform on its inputs?
   Why might this circuit include this intermediate rather than direct connections?

2. **Integrate across input categories**: Systematically analyze how different inputs work together. Recognize patterns:
   Which inputs converge? Consider sign combinations: what does combinations of excitation/inhibition from different sources enable?
   Synthesize distinct input types into a coherent functional picture. Reason about these through the lens of what the cell needs to *compute*, not abstract templates.

3. **Identify the computational abstraction**: What higher-level problem does this circuit solve? What adaptive function emerges from how inputs are combined?
   (e.g. detecting self-motion vs. external motion, gain-of-field vs. local features, state-dependent gating)

4. **Generate multiple hypotheses if justified**: If the connectivity supports distinct mechanistic interpretations, present them
   as alternatives with marked confidence. Distinguish between [grounded] claims (directly supported by connectivity patterns),
   [likely] claims (coherent logic), and [speculative] claims (requiring additional mechanisms or less directly supported).

CRITICAL: Ensure your analysis accounts for all major input categories and functional types provided, not just the most obvious.
An incomplete or cherry-picked explanation reduces confidence.

Confidence should primarily reflect internal coherence (does the inferred computation logically explain the observed connectivity?)
and completeness (do all input categories contribute meaningfully?). Connection strength is supporting evidence, not the primary criterion."""


USER_PROMPT_TEMPLATE = """Cell Type: {cell_type} on the {side} hemisphere

Intermediate neurons directly upstream of {cell_type} ({num_intermediates} shown):
{intermediate_table}

Known functional inputs to these intermediates (contextual receptive field):
{known_table}

Sensory inputs to {cell_type}:
{sensory_table}

Functionally characterized inputs to {cell_type}:
{known_dn_table}

Outputs from {cell_type} to functionally characterized neurons:
{dn_output_table}

---

ANALYSIS TASK:

Your goal is to infer what adaptive computation {cell_type} performs by analyzing how its inputs—sensory, motor,
and state-related—combine through intermediates and direct pathways. Use the "contextual receptive field" to reason about
the problem this circuit likely solves.

Respond with a JSON object containing:

{{
  "intermediate_computations": "In a single string altogether (NOT as a nested JSON object): for every key intermediate: What functional operation does it perform? How do its inputs (sensory + characterized) combine to create a specific transformation? Why might this circuit route signals through these intermediates rather than directly?",

  "circuit_logic": "Describe the complete information flow: Which inputs are integrated? How do excitatory and inhibitory pathways work together to enable the computation? What does the overall routing suggest about selective combination, gating, modulation, or comparison of inputs?",

  "inferred_computation": "State the core adaptive problem this cell type solves. Be mechanistic and grounded in the connectivity. Synthesize across all inputs if possible. If multiple distinct interpretations exist, present them as alternatives. Mark each: [grounded] (directly supported by wiring), [likely] (coherent with known motifs), or [speculative] (requires additional mechanisms).",

  "behavioral_context": "In what sensory scene, motor state, or task-context would this circuit be active? What adaptive behavior or decision might depend on this computation?",

  "confidence": "high/medium/low (based on coherence of the computation and completeness of input integration)",

  "coherence_assessment": "How well do all the inputs (sensory, characterized, via intermediates) fit together into a single coherent function? Are there unexplained features? Does every major input category contribute meaningfully to the hypothesis?",

  "key_supporting_connections": "Which connectivity patterns most strongly support your main hypothesis?"
}}

If you feel not enough information is available to generate a grounded hypothesis, then say exactly "insufficient_info" in the corresponding fields. 

STRICT FORMAT RULES — VIOLATIONS WILL CAUSE REJECTION AND RETRY:
- Return EXACTLY one JSON object. No markdown, no code fences, no prose before or after.
- The object must contain EXACTLY these 7 keys and NO others:
  "intermediate_computations", "circuit_logic", "inferred_computation",
  "behavioral_context", "confidence", "coherence_assessment", "key_supporting_connections"
- Do NOT add any extra keys (no "note", "output_summary", "error_handling", etc.).
- All values must be plain strings. No nested objects, no arrays.
- Do not nest JSON inside JSON.
- If insufficient information is available, write "insufficient_info" as the string value.
"""


# ---------------------------------------------------------------------------
# Circuit formatting (from notebook)
# ---------------------------------------------------------------------------

def format_series(series, top_n, add_function=True):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0] if series.shape[1] == 1 else series.sum(axis=1)
    # get strongly connected, at least on one side 
    top_n_selected = series.sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)
    # get the corresponding cell types
    if add_function:
        cell_types_selected = top_n_selected.index.map(side_function_to_type)
        # get the corresponding side_known_function
        sides_functions = meta.loc[meta.cell_type.isin(cell_types_selected), "side_known_function"]
    else:
        cell_types_selected = top_n_selected.index.map(side_random_name_to_type)
        sides_functions = meta.loc[meta.cell_type.isin(cell_types_selected), "side_random_name"]

    filtered = series[series.index.isin(sides_functions)]

    rows = []
    for k, v in filtered.items():
        sign = "Excitatory" if v > 0 else "Inhibitory"
        rows.append(f"  - {k}: {v:.5f} ({sign})")
    return "\n".join(rows) if rows else "  None found", filtered


def format_circuit_data(
    cell_type,
    intermediate_to_cell,
    known_to_intermediate,
    sensory_to_cell=None,
    known_to_cell=None,
    cell_to_known=None,
    top_n=15,
    side="left",
    add_function=True,
):
    # Intermediate → cell
    inter_table, inter_filtered = format_series(intermediate_to_cell, top_n, add_function=add_function)

    # Known → intermediates
    if isinstance(known_to_intermediate, pd.DataFrame):
        valid = [c for c in known_to_intermediate.columns if c in inter_filtered.index]
        kdf = known_to_intermediate[valid]
        known_rows = []
        for intermediate in inter_filtered.index:
            if intermediate not in kdf.columns:
                continue
            col = kdf[intermediate].dropna()
            if len(col) == 0:
                continue
            for kf, w in (
                col.sort_values(key=lambda x: x.abs(), ascending=False).head(top_n).items()
            ):
                sign = "Excitatory" if w > 0 else "Inhibitory"
                known_rows.append(f"  - {kf} → {intermediate}: {w:.5f} ({sign})")
            known_rows.append('\n')
        known_table = "\n".join(known_rows) if known_rows else "  None found"
    else:
        known_table, _ = format_series(known_to_intermediate, top_n, add_function=add_function)

    sensory_table = "  None found"
    if sensory_to_cell is not None:
        t, _ = format_series(sensory_to_cell, top_n, add_function=add_function)
        sensory_table = t

    known_dn_table = "  None found"
    if known_to_cell is not None:
        t, _ = format_series(known_to_cell, top_n, add_function=add_function)
        known_dn_table = t

    dn_output_table = "  None found"
    if cell_to_known is not None:
        t, _ = format_series(cell_to_known, top_n, add_function=add_function)
        dn_output_table = t

    return {
        "cell_type": cell_type,
        "intermediate_table": inter_table,
        "known_table": known_table,
        "sensory_table": sensory_table,
        "known_dn_table": known_dn_table,
        "dn_output_table": dn_output_table,
        "num_intermediates": len(inter_filtered),
        "side": side,
    }

# ---------------------------------------------------------------------------
# Circuit connectivity (from notebook — uses module-level globals)
# ---------------------------------------------------------------------------

def calculate_circuit_connectivity(dn_type, n_steps, add_function=True, side="left"):
    from connectome_interpreter import (
        signed_conn_by_path_length_data,
        el_within_n_steps,
    )

    dn_type_side = dn_type + "_" + side

    # Sensory → DN
    esensdn, isensdn = signed_conn_by_path_length_data(
        inprop,
        meta.idx[(meta.super_class == "sensory") & ~meta.cell_type.str.isdigit()],
        meta.idx[(meta.cell_type == dn_type) & (meta.side == side)],
        n_steps,
        type_side_to_sign,
        idx_to_type_side,
        idx_to_type_side,
    )
    esensdn_sum = reduce(lambda a, b: a.add(b, fill_value=0), esensdn)
    isensdn_sum = reduce(lambda a, b: a.add(b, fill_value=0), isensdn)
    difference_sensdn = esensdn_sum.subtract(isensdn_sum, fill_value=0)
    if add_function:
        difference_sensdn.index = difference_sensdn.index.map(type_side_to_side_function)
    else:
        difference_sensdn.index = difference_sensdn.index.map(cell_type_side_to_random_name)

    # Circuit within n steps
    circuit = el_within_n_steps(
        inprop,
        meta.idx[(meta.super_class == "sensory") & ~meta.cell_type.str.isdigit()],
        meta.idx[(meta.cell_type == dn_type) & (meta.side == side)],
        n_steps,
        0.01,
        idx_to_type_side,
        idx_to_type_side,
    )

    # Direct intermediate → DN
    intermediate_dn = circuit[
        # remove self connection, even if on the opposite side 
        (circuit.post == dn_type_side) & (~circuit.pre.str.contains(dn_type))
    ].copy()
    intermediate_dn["pre_sign"] = intermediate_dn.pre.map(type_side_to_sign)
    intermediate_dn["weight"] = intermediate_dn.weight * intermediate_dn.pre_sign
    if add_function:
        intermediate_dn["pre"] = intermediate_dn.pre.map(type_side_to_side_function)
    else:
        intermediate_dn["pre"] = intermediate_dn.pre.map(cell_type_side_to_random_name)
    intermediate_dn = intermediate_dn.set_index("pre")["weight"]

    # Known → intermediates upstream of DN
    ekunk, ikunk = signed_conn_by_path_length_data(
        inprop,
        meta.idx[(meta.cell_type != meta.known_function) & (meta.cell_type != dn_type)],
        meta.idx[
            meta.type_side.isin(
                circuit.pre[
                    (circuit.post == dn_type_side) & (~circuit.pre.str.contains(dn_type))
                ]
            )
        ],
        n_steps,
        type_side_to_sign,
        idx_to_type_side,
        idx_to_type_side,
    )
    ekunk_sum = reduce(lambda a, b: a.add(b, fill_value=0), ekunk)
    ikunk_sum = reduce(lambda a, b: a.add(b, fill_value=0), ikunk)
    difference_kunk = ekunk_sum.subtract(ikunk_sum, fill_value=0)
    if add_function:
        difference_kunk.index = difference_kunk.index.map(type_side_to_side_function)
        difference_kunk.columns = difference_kunk.columns.map(type_side_to_side_function)
    else:
        difference_kunk.index = difference_kunk.index.map(cell_type_side_to_random_name)
        difference_kunk.columns = difference_kunk.columns.map(cell_type_side_to_random_name)

    # Known → DN
    ekdn, ikdn = signed_conn_by_path_length_data(
        inprop,
        meta.idx[(meta.cell_type != meta.known_function) & (meta.cell_type != dn_type)],
        meta.idx[(meta.cell_type == dn_type) & (meta.side == side)],
        n_steps,
        type_side_to_sign,
        idx_to_type_side,
        idx_to_type_side,
    )
    ekdn_sum = reduce(lambda a, b: a.add(b, fill_value=0), ekdn)
    ikdn_sum = reduce(lambda a, b: a.add(b, fill_value=0), ikdn)
    difference_kdn = ekdn_sum.subtract(ikdn_sum, fill_value=0)
    if add_function:
        difference_kdn.index = difference_kdn.index.map(type_side_to_side_function)
    else:
        difference_kdn.index = difference_kdn.index.map(cell_type_side_to_random_name)

    # DN → known
    ednk, idnk = signed_conn_by_path_length_data(
        inprop,
        meta.idx[(meta.cell_type == dn_type) & (meta.side == side)],
        meta.idx[(meta.cell_type != meta.known_function) & (meta.cell_type != dn_type)],
        n_steps,
        type_side_to_sign,
        idx_to_type_side,
        idx_to_type_side,
    )
    ednk_sum = reduce(lambda a, b: a.add(b, fill_value=0), ednk)
    idnk_sum = reduce(lambda a, b: a.add(b, fill_value=0), idnk)
    difference_dnk = ednk_sum.subtract(idnk_sum, fill_value=0).T
    if add_function:
        difference_dnk.index = difference_dnk.index.map(type_side_to_side_function)
    else:
        difference_dnk.index = difference_dnk.index.map(cell_type_side_to_random_name)

    return (
        difference_sensdn,
        intermediate_dn,
        difference_kunk,
        difference_kdn,
        difference_dnk,
    )


REQUIRED_KEYS = {
    "intermediate_computations", "circuit_logic", "inferred_computation",
    "behavioral_context", "confidence", "coherence_assessment",
    "key_supporting_connections",
}

def build_prompts_for_cell(cell_type, args):
    """Pure: returns (system, user) for one cell type. No LLM call."""
    (difference_sensdn, intermediate_dn, difference_kunk,
     difference_kdn, difference_dnk) = calculate_circuit_connectivity(
        cell_type, args.n_steps, add_function=args.add_function, side=args.side
    )
    fmt = format_circuit_data(
        cell_type=cell_type,
        intermediate_to_cell=intermediate_dn,
        known_to_intermediate=difference_kunk,
        sensory_to_cell=difference_sensdn,
        known_to_cell=difference_kdn,
        cell_to_known=difference_dnk,
        top_n=args.top_n,
        side=args.side,
        add_function=args.add_function,
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(**fmt)
    return SYSTEM_PROMPT, user_prompt

def load_data(base_path, known_types_csv):
    """Load connectome data and set module globals.

    Called both by main() (HPC path) and by Colab notebooks. Sets globals
    in this module so calculate_circuit_connectivity and format_circuit_data
    can use them.
    """
    global inprop, meta, cell_type_to_function, type_to_sign, idx_to_type
    global cell_type_to_random_name, idx_to_type_side, type_side_to_sign
    global cell_type_side_to_random_name, type_side_to_side_function
    global side_function_to_type, side_random_name_to_type

    # ---- Load data ----
    print("Loading connectome data...")
    base_path = args.base_path

    global inprop, meta, cell_type_to_function, type_to_sign, idx_to_type, cell_type_to_random_name, idx_to_type_side, type_side_to_sign, cell_type_side_to_random_name, type_side_to_side_function, side_function_to_type, side_random_name_to_type

    cell_type_to_function_df = pd.read_csv(os.path.expanduser(args.known_types_csv))
    cell_type_to_function = dict(
        zip(cell_type_to_function_df.cell_type, cell_type_to_function_df.known_function)
    )

    inprop = sp.sparse.load_npz(
        os.path.join(base_path, "data", "fafb_all_neuron", "fafb_inprop_all_neuron.npz")
    )
    meta = pd.read_csv(
        os.path.join(base_path, "data", "fafb_all_neuron", "fafb_all_neuron_meta.csv"),
        index_col=0, low_memory=False
    )
    meta["type_side"] = meta.cell_type + "_" + meta.side

    idx_to_type_side = dict(zip(meta.idx, meta.type_side))
    idx_to_type = dict(zip(meta.idx, meta.cell_type))
    type_to_sign = dict(zip(meta.cell_type, meta.sign))
    type_side_to_sign = dict(zip(meta.type_side, meta.sign))

    meta.loc[meta.cell_type.isin(cell_type_to_function), "known_function"] = (
        meta[meta.cell_type.isin(cell_type_to_function)].cell_type.map(cell_type_to_function)
    )
    meta.loc[meta.super_class == "motor", "known_function"] = meta[
        meta.super_class == "motor"
    ].agg(
        lambda x: x["cell_sub_class"] if x["cell_type"] == x["known_function"] else x["known_function"],
        axis=1,
    )
    meta["known_function"] = meta["known_function"].fillna(meta.cell_type)
    cell_type_to_function = dict(zip(meta.cell_type, meta.known_function))

    meta['side_known_function'] = meta.side + " " + meta.known_function
    type_side_to_side_function = dict(zip(meta.cell_type + "_" + meta.side, meta.side_known_function))
    side_function_to_type = dict(zip(meta.side_known_function, meta.cell_type))

    # random cell type names preserving side correspondence 
    random_names = [f"cell_type_{i}" for i in range(len(set(meta.cell_type)))]
    cell_type_to_random_name = dict(zip(meta.cell_type.unique(), random_names))
    meta['random_name'] = meta.cell_type.map(cell_type_to_random_name)
    meta['side_random_name'] = meta.cell_type.map(cell_type_to_random_name) + "_" + meta.side
    cell_type_side_to_random_name = dict(zip(meta.cell_type + "_" + meta.side, meta.side_random_name))
    side_random_name_to_type = dict(zip(meta.side_random_name, meta.cell_type))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load env + set model default
    dotenv.load_dotenv()
    if args.model is None:
        args.model = PROVIDER_DEFAULTS[args.provider]

    load_data(args.base_path, args.known_types_csv)
    
    # ---- Build dns list ----
    known_types = set(
        meta.cell_type[(meta.cell_type != meta.known_function) & (meta.super_class != "sensory")]
    )
    # dns = sorted(known_types)  # sorted for reproducibility
    dns = sorted(set(meta.cell_type[
        (~meta.cell_type.str.isdigit()) & (~meta.super_class.isin(['sensory','ascending']))
    ]))
    print(f"Total cell types to process: {len(dns)}")

    # ---- Split into chunks ----
    chunk_size = (len(dns) + args.n_chunks - 1) // args.n_chunks
    start = args.chunk_id * chunk_size
    end = min(start + chunk_size, len(dns))
    chunk = dns[start:end]
    print(f"Chunk {args.chunk_id}/{args.n_chunks}: cells {start}-{end} ({len(chunk)} cells)")

    # ---- Output setup ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"results_chunk_{args.chunk_id}.jsonl"

    # Resume: skip already-done cells
    done = set()
    if args.resume and out_file.exists():
        with open(out_file) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["cell_type"])
                except Exception:
                    pass
        print(f"Resuming: {len(done)} cells already done, skipping.")
    chunk = [c for c in chunk if c not in done]

    if args.dump_prompts:
        from llm_core import write_prompt_file, safe_filename
        out_dir = Path(args.dump_prompts)
        out_dir.mkdir(parents=True, exist_ok=True)
        for cell_type in chunk:
            if not ((meta.cell_type == cell_type) & (meta.side == args.side)).any():
                print(f"  [skip] {cell_type} not on side {args.side}")
                continue
            try:
                system_prompt, user_prompt = build_prompts_for_cell(cell_type, args)
            except Exception as e:
                print(f"  [skip] {cell_type}: {e}")
                continue
            out_path = out_dir / f"{safe_filename(cell_type)}.md"
            write_prompt_file(out_path, cell_type, system_prompt, user_prompt,
                            required_keys=REQUIRED_KEYS)
        print(f"Dumped prompts to {out_dir}. Exiting (no LLM calls).")
        return

    if not chunk:
        print("All cells in this chunk already processed. Exiting.")
        return

    # ---- Start llama-server if needed ----
    llama_proc = None
    llama_port = None
    if args.provider == "llama":
        llama_proc, llama_port = setup_llama_server(args)

    # ---- Process cells ----
    try:
        with open(out_file, "a") as f:
            for cell_type in tqdm(chunk, desc=f"Chunk {args.chunk_id}"):
                # first check if this cell type exists with this side 
                if not ((meta.cell_type == cell_type) & (meta.side == args.side)).any():
                    print(f"  [skipping] {cell_type} on side {args.side} not found in metadata")
                    continue
                # Check server health before each cell; restart if needed
                if args.provider == "llama":
                    if not is_llama_server_healthy(llama_port):
                        print(f"  [llama-server] Unhealthy before {cell_type}, restarting...")
                        llama_proc = restart_llama_server(
                            args.llama_bin, args.gguf, llama_port, llama_proc
                        )
                try:
                    system_prompt, user_prompt = build_prompts_for_cell(cell_type, args)
                    result = call_llm_with_retry(
                        system_prompt, user_prompt, REQUIRED_KEYS, args,
                        llama_port=llama_port,
                        extra_meta_keys={"cell_type", "add_function"},
                    )
                    cleaned = {k: result[k] for k in REQUIRED_KEYS if k in result}
                    cleaned["cell_type"] = cell_type
                    cleaned["model"] = args.model
                    cleaned["provider"] = args.provider
                    cleaned["add_function"] = args.add_function
                    cleaned["chunk_id"] = args.chunk_id
                    if "validation_error" in result:
                        cleaned["validation_error"] = result["validation_error"]
                    f.write(json.dumps(cleaned) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"  ERROR processing {cell_type}: {e}")
                    error_record = {
                        "cell_type": cell_type,
                        "error": str(e),
                        "chunk_id": args.chunk_id,
                    }
                    f.write(json.dumps(error_record) + "\n")
                    f.flush()
                    continue
    finally:
        if llama_proc is not None:
            print("[llama-server] Shutting down...")
            llama_proc.terminate()
            llama_proc.wait()

    print(f"Done. Results saved to {out_file}")


if __name__ == "__main__":
    main()