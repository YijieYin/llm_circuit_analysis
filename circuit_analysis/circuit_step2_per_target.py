"""
circuit_step2_per_target.py — Per-target integration analysis.

For each target, aggregates all pairwise pathway analyses (from step 1)
and asks the LLM how the target integrates information from all sources.

Includes:
  - Raw data summary: (source, net_sign, net_strength) table for sanity-checking
  - Shared intermediates: neurons that appear in pathways from 2+ sources
  - Critic flags from step 1 are surfaced for the LLM to weigh

Usage:
    python circuit_step2_per_target.py --chunk-id 0 --n-chunks 2 [options]

Reads:  results_step1/pairwise_chunk_*.jsonl (from step 1)
Writes: results_step2/target_chunk_{chunk_id}.jsonl
"""

import argparse
import json
import os
from pathlib import Path

import dotenv
from tqdm import tqdm

from circuit_utils import (
    PROVIDER_DEFAULTS,
    add_common_args,
    call_llm_with_retry,
    compute_shared_intermediates,
    format_known_to_target,
    format_net_effect_table,
    is_llama_server_healthy,
    load_connectome_data,
    load_hypotheses,
    load_types_file,
    restart_llama_server,
    setup_llama_server,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """\
You are a neuroscience expert specializing in Drosophila neural circuits and computational neuroscience.

Your task is to infer how a TARGET neuron type integrates information from MULTIPLE source neuron types.
You will be given:
1. The target's own functional hypothesis (from prior single-neuron analysis)
2. Pairwise mechanistic analyses describing how each source influences the target
3. A raw data summary table of net signed effects (use this to sanity-check the verbal analyses)
4. Known functional inputs to the target for broader context{shared_intermediates_note}

Focus on:
- How do the different source pathways interact at the target? Do they converge, compete,
  or gate each other?{shared_intermediates_focus}
- What emergent computation arises from integrating these pathways that wouldn't exist with
  any single pathway alone?
- Does the integration pattern refine, extend, or challenge the target's existing hypothesis?
- Sign interactions across pathways: use the raw data table to verify which sources are
  excitatory vs inhibitory to the target, and what the balance enables.
- If any pathway has a CRITIC FLAG, treat that analysis with appropriate skepticism —
  the sign logic may be incorrect.

Be MECHANISTIC. Synthesize across pathways rather than summarizing each one separately.

IMPORTANT: This output will be consumed by a final circuit-level synthesis.
Be concise (200-400 words per field). Focus on the integrative insight."""

SHARED_INTERMEDIATES_NOTE = """
5. Shared intermediates: neurons in pathways from 2+ sources to this target, with their
   signed effective connectivity (a subset of the circuit for additional context)"""

SHARED_INTERMEDIATES_FOCUS = """
- If shared intermediates are provided, consider how signals from multiple sources converge
  at these nodes before reaching the target. What computation happens at these convergence
  points? But remember these are a subset — do not focus on them exclusively."""

USER_PROMPT_TEMPLATE = """\
Target: {target}
Target hypothesis (from single-neuron analysis): {target_hypothesis}

{known_to_target}

{net_effect_table}

{shared_intermediates}

---

Pairwise pathway analyses (how each source influences {target}):

{pairwise_summaries}

---

Analyze how {target} integrates information from all these sources.

Respond with a JSON object containing EXACTLY these keys and no others:

{{
  "integration_mechanism": "How does {target} combine information from these multiple sources?
    What computation emerges from the convergence — especially at shared intermediates?
    How do excitatory and inhibitory pathways from different sources interact?
    Does the integration pattern suggest a specific adaptive function? 200-400 words.",

  "refined_hypothesis": "Based on the multi-source integration, does the original single-neuron
    hypothesis need refinement? State the updated functional hypothesis for {target} in the
    context of this specific circuit. Be concise.",

  "confidence": "high/medium/low — based on coherence across pathways and completeness
    of integration"
}}

STRICT FORMAT: Return EXACTLY one JSON object with these 3 keys. No markdown, no code fences,
no prose before or after. All values must be plain strings."""


REQUIRED_KEYS = {"integration_mechanism", "refined_hypothesis", "confidence"}


# ---------------------------------------------------------------------------
# Load step 1 results
# ---------------------------------------------------------------------------

def load_step1_results(step1_dir):
    """Load all pairwise results from step 1, grouped by target.
    
    Returns dict: target -> list of pairwise result dicts.
    """
    step1_dir = Path(step1_dir)
    by_target = {}

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
                # Skip error records
                if "error" in rec and "pathway_mechanism" not in rec:
                    continue
                target = rec.get("target")
                if target:
                    by_target.setdefault(target, []).append(rec)

    return by_target


def format_pairwise_summaries(pairwise_results):
    """Format all pairwise results for a target into a prompt section.
    
    Includes critic flags where present.
    """
    lines = []
    for rec in pairwise_results:
        source = rec.get("source", "?")
        mechanism = rec.get("pathway_mechanism", "No analysis available")
        key_transforms = rec.get("key_transformations", "")
        confidence = rec.get("confidence", "?")
        critic_flag = rec.get("critic_flag", "")

        lines.append(f"=== {source} → target ===")
        if critic_flag:
            lines.append(f"⚠ CRITIC FLAG: {critic_flag}")
        lines.append(f"Pathway mechanism: {mechanism}")
        if key_transforms:
            lines.append(f"Key transformations: {key_transforms}")
        lines.append(f"Confidence: {confidence}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_target(target, pairwise_results, sources, data, args, llama_port=None):
    """Process a single target's integration."""
    type_to_hypothesis = data["type_to_hypothesis"]
    target_hyp = type_to_hypothesis.get(target, "No hypothesis available")

    # Format pairwise summaries (includes critic flags)
    pairwise_summaries = format_pairwise_summaries(pairwise_results)

    # Raw data summary table
    net_effect_table = format_net_effect_table(pairwise_results)

    # Shared intermediates (optional)
    shared_intermediates_text = ""
    if args.shared_intermediates:
        shared_intermediates_text = compute_shared_intermediates(
            data, sources, target, args.side, args.n_steps, args.threshold,
            top_k=args.shared_intermediates_top_k,
        ) or ""

    # Build system prompt (adapts to whether shared intermediates are included)
    if shared_intermediates_text:
        system_prompt = SYSTEM_PROMPT_BASE.format(
            shared_intermediates_note=SHARED_INTERMEDIATES_NOTE,
            shared_intermediates_focus=SHARED_INTERMEDIATES_FOCUS,
        )
    else:
        system_prompt = SYSTEM_PROMPT_BASE.format(
            shared_intermediates_note="",
            shared_intermediates_focus="",
        )

    # Known functional inputs
    known_to_target = format_known_to_target(
        data, target, args.side, args.n_steps, top_n=5
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        target=target,
        target_hypothesis=target_hyp,
        known_to_target=known_to_target,
        net_effect_table=net_effect_table,
        shared_intermediates=shared_intermediates_text,
        pairwise_summaries=pairwise_summaries,
    )

    result = call_llm_with_retry(
        system_prompt, user_prompt, REQUIRED_KEYS, args, llama_port
    )

    result["target"] = target
    result["sources"] = [r["source"] for r in pairwise_results]
    result["model"] = args.model
    result["provider"] = args.provider
    result["chunk_id"] = args.chunk_id

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 2: Per-target integration analysis")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--n-chunks", type=int, required=True)
    parser.add_argument("--step1-dir", type=str, default="./results_step1")
    parser.add_argument("--output-dir", type=str, default="./results_step2")
    parser.add_argument("--shared-intermediates", action="store_true", default=False,
                        help="Include shared intermediates analysis (neurons in pathways "
                             "from 2+ sources). Adds compute cost and prompt length.")
    parser.add_argument("--shared-intermediates-top-k", type=int, default=10,
                        help="Max shared intermediates to include (default: 10)")
    add_common_args(parser)
    args = parser.parse_args()

    dotenv.load_dotenv()
    if args.model is None:
        args.model = PROVIDER_DEFAULTS[args.provider]

    # Load data
    print("Loading connectome data...")
    data = load_connectome_data(args.base_path, args.known_types_csv)
    data["type_to_hypothesis"] = load_hypotheses(args.hypotheses_csv)
    sources, targets = load_types_file(args.types_file)

    # Load step 1 results
    print(f"Loading step 1 results from {args.step1_dir}...")
    by_target = load_step1_results(args.step1_dir)
    print(f"Found pairwise results for {len(by_target)} targets")

    # Filter to only the targets in our types file that have step 1 results
    valid_targets = [t for t in targets if t in by_target]
    print(f"Targets with pairwise data: {len(valid_targets)}")

    if not valid_targets:
        print("No targets have pairwise results. Run step 1 first.")
        return

    # Chunk over targets
    chunk_size = (len(valid_targets) + args.n_chunks - 1) // args.n_chunks
    start = args.chunk_id * chunk_size
    end = min(start + chunk_size, len(valid_targets))
    chunk = valid_targets[start:end]
    print(f"Chunk {args.chunk_id}/{args.n_chunks}: targets {start}-{end} ({len(chunk)} targets)")

    # Output setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"target_chunk_{args.chunk_id}.jsonl"

    # Resume
    done = set()
    if args.resume and out_file.exists():
        with open(out_file) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["target"])
                except Exception:
                    pass
        print(f"Resuming: {len(done)} targets already done.")
    chunk = [t for t in chunk if t not in done]

    if not chunk:
        print("All targets in this chunk already processed. Exiting.")
        return

    # llama-server
    llama_proc = None
    llama_port = None
    if args.provider == "llama":
        llama_proc, llama_port = setup_llama_server(args)

    # Process targets
    try:
        with open(out_file, "a") as f:
            for target in tqdm(chunk, desc=f"Chunk {args.chunk_id}"):
                if args.provider == "llama" and not is_llama_server_healthy(llama_port):
                    llama_proc = restart_llama_server(
                        args.llama_bin, args.gguf, llama_port, llama_proc
                    )
                try:
                    result = process_target(
                        target, by_target[target], sources, data, args, llama_port
                    )
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"  ERROR {target}: {e}")
                    error_rec = {
                        "target": target, "error": str(e),
                        "chunk_id": args.chunk_id,
                    }
                    f.write(json.dumps(error_rec) + "\n")
                    f.flush()
    finally:
        if llama_proc is not None:
            print("[llama-server] Shutting down...")
            llama_proc.terminate()
            llama_proc.wait()

    print(f"Done. Results saved to {out_file}")


if __name__ == "__main__":
    main()