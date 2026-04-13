"""
circuit_step3_synthesis.py — Circuit-level synthesis.

Aggregates all per-target integration analyses (from step 2) and asks
the LLM how the circuit works as a whole.

Includes a source×target signed effect matrix from step 1 data so the LLM
can sanity-check the verbal summaries against raw numbers.

Usage:
    python circuit_step3_synthesis.py [options]

Reads:  results_step1/pairwise_chunk_*.jsonl (for matrix)
        results_step2/target_chunk_*.jsonl (for integration analyses)
Writes: results_step3/circuit_synthesis.json
"""

import argparse
import json
import os
from pathlib import Path

import dotenv

from circuit_utils import (
    PROVIDER_DEFAULTS,
    add_common_args,
    build_source_target_matrix,
    call_llm_with_retry,
    load_hypotheses,
    load_types_file,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a neuroscience expert specializing in Drosophila neural circuits, computational neuroscience,
and systems-level circuit analysis.

Your task is to synthesize per-neuron integration analyses into a CIRCUIT-LEVEL understanding of how
a group of interconnected neurons works together to implement a coherent computation.

You will be given:
1. Per-target integration analyses (how each target integrates inputs from the sources)
2. Source and target hypotheses (from single-neuron analysis)
3. A source×target signed effect matrix (raw data — use to verify the verbal analyses)

Focus on:
- What overall computation does this circuit implement? What adaptive problem does it solve?
- How do the individual neurons' roles compose into a system? What division of labour emerges?
- Are there functional modules, push-pull pairs, mutual inhibition, winner-take-all dynamics,
  or other circuit motifs? Use the source×target matrix to identify these patterns.
- What predictions does this circuit model make about behaviour or neural dynamics?
- Where are the gaps — what aspects of the circuit remain unexplained?

Be MECHANISTIC and INTEGRATIVE. The goal is a concise yet comprehensive description of how
the circuit works, not a list of what each neuron does individually."""

USER_PROMPT_TEMPLATE = """\
Circuit analysis: {n_sources} source types → {n_targets} target types ({side} hemisphere)

Sources: {source_list}
Targets: {target_list}

Source and target hypotheses (from single-neuron analysis):
{hypotheses_section}

{source_target_matrix}

---

Per-target integration analyses:

{target_summaries}

---

Synthesize how this circuit works as a whole.

Respond with a JSON object containing EXACTLY these keys and no others:

{{
  "circuit_mechanism": "What overall computation does this circuit implement? How do the individual
    neuron types compose into a functional system? Describe the information flow, key circuit motifs
    (e.g., mutual inhibition, push-pull, recurrence, gating), and the division of labour. Use the
    source×target matrix to ground your claims about inter-neuron interactions. Be mechanistic and
    integrative. 300-600 words.",

  "adaptive_function": "What adaptive problem does this circuit solve? In what behavioural context
    would this circuit be active? What would go wrong if specific components were removed or silenced?
    150-300 words.",

  "predictions": "What testable predictions does this circuit model make about neural dynamics,
    behavioural phenotypes upon silencing, or responses to specific stimuli? Be specific and concrete.",

  "open_questions": "What aspects of the circuit remain unexplained? Where are the biggest gaps
    in the mechanistic story? What additional data would help resolve ambiguities?",

  "confidence": "high/medium/low — based on overall coherence, completeness, and consistency
    across all per-target analyses"
}}

STRICT FORMAT: Return EXACTLY one JSON object with these 5 keys. No markdown, no code fences,
no prose before or after. All values must be plain strings."""


REQUIRED_KEYS = {
    "circuit_mechanism", "adaptive_function", "predictions",
    "open_questions", "confidence",
}


# ---------------------------------------------------------------------------
# Load step 2 results
# ---------------------------------------------------------------------------

def load_step2_results(step2_dir):
    """Load all per-target results from step 2."""
    step2_dir = Path(step2_dir)
    results = []

    for f in sorted(step2_dir.glob("target_chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in rec and "integration_mechanism" not in rec:
                    continue
                results.append(rec)

    return results


def format_target_summaries(target_results):
    """Format all per-target integration results into a prompt section."""
    lines = []
    for rec in target_results:
        target = rec.get("target", "?")
        sources = rec.get("sources", [])
        mechanism = rec.get("integration_mechanism", "No analysis available")
        refined_hyp = rec.get("refined_hypothesis", "")
        confidence = rec.get("confidence", "?")

        lines.append(f"=== {target} (integrating from: {', '.join(sources)}) ===")
        lines.append(f"Integration mechanism: {mechanism}")
        if refined_hyp:
            lines.append(f"Refined hypothesis: {refined_hyp}")
        lines.append(f"Confidence: {confidence}")
        lines.append("")

    return "\n".join(lines)


def format_hypotheses_section(sources, targets, type_to_hypothesis):
    """Format source and target hypotheses as a reference section."""
    lines = []
    all_types = sorted(set(sources + targets))
    for ct in all_types:
        hyp = type_to_hypothesis.get(ct, "No hypothesis available")
        if isinstance(hyp, str) and len(hyp) > 400:
            hyp = hyp[:400] + "..."
        role = []
        if ct in sources:
            role.append("source")
        if ct in targets:
            role.append("target")
        lines.append(f"  {ct} ({'/'.join(role)}): {hyp}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 3: Circuit-level synthesis")
    parser.add_argument("--step1-dir", type=str, default="./results_step1")
    parser.add_argument("--step2-dir", type=str, default="./results_step2")
    parser.add_argument("--output-dir", type=str, default="./results_step3")
    add_common_args(parser)
    args = parser.parse_args()

    args.chunk_id = 0  # no chunking for step 3

    dotenv.load_dotenv()
    if args.model is None:
        args.model = PROVIDER_DEFAULTS[args.provider]

    # Load hypotheses and types
    type_to_hypothesis = load_hypotheses(args.hypotheses_csv)
    sources, targets = load_types_file(args.types_file)

    # Load step 2 results
    print(f"Loading step 2 results from {args.step2_dir}...")
    target_results = load_step2_results(args.step2_dir)
    print(f"Found integration results for {len(target_results)} targets")

    if not target_results:
        print("No target results found. Run step 2 first.")
        return

    # Build source×target matrix from step 1 data
    print(f"Building source-target matrix from {args.step1_dir}...")
    source_target_matrix = build_source_target_matrix(args.step1_dir)

    # Format prompt
    target_summaries = format_target_summaries(target_results)
    hypotheses_section = format_hypotheses_section(
        sources, targets, type_to_hypothesis
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        n_sources=len(sources),
        n_targets=len(targets),
        source_list=", ".join(sources),
        target_list=", ".join(targets),
        hypotheses_section=hypotheses_section,
        source_target_matrix=source_target_matrix,
        target_summaries=target_summaries,
        side=args.side,
    )

    # llama-server
    llama_proc = None
    llama_port = None
    if args.provider == "llama":
        from circuit_utils import setup_llama_server
        llama_proc, llama_port = setup_llama_server(args)

    try:
        print("Calling LLM for circuit synthesis...")
        result = call_llm_with_retry(
            SYSTEM_PROMPT, user_prompt, REQUIRED_KEYS, args, llama_port
        )

        result["sources"] = sources
        result["targets"] = targets
        result["model"] = args.model
        result["provider"] = args.provider

        # Save
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "circuit_synthesis.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved circuit synthesis to {out_file}")

        # Print readable summary
        print("\n" + "=" * 70)
        print("CIRCUIT SYNTHESIS")
        print("=" * 70)
        for key in ["circuit_mechanism", "adaptive_function", "predictions",
                     "open_questions", "confidence"]:
            if key in result:
                print(f"\n--- {key} ---")
                print(result[key])

    finally:
        if llama_proc is not None:
            print("[llama-server] Shutting down...")
            llama_proc.terminate()
            llama_proc.wait()


if __name__ == "__main__":
    main()