"""
circuit_step1_pairwise.py — Pairwise pathway mechanism analysis.

For each (source, target) pair, finds paths within n_steps, formats them
by recipient at each layer, and asks the LLM to infer the mechanism by
which the source influences the target through intermediates.

Optionally enable an actor-critic loop where a critic reviews the analysis
and the actor revises based on feedback, repeating until approved or
--max-critic-rounds is reached.

Usage:
    python circuit_step1_pairwise.py --chunk-id 0 --n-chunks 4 [options]
    python circuit_step1_pairwise.py --chunk-id 0 --n-chunks 4 --actor-critic --max-critic-rounds 3

Reads:  circuit_types.json (from select_types.py)
Writes: results_step1/pairwise_chunk_{chunk_id}.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

import dotenv
import pandas as pd
from tqdm import tqdm

from circuit_utils import (
    PROVIDER_DEFAULTS,
    add_common_args,
    call_llm,
    call_llm_with_retry,
    compute_net_signed_effect,
    find_paths_for_pair,
    format_known_to_target,
    format_paths_by_recipient,
    is_llama_server_healthy,
    load_connectome_data,
    load_hypotheses,
    load_types_file,
    parse_llm_response,
    resolve_model_config,
    restart_llama_server,
    setup_llama_server,
)


# ---------------------------------------------------------------------------
# Actor prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a neuroscience expert specializing in Drosophila neural circuits and computational neuroscience.

Your task is to infer the MECHANISM by which a source neuron type influences a target neuron type,
based on the pathway connectivity and the existing functional hypotheses for each intermediate neuron.

Focus on:
- What transformations occur at each intermediate? How does each intermediate's known computation
  shape the signal as it flows from source to target?
- Sign interactions: how do excitatory/inhibitory connections combine across layers to produce
  the net effect? What does the sign pattern enable (e.g., disinhibition, gating, gain control)?
- Why might this circuit route through these specific intermediates rather than a direct connection?
  What computational advantage does this multi-step pathway provide?
- Laterality: does the side the input comes from (ipsi vs contra) matter in this case? If so, how?

Be MECHANISTIC, not descriptive. Do not just restate connectivity — explain what computation it implements.

IMPORTANT: This output will be consumed by a downstream analysis that aggregates across many pathways.
Be concise (150-300 words per field). Focus on the key mechanistic insight."""

USER_PROMPT_TEMPLATE = """\
Source: {source} — Hypothesis: {source_hypothesis}
Target: {target} — Hypothesis: {target_hypothesis}

Pathway connectivity from {source} to {target} ({side} hemisphere), grouped by layer and recipient:

{paths_formatted}

{net_effect_line}

{known_to_target}

---

Analyze the mechanism by which {source} influences {target} through these intermediates.

Respond with a JSON object containing EXACTLY these keys and no others:

{{
  "pathway_mechanism": "What computation does this pathway implement? How does the source's signal
    get transformed step-by-step through the intermediates to influence the target? Be specific
    about sign interactions and what they enable. 150-300 words.",

  "key_transformations": "For each major intermediate, one sentence on what transformation it
    performs in this specific pathway context. Be concise.",

  "confidence": "high/medium/low — based on how well the intermediate hypotheses and sign
    patterns support a coherent mechanistic story"
}}

STRICT FORMAT: Return EXACTLY one JSON object with these 3 keys. No markdown, no code fences,
no prose before or after. All values must be plain strings."""


REQUIRED_KEYS = {"pathway_mechanism", "key_transformations", "confidence"}


# ---------------------------------------------------------------------------
# Actor revision prompt (used in actor-critic loop)
# ---------------------------------------------------------------------------

REVISION_PROMPT_TEMPLATE = """\
Source: {source} — Hypothesis: {source_hypothesis}
Target: {target} — Hypothesis: {target_hypothesis}

Pathway connectivity from {source} to {target} ({side} hemisphere), grouped by layer and recipient:

{paths_formatted}

{net_effect_line}

{known_to_target}

---

Your previous analysis of this pathway was:

pathway_mechanism: {prev_mechanism}

key_transformations: {prev_transformations}

confidence: {prev_confidence}

---

A reviewer provided the following feedback:

{critique}

---

Revise your analysis to address the reviewer's feedback. Retain what was correct and fix what was
flagged. Do not just acknowledge the feedback — incorporate it into an improved analysis.

Respond with a JSON object containing EXACTLY these keys and no others:

{{
  "pathway_mechanism": "Revised analysis. 150-300 words.",
  "key_transformations": "Revised transformations. Be concise.",
  "confidence": "high/medium/low"
}}

STRICT FORMAT: Return EXACTLY one JSON object with these 3 keys. No markdown, no code fences,
no prose before or after. All values must be plain strings."""


# ---------------------------------------------------------------------------
# Critic prompts
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = """\
You are a rigorous neuroscience reviewer evaluating a circuit mechanism analysis.

You are given raw connectivity data (layer-by-layer edges with weights and signs) and an
LLM-generated mechanistic interpretation. Your job is to evaluate whether the analysis is
correct, complete, and genuinely mechanistic.

Check ALL of the following:

1. SIGN LOGIC: Does the analysis correctly track excitatory/inhibitory interactions through
   the layers? Are disinhibition, gating, and sign-flip claims supported by the actual sign
   sequence in the data? Does the claimed net effect match the computed net effect?

2. COMPLETENESS: Does the analysis address all major intermediates shown in the data, or does
   it cherry-pick a subset? Are important pathways ignored?

3. MECHANISTIC DEPTH: Is the analysis explaining what computation the pathway implements, or
   is it merely restating the connectivity in different words? "A excites B which inhibits C"
   is description, not mechanism. What PROBLEM does this pathway solve?

4. INTERNAL COHERENCE: Do the claims in pathway_mechanism and key_transformations agree with
   each other? Is the confidence level justified by the actual evidence?

Be specific and actionable in your critique. Point to concrete connections or intermediates
where you see problems."""

CRITIC_USER_TEMPLATE = """\
Raw connectivity data:

{paths_formatted}

{net_effect_line}

---

Analysis to review:

pathway_mechanism: {pathway_mechanism}

key_transformations: {key_transformations}

confidence: {confidence}

---

Evaluate this analysis against the raw data.

Respond with a JSON object containing EXACTLY these keys and no others:

{{
  "approved": "yes/no — 'yes' only if the analysis is correct on sign logic, addresses major
    intermediates, offers genuine mechanistic insight (not just connectivity description),
    and is internally coherent.",
  "critique": "If not approved: specific, actionable feedback (2-5 sentences). Point to concrete
    connections or intermediates where the analysis is wrong, incomplete, or superficial.
    If approved: briefly state what makes this analysis solid (1-2 sentences)."
}}

STRICT FORMAT: Return EXACTLY one JSON object with these 2 keys. No markdown, no code fences."""

CRITIC_KEYS = {"approved", "critique"}


def _call_critic(paths_formatted, net_effect_line, result, args, llama_port=None):
    """Run one critic evaluation. Returns (approved: bool, critique: str)."""
    critic_prompt = CRITIC_USER_TEMPLATE.format(
        paths_formatted=paths_formatted,
        net_effect_line=net_effect_line,
        pathway_mechanism=result.get("pathway_mechanism", ""),
        key_transformations=result.get("key_transformations", ""),
        confidence=result.get("confidence", ""),
    )

    config = resolve_model_config(args.model)
    reasoning_effort = config.get("reasoning_effort", getattr(args, "reasoning_effort", None))
    verbosity = config.get("verbosity", getattr(args, "verbosity", None))
    temperature = config.get("temperature", None)

    try:
        response_text, _ = call_llm(
            CRITIC_SYSTEM_PROMPT, critic_prompt,
            provider=args.provider,
            model_name=args.model,
            max_tokens=1024,
            llama_port=llama_port,
            thinking_budget=getattr(args, "thinking_budget", None),
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            temperature=temperature,
        )
        parsed = parse_llm_response(response_text)
        approved = parsed.get("approved", "no").strip().lower() == "yes"
        critique = parsed.get("critique", "No critique returned")
        return approved, critique
    except Exception as e:
        print(f"  [critic] Call failed: {e}")
        return True, ""  # On failure, don't block — treat as approved


def _call_actor_revision(user_prompt_data, prev_result, critique, args, llama_port=None):
    """Call the actor with its previous output and the critic's feedback.
    
    Returns the revised result dict.
    """
    revision_prompt = REVISION_PROMPT_TEMPLATE.format(
        **user_prompt_data,
        prev_mechanism=prev_result.get("pathway_mechanism", ""),
        prev_transformations=prev_result.get("key_transformations", ""),
        prev_confidence=prev_result.get("confidence", ""),
        critique=critique,
    )

    result = call_llm_with_retry(
        SYSTEM_PROMPT, revision_prompt, REQUIRED_KEYS, args, llama_port
    )
    return result


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_pair(source, target, data, args, llama_port=None):
    """Process a single (source, target) pair.
    
    If --actor-critic is enabled, runs an iterative review-revision loop.
    Otherwise, actor only (no review).
    """
    type_to_hypothesis = data["type_to_hypothesis"]

    # Find paths
    paths_df = find_paths_for_pair(
        source, target, data,
        n_steps=args.n_steps,
        threshold=args.threshold,
        side=args.side,
    )

    if paths_df is None or len(paths_df) == 0:
        return None  # No paths — skip

    # Format paths by recipient at each layer
    paths_formatted = format_paths_by_recipient(
        paths_df, source, target, type_to_hypothesis
    )

    # Compute net signed effect
    net_effect = compute_net_signed_effect(paths_df, data["type_side_to_sign"])
    net_excit, net_inhib = None, None
    if net_effect:
        net_excit, net_inhib = net_effect
        net = net_excit - net_inhib
        net_sign = "Excitatory" if net > 0 else "Inhibitory"
        net_effect_line = (
            f"Net signed effect of {source} on {target}: "
            f"{net_sign} (excit={net_excit:.5f}, inhib={net_inhib:.5f}, net={net:.5f})"
        )
    else:
        net_effect_line = f"Net signed effect: could not compute"

    # Known functional inputs to target (top 5 for context)
    known_to_target = format_known_to_target(
        data, target, args.side, args.n_steps, top_n=5
    )

    # Source and target hypotheses
    source_hyp = type_to_hypothesis.get(source, "No hypothesis available")
    target_hyp = type_to_hypothesis.get(target, "No hypothesis available")

    # Data dict for formatting prompts (reused in revision rounds)
    prompt_data = dict(
        source=source,
        target=target,
        source_hypothesis=source_hyp,
        target_hypothesis=target_hyp,
        paths_formatted=paths_formatted,
        net_effect_line=net_effect_line,
        known_to_target=known_to_target,
        side=args.side,
    )

    # --- Initial actor call ---
    user_prompt = USER_PROMPT_TEMPLATE.format(**prompt_data)
    result = call_llm_with_retry(
        SYSTEM_PROMPT, user_prompt, REQUIRED_KEYS, args, llama_port
    )

    # --- Actor-critic loop (optional) ---
    critic_flag = ""
    n_rounds = 0
    critic_history = []

    if args.actor_critic:
        max_rounds = args.max_critic_rounds

        for round_i in range(max_rounds):
            n_rounds = round_i + 1
            approved, critique = _call_critic(
                paths_formatted, net_effect_line, result, args, llama_port
            )
            critic_history.append({
                "round": n_rounds,
                "approved": approved,
                "critique": critique,
            })
            print(f"  [critic r{n_rounds}] {source}→{target}: "
                  f"{'APPROVED' if approved else 'REVISE'}"
                  f"{'' if approved else ': ' + critique[:80] + '...'}")

            if approved:
                break

            if round_i < max_rounds - 1:
                # Actor revises
                result = _call_actor_revision(
                    prompt_data, result, critique, args, llama_port
                )

        # If never approved, flag with the last critique
        if not approved:
            critic_flag = critique

    # --- Attach metadata ---
    result["source"] = source
    result["target"] = target
    result["model"] = args.model
    result["provider"] = args.provider
    result["chunk_id"] = args.chunk_id
    if net_excit is not None:
        result["net_excit"] = net_excit
        result["net_inhib"] = net_inhib
    if critic_flag:
        result["critic_flag"] = critic_flag
    if n_rounds > 0:
        result["critic_rounds"] = n_rounds
        result["critic_approved"] = critic_history[-1]["approved"] if critic_history else None
    if args.actor_critic and critic_history:
        # Store full history for debugging / analysis
        result["critic_history"] = critic_history

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1: Pairwise pathway analysis")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--n-chunks", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="./results_step1")

    parser.add_argument(
        "--actor-critic", action="store_true", default=False,
        help="Enable iterative actor-critic loop (actor revises based on critic feedback)")
    parser.add_argument(
        "--max-critic-rounds", type=int, default=2,
        help="Max critic rounds in actor-critic mode (default: 2). "
             "Each round = one critic review + one actor revision.")

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

    # Build (source, target) pairs, excluding self-pairs
    pairs = [(s, t) for s in sources for t in targets if s != t]
    print(f"Total pairs: {len(pairs)}")

    # Report mode
    if args.actor_critic:
        print(f"Critic mode: ACTOR-CRITIC LOOP (max {args.max_critic_rounds} rounds)")
    else:
        print("Critic mode: DISABLED (use --actor-critic to enable)")

    # Chunk
    chunk_size = (len(pairs) + args.n_chunks - 1) // args.n_chunks
    start = args.chunk_id * chunk_size
    end = min(start + chunk_size, len(pairs))
    chunk = pairs[start:end]
    print(f"Chunk {args.chunk_id}/{args.n_chunks}: pairs {start}-{end} ({len(chunk)} pairs)")

    # Output setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"pairwise_chunk_{args.chunk_id}.jsonl"

    # Resume
    done = set()
    if args.resume and out_file.exists():
        with open(out_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["source"], rec["target"]))
                except Exception:
                    pass
        print(f"Resuming: {len(done)} pairs already done.")
    chunk = [p for p in chunk if p not in done]

    if not chunk:
        print("All pairs in this chunk already processed. Exiting.")
        return

    # llama-server
    llama_proc = None
    llama_port = None
    if args.provider == "llama":
        llama_proc, llama_port = setup_llama_server(args)

    # Process pairs
    try:
        with open(out_file, "a") as f:
            for source, target in tqdm(chunk, desc=f"Chunk {args.chunk_id}"):
                if args.provider == "llama" and not is_llama_server_healthy(llama_port):
                    print(f"  [llama-server] Unhealthy, restarting...")
                    llama_proc = restart_llama_server(
                        args.llama_bin, args.gguf, llama_port, llama_proc
                    )
                try:
                    result = process_pair(source, target, data, args, llama_port)
                    if result is None:
                        print(f"  [skipping] No paths from {source} to {target}")
                        continue
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"  ERROR {source} → {target}: {e}")
                    error_rec = {
                        "source": source, "target": target,
                        "error": str(e), "chunk_id": args.chunk_id,
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