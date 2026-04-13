"""
circuit_utils.py — Shared utilities for the circuit analysis pipeline.

Contains:
  - LLM provider wrappers (llama, openai, anthropic)
  - JSON parsing / validation
  - Connectome data loading
  - Path formatting for LLM prompts
"""

import argparse
import json
import os
import re
import socket
import subprocess
import time
from functools import reduce
from pathlib import Path

import pandas as pd
import scipy as sp


# ---------------------------------------------------------------------------
# Provider defaults & model config
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS = {
    "llama":     "qwen3.5:35b",
    "openai":    "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}

RESPONSES_API_MODELS = {"gpt-5", "gpt-5.4"}

MODEL_DEFAULTS = {
    "gpt-4o":       {"temperature": 1.0},
    "gpt-4-turbo":  {"temperature": 1.0},
    "gpt-4":        {"temperature": 1.0},
    "gpt-5":        {"temperature": 1.0},
    "gpt-5.4":      {"temperature": 1.0, "reasoning_effort": "medium", "verbosity": "low"},
}


def resolve_model_config(model_name, extra_config=None):
    config = {"model": model_name}
    if model_name in MODEL_DEFAULTS:
        config.update(MODEL_DEFAULTS[model_name])
    if extra_config:
        config.update(extra_config)
    return config


def _use_responses_api(model_name):
    return model_name in RESPONSES_API_MODELS


# ---------------------------------------------------------------------------
# llama-server management
# ---------------------------------------------------------------------------

def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_llama_server(llama_bin, gguf_path, port):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/public/gcc/12_2_0/lib64:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        llama_bin, "-m", gguf_path,
        "--port", str(port),
        "-ngl", "99", "--ctx-size", "24576",
        "--no-mmap", "-np", "1",
    ]
    print(f"[llama-server] Starting on port {port}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    return proc


def is_llama_server_healthy(port):
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def wait_for_llama_server(port, timeout=300):
    import urllib.request
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    print(f"[llama-server] Health check passed on port {port}, waiting for model load...")
                    time.sleep(30)
                    print(f"[llama-server] Ready on port {port}")
                    return True
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"llama-server did not start within {timeout}s")


def restart_llama_server(llama_bin, gguf_path, port, old_proc=None):
    if old_proc is not None:
        print("[llama-server] Restarting...")
        old_proc.terminate()
        try:
            old_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            old_proc.kill()
    proc = start_llama_server(llama_bin, gguf_path, port)
    wait_for_llama_server(port)
    print("[llama-server] Warmup after restart...")
    try:
        call_llama("You are helpful.", "Say OK.", port, max_tokens=512)
    except Exception:
        pass
    return proc


# ---------------------------------------------------------------------------
# LLM call wrappers
# ---------------------------------------------------------------------------

def call_llama(system_prompt, user_prompt, port, max_tokens=4096, retries=3):
    from openai import OpenAI
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")
    for attempt in range(retries):
        response = client.chat.completions.create(
            model="",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if text and text.strip():
            return text, response
        print(f"  [llama] Empty response on attempt {attempt + 1}/{retries}, retrying...")
        time.sleep(5)
    raise ValueError(f"llama-server returned empty response after {retries} attempts")


def call_openai(system_prompt, user_prompt, model_name, max_tokens=4096,
                reasoning_effort=None, verbosity=None, temperature=None):
    from openai import OpenAI
    client = OpenAI()
    if _use_responses_api(model_name):
        api_params = {
            "model": model_name, "input": user_prompt,
            "instructions": system_prompt, "max_completion_tokens": max_tokens,
            "store": True,
        }
        if verbosity:
            api_params["text"] = {"verbosity": verbosity}
        if reasoning_effort:
            api_params["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(**api_params)
        return response.output_text, response
    else:
        api_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None and "o1" not in model_name.lower():
            api_params["temperature"] = temperature
        if reasoning_effort and "o1" in model_name.lower():
            api_params["reasoning_effort"] = reasoning_effort
        response = client.chat.completions.create(**api_params)
        return response.choices[0].message.content, response


def call_anthropic(system_prompt, user_prompt, model_name, max_tokens=4096,
                   thinking_budget=None):
    from anthropic import Anthropic
    client = Anthropic()
    api_params = {
        "model": model_name, "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if thinking_budget:
        api_params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    response = client.messages.create(**api_params)
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    return text, response


def call_llm(system_prompt, user_prompt, provider, model_name, max_tokens,
             llama_port=None, thinking_budget=None, reasoning_effort=None,
             verbosity=None, temperature=None):
    if provider == "llama":
        return call_llama(system_prompt, user_prompt, llama_port, max_tokens)
    elif provider == "openai":
        return call_openai(system_prompt, user_prompt, model_name, max_tokens,
                           reasoning_effort=reasoning_effort, verbosity=verbosity,
                           temperature=temperature)
    elif provider == "anthropic":
        return call_anthropic(system_prompt, user_prompt, model_name, max_tokens,
                              thinking_budget=thinking_budget)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# JSON parsing (same robust multi-strategy parser)
# ---------------------------------------------------------------------------

def _try_parse(text):
    return json.loads(text)


def _strip_markdown_fences(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


def _merge_json_blocks(text):
    merged = {}
    depth = 0
    start = None
    candidates = []
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i+1])
                start = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                merged.update(parsed)
        except json.JSONDecodeError:
            pass
    return merged if merged else None


def _fix_common_syntax_errors(text):
    text = _strip_markdown_fences(text)
    text = re.sub(r'\*(\w)', r'\1', text)
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    return text


def _recover_truncated_json(text):
    result = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        result[m.group(1)] = m.group(2)
    for m in re.finditer(r'"(\w+)"\s*:\s*(true|false|null|-?\d+\.?\d*)', text):
        v = m.group(2)
        if v == 'true': v = True
        elif v == 'false': v = False
        elif v == 'null': v = None
        else:
            try: v = float(v) if '.' in v else int(v)
            except ValueError: pass
        result[m.group(1)] = v
    return result if result else None


def parse_llm_response(text):
    if not text or not text.strip():
        return {"error": "Empty response"}
    strategies = [
        lambda t: _try_parse(t),
        lambda t: _try_parse(_strip_markdown_fences(t)),
        lambda t: _try_parse(_fix_common_syntax_errors(t)),
        lambda t: _merge_json_blocks(t),
        lambda t: _try_parse(_fix_common_syntax_errors(
            re.search(r'\{.*\}', t, re.DOTALL).group()
        )) if re.search(r'\{.*\}', t, re.DOTALL) else None,
    ]
    for strategy in strategies:
        try:
            result = strategy(text)
            if result and isinstance(result, dict) and len(result) > 0:
                return result
        except Exception:
            continue
    partial = _recover_truncated_json(text)
    if partial:
        partial["error"] = "truncated_response_partial_recovery"
        return partial
    return {"error": "JSON parse failed", "raw_response": text}


def validate_response(parsed, required_keys):
    """Check parsed dict has exactly the required keys. Returns (ok, reason)."""
    if "error" in parsed and not required_keys.intersection(parsed.keys()):
        return False, f"parse error: {parsed.get('error')}"
    missing = required_keys - parsed.keys()
    # Allow metadata keys we add ourselves
    meta_keys = {"source", "target", "model", "provider", "chunk_id",
                 "validation_error", "sources", "targets",
                 "net_excit", "net_inhib", "critic_flag",
                 "critic_rounds", "critic_approved", "critic_history"}
    extra = parsed.keys() - required_keys - meta_keys
    if missing:
        return False, f"missing keys: {missing}"
    if extra:
        return False, f"extra keys present: {extra}"
    return True, ""


MAX_RETRIES = 3


def call_llm_with_retry(system_prompt, user_prompt, required_keys, args,
                         llama_port=None):
    """Call LLM with retry logic and JSON validation."""
    config = resolve_model_config(args.model)
    reasoning_effort = config.get("reasoning_effort", getattr(args, "reasoning_effort", None))
    verbosity = config.get("verbosity", getattr(args, "verbosity", None))
    temperature = config.get("temperature", None)

    last_result = None
    for attempt in range(MAX_RETRIES):
        response_text, _ = call_llm(
            system_prompt, user_prompt,
            provider=args.provider,
            model_name=args.model,
            max_tokens=args.max_tokens,
            llama_port=llama_port,
            thinking_budget=getattr(args, "thinking_budget", None),
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            temperature=temperature,
        )
        result = parse_llm_response(response_text)
        ok, reason = validate_response(result, required_keys)
        if ok:
            return result
        print(f"  [retry {attempt+1}/{MAX_RETRIES}] {reason}")
        last_result = result

    result = last_result or result
    result["validation_error"] = reason
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_connectome_data(base_path, known_types_csv):
    """Load connectome data and build all lookup dicts.
    
    Returns a dict with all the data structures needed by the pipeline.
    """
    from connectome_interpreter import find_paths_of_length, group_paths, filter_paths

    cell_type_to_function_df = pd.read_csv(os.path.expanduser(known_types_csv))
    cell_type_to_function = dict(
        zip(cell_type_to_function_df.cell_type, cell_type_to_function_df.known_function)
    )

    inprop = sp.sparse.load_npz(
        os.path.join(base_path, "data", "fafb_all_neuron", "fafb_inprop_all_neuron.npz")
    )
    meta = pd.read_csv(
        os.path.join(base_path, "data", "fafb_all_neuron", "fafb_all_neuron_meta.csv"),
        index_col=0, low_memory=False,
    )
    meta["type_side"] = meta.cell_type + "_" + meta.side

    idx_to_type_side = dict(zip(meta.idx, meta.type_side))
    idx_to_type = dict(zip(meta.idx, meta.cell_type))
    type_to_sign = dict(zip(meta.cell_type, meta.sign))
    type_side_to_sign = dict(zip(meta.type_side, meta.sign))
    type_side_to_type = dict(zip(meta.type_side, meta.cell_type))

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


# ---------------------------------------------------------------------------
# Common CLI arguments
# ---------------------------------------------------------------------------

def add_common_args(parser):
    """Add CLI arguments shared across all pipeline steps."""
    parser.add_argument("--provider", type=str, default="llama",
                        choices=["llama", "openai", "anthropic"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--gguf", type=str,
                        default="/cephfs2/yyin/huggingface/hub/qwen35_gguf/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf")
    parser.add_argument("--llama-bin", type=str,
                        default=os.path.expanduser("~/llama.cpp/build/bin/llama-server"))
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
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"])
    parser.add_argument("--verbosity", type=str, default=None,
                        choices=["low", "medium", "high"])
    parser.add_argument("--llama-port", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def load_types_file(path):
    """Load the circuit_types.json file."""
    with open(path) as f:
        types = json.load(f)
    sources = types["sources"]
    targets = types["targets"]
    print(f"Sources ({len(sources)}): {sources}")
    print(f"Targets ({len(targets)}): {targets}")
    return sources, targets



# ---------------------------------------------------------------------------
# llama-server lifecycle for scripts
# ---------------------------------------------------------------------------

def setup_llama_server(args):
    """Start and warm up llama-server. Returns (proc, port)."""
    port = args.llama_port or find_free_port()
    proc = start_llama_server(args.llama_bin, args.gguf, port)
    try:
        wait_for_llama_server(port)
    except TimeoutError as e:
        proc.terminate()
        raise

    print("[llama-server] Warming up model...")
    warmup_deadline = time.time() + 120
    warmed = False
    while time.time() < warmup_deadline:
        try:
            call_llama("You are helpful.", "Say OK.", port, max_tokens=512)
            print("[llama-server] Warmup complete.")
            warmed = True
            break
        except Exception as e:
            print(f"[llama-server] Warmup attempt failed ({e}), retrying in 10s...")
            time.sleep(10)
    if not warmed:
        proc.terminate()
        raise RuntimeError("llama-server failed to generate tokens after 120s warmup")

    return proc, port