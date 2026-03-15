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
import re
import subprocess
import time
import socket
from functools import reduce
from pathlib import Path

import dotenv
import pandas as pd
import scipy as sp
from tqdm import tqdm


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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Provider defaults & model config (from notebook)
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS = {
    "llama":     "qwen3.5:35b",   # just a label, actual model is the GGUF
    "openai":    "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}

# Models that use the Responses API (supports verbosity, reasoning)
RESPONSES_API_MODELS = {"gpt-5", "gpt-5.4"}

MODEL_DEFAULTS = {
    "gpt-4o":       {"temperature": 1.0},
    "gpt-4-turbo":  {"temperature": 1.0},
    "gpt-4":        {"temperature": 1.0},
    "gpt-5":        {"temperature": 1.0},
    "gpt-5.4":      {"temperature": 1.0, "reasoning_effort": "medium", "verbosity": "low"},
}


def _resolve_model_config(model_name, extra_config=None):
    """Apply model-specific defaults then override with any extra_config."""
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
    """Start llama-server as a background process. Returns the process."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/public/gcc/12_2_0/lib64:" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        llama_bin,
        "-m", gguf_path,
        "--port", str(port),
        "-ngl", "99",
        "--ctx-size", "24576",
        "--no-mmap",
        "-np", "1",
        # Note: --reasoning-format none would disable thinking but requires a newer build.
        # Instead, <think>...</think> blocks are stripped from responses in call_llama().
    ]
    print(f"[llama-server] Starting on port {port}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    return proc


def is_llama_server_healthy(port):
    """Check if llama-server is still responding."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
            data = json.loads(r.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def wait_for_llama_server(port, timeout=300):
    """Poll until llama-server is ready and can generate tokens."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                data = json.loads(r.read())
                if data.get("status") == "ok":
                    print(f"[llama-server] Health check passed on port {port}, waiting for model load...")
                    # Health check passes before weights are fully loaded into GPU —
                    # wait an extra 30s to be safe
                    time.sleep(30)
                    print(f"[llama-server] Ready on port {port}")
                    return True
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"llama-server did not start within {timeout}s")


def restart_llama_server(llama_bin, gguf_path, port, old_proc=None):
    """Terminate old server and start a fresh one."""
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
    """Call local llama-server via OpenAI-compatible API."""
    from openai import OpenAI
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="not-needed",
    )
    # Qwen3.5 puts extended thinking in content as <think>...</think>.
    # These are stripped after each response; only the JSON answer is kept.
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
        # This build puts thinking in content as <think>...</think>.
        # Strip it and keep only what comes after.
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
        # Responses API — supports verbosity and reasoning_effort (gpt-5, gpt-5.4)
        api_params = {
            "model": model_name,
            "input": user_prompt,
            "instructions": system_prompt,
            "max_completion_tokens": max_tokens,
            "store": True,
        }
        if verbosity:
            api_params["text"] = {"verbosity": verbosity}
        if reasoning_effort:
            api_params["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(**api_params)
        return response.output_text, response

    else:
        # Chat Completions API — standard models
        api_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
        }
        # o1-style models don't support temperature
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
        "model": model_name,
        "max_tokens": max_tokens,
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


USER_PROMPT_TEMPLATE = """Cell Type: {cell_type}

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
  "intermediate_computations": "For each key intermediate: What functional operation does it perform? How do its inputs (sensory + characterized) combine to create a specific transformation? Why might this circuit route signals through these intermediates rather than directly?",

  "circuit_logic": "Describe the complete information flow: Which inputs are integrated? How do excitatory and inhibitory pathways work together to enable the computation? What does the overall routing suggest about selective combination, gating, modulation, or comparison of inputs?",

  "inferred_computation": "State the core adaptive problem this cell type solves. Be mechanistic and grounded in the connectivity. Synthesize across all inputs if possible. If multiple distinct interpretations exist, present them as alternatives. Mark each: [grounded] (directly supported by wiring), [likely] (coherent with known motifs), or [speculative] (requires additional mechanisms).",

  "behavioral_context": "In what sensory scene, motor state, or task-context would this circuit be active? What adaptive behavior or decision might depend on this computation?",

  "confidence": "high/medium/low (based on coherence of the computation and completeness of input integration)",

  "coherence_assessment": "How well do all the inputs (sensory, characterized, via intermediates) fit together into a single coherent function? Are there unexplained features? Does every major input category contribute meaningfully to the hypothesis?",

  "key_supporting_connections": "Which connectivity patterns most strongly support your main hypothesis?"
}}"""


# ---------------------------------------------------------------------------
# Circuit formatting (from notebook)
# ---------------------------------------------------------------------------

def format_series(series, top_n):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0] if series.shape[1] == 1 else series.sum(axis=1)
    filtered = series.sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)
    rows = []
    for k, v in filtered.items():
        sign = "Excitatory" if v > 0 else "Inhibitory"
        rows.append(f"  - {k}: {v:.3f} ({sign})")
    return "\n".join(rows) if rows else "  None found", filtered


def format_circuit_data(cell_type, intermediate_to_cell, known_to_intermediate,
                        sensory_to_cell=None, known_to_cell=None, cell_to_known=None,
                        top_n=15):
    # Intermediate → cell
    inter_table, inter_filtered = format_series(intermediate_to_cell, top_n)

    # Known → intermediates
    if isinstance(known_to_intermediate, pd.DataFrame):
        valid = [c for c in known_to_intermediate.columns if c in inter_filtered.index]
        kdf = known_to_intermediate[valid]
        known_rows = []
        conn_count = 0
        for intermediate in inter_filtered.index:
            if intermediate not in kdf.columns:
                continue
            col = kdf[intermediate].dropna()
            col = col[col.abs() > 1e-6]
            if len(col) == 0:
                continue
            for kf, w in col.sort_values(key=lambda x: x.abs(), ascending=False).head(5).items():
                sign = "Excitatory" if w > 0 else "Inhibitory"
                known_rows.append(f"  - {kf} → {intermediate}: {w:.3f} ({sign})")
                conn_count += 1
        known_table = "\n".join(known_rows) if known_rows else "  None found"
    else:
        known_table, _ = format_series(known_to_intermediate, top_n)
        conn_count = top_n

    sensory_table = "  None found"
    if sensory_to_cell is not None:
        t, _ = format_series(sensory_to_cell, top_n)
        sensory_table = t

    known_dn_table = "  None found"
    if known_to_cell is not None:
        t, _ = format_series(known_to_cell, top_n)
        known_dn_table = t

    dn_output_table = "  None found"
    if cell_to_known is not None:
        t, _ = format_series(cell_to_known, top_n)
        dn_output_table = t

    return {
        "cell_type": cell_type,
        "intermediate_table": inter_table,
        "known_table": known_table,
        "sensory_table": sensory_table,
        "known_dn_table": known_dn_table,
        "dn_output_table": dn_output_table,
        "num_intermediates": len(inter_filtered),
        "num_known": conn_count,
    }


# ---------------------------------------------------------------------------
# Circuit connectivity (from notebook — uses module-level globals)
# ---------------------------------------------------------------------------

def calculate_circuit_connectivity(dn_type, n_steps, add_function=True):
    from connectome_interpreter import signed_conn_by_path_length_data, el_within_n_steps

    # Sensory → DN
    esensdn, isensdn = signed_conn_by_path_length_data(
        inprop,
        meta.idx[(meta.super_class == "sensory") & ~meta.cell_type.str.isdigit()],
        meta.idx[meta.cell_type == dn_type],
        n_steps, type_to_sign, idx_to_type, idx_to_type,
    )
    esensdn_sum = reduce(lambda a, b: a.add(b, fill_value=0), esensdn)
    isensdn_sum = reduce(lambda a, b: a.add(b, fill_value=0), isensdn)
    difference_sensdn = esensdn_sum.subtract(isensdn_sum, fill_value=0)
    if add_function:
        difference_sensdn.index = difference_sensdn.index.map(cell_type_to_function)

    # Circuit within n steps
    circuit = el_within_n_steps(
        inprop,
        meta.idx[(meta.super_class == "sensory") & ~meta.cell_type.str.isdigit()],
        meta.idx[meta.cell_type == dn_type],
        n_steps, 0.01, idx_to_type, idx_to_type,
    )

    # Direct intermediate → DN
    intermediate_dn = circuit[(circuit.post == dn_type) & (circuit.pre != dn_type)].copy()
    intermediate_dn["pre_sign"] = intermediate_dn.pre.map(type_to_sign)
    intermediate_dn["weight"] = intermediate_dn.weight * intermediate_dn.pre_sign
    intermediate_dn = intermediate_dn.set_index("pre")["weight"]

    # Known → intermediates upstream of DN
    ekunk, ikunk = signed_conn_by_path_length_data(
        inprop,
        meta.idx[(meta.cell_type != meta.known_function) & (meta.cell_type != dn_type)],
        meta.idx[meta.cell_type.isin(circuit.pre[(circuit.post == dn_type) & (circuit.pre != dn_type)])],
        n_steps, type_to_sign, idx_to_type, idx_to_type,
    )
    ekunk_sum = reduce(lambda a, b: a.add(b, fill_value=0), ekunk)
    ikunk_sum = reduce(lambda a, b: a.add(b, fill_value=0), ikunk)
    difference_kunk = ekunk_sum.subtract(ikunk_sum, fill_value=0)
    if add_function:
        difference_kunk.index = difference_kunk.index.map(cell_type_to_function)

    # Known → DN
    ekdn, ikdn = signed_conn_by_path_length_data(
        inprop,
        meta.idx[(meta.cell_type != meta.known_function) & (meta.cell_type != dn_type)],
        meta.idx[meta.cell_type == dn_type],
        n_steps, type_to_sign, idx_to_type, idx_to_type,
    )
    ekdn_sum = reduce(lambda a, b: a.add(b, fill_value=0), ekdn)
    ikdn_sum = reduce(lambda a, b: a.add(b, fill_value=0), ikdn)
    difference_kdn = ekdn_sum.subtract(ikdn_sum, fill_value=0)
    if add_function:
        difference_kdn.index = difference_kdn.index.map(cell_type_to_function)

    # DN → known
    ednk, idnk = signed_conn_by_path_length_data(
        inprop,
        meta.idx[meta.cell_type == dn_type],
        meta.idx[(meta.cell_type != meta.known_function) & (meta.cell_type != dn_type)],
        n_steps, type_to_sign, idx_to_type, idx_to_type,
    )
    ednk_sum = reduce(lambda a, b: a.add(b, fill_value=0), ednk)
    idnk_sum = reduce(lambda a, b: a.add(b, fill_value=0), idnk)
    difference_dnk = ednk_sum.subtract(idnk_sum, fill_value=0).T
    if add_function:
        difference_dnk.index = difference_dnk.index.map(cell_type_to_function)

    return difference_sensdn, intermediate_dn, difference_kunk, difference_kdn, difference_dnk


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _try_parse(text):
    """Attempt json.loads, return parsed dict or raise."""
    return json.loads(text)


def _strip_markdown_fences(text):
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    # Remove all ```json and ``` markers, then strip whitespace
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


def _merge_json_blocks(text):
    """
    If the model output multiple separate JSON objects (e.g. one per field),
    merge them into a single dict.
    """
    blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL)
    # Also try finding top-level {...} blocks with nested braces
    # Use a brace-counting approach for robustness
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
    """
    Fix common model-generated JSON syntax errors:
    - Trailing commas before } or ]
    - Missing comma between adjacent string/value pairs
    - Stray * characters in keys
    - Single-quoted strings (limited)
    """
    # Strip markdown fences first
    text = _strip_markdown_fences(text)
    # Remove stray * characters that appear at start of keys
    text = re.sub(r'\*(\w)', r'\1', text)
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    # Truncated strings at end — if JSON is cut off, truncate cleanly
    # (handled separately by truncation recovery)
    return text


def _recover_truncated_json(text):
    """
    If the JSON was cut off (max_tokens), try to recover the fields
    that were successfully parsed before truncation.
    Extract all complete "key": value pairs at the top level.
    """
    result = {}
    # Match complete string-valued fields
    for m in re.finditer(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        result[m.group(1)] = m.group(2)
    # Match complete non-string fields (numbers, booleans, null)
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
    """
    Multi-strategy JSON parser handling:
    1. Clean JSON
    2. Markdown-fenced JSON
    3. Multiple JSON blocks (merge them)
    4. Common syntax errors (trailing commas, stray *)
    5. Truncated JSON (recover partial fields)
    """
    if not text or not text.strip():
        return {"error": "Empty response"}

    strategies = [
        # 1. Direct parse
        lambda t: _try_parse(t),
        # 2. Strip markdown fences
        lambda t: _try_parse(_strip_markdown_fences(t)),
        # 3. Fix common syntax errors then parse
        lambda t: _try_parse(_fix_common_syntax_errors(t)),
        # 4. Merge multiple JSON blocks
        lambda t: _merge_json_blocks(t),
        # 5. Find first top-level {...} and fix/parse it
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

    # Last resort: recover whatever key-value pairs we can from truncated output
    partial = _recover_truncated_json(text)
    if partial:
        partial["error"] = "truncated_response_partial_recovery"
        return partial

    return {"error": "JSON parse failed", "raw_response": text}


# ---------------------------------------------------------------------------
# Per-cell processing
# ---------------------------------------------------------------------------

def process_cell(cell_type, args, llama_port=None):
    # Circuit computation
    (difference_sensdn, intermediate_dn, difference_kunk,
     difference_kdn, difference_dnk) = calculate_circuit_connectivity(
        cell_type, args.n_steps, add_function=args.add_function
    )

    # Format data
    fmt = format_circuit_data(
        cell_type=cell_type,
        intermediate_to_cell=intermediate_dn,
        known_to_intermediate=difference_kunk,
        sensory_to_cell=difference_sensdn,
        known_to_cell=difference_kdn,
        cell_to_known=difference_dnk,
        top_n=args.top_n,
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(**fmt)

    # Resolve model config (applies model-specific defaults like reasoning_effort for gpt-5.4)
    config = _resolve_model_config(args.model)
    reasoning_effort = config.get("reasoning_effort", args.reasoning_effort)
    verbosity = config.get("verbosity", args.verbosity)
    temperature = config.get("temperature", None)

    # LLM call
    response_text, _ = call_llm(
        SYSTEM_PROMPT, user_prompt,
        provider=args.provider,
        model_name=args.model,
        max_tokens=args.max_tokens,
        llama_port=llama_port,
        thinking_budget=args.thinking_budget,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        temperature=temperature,
    )

    result = parse_llm_response(response_text)

    result["cell_type"] = cell_type
    result["model"] = args.model
    result["provider"] = args.provider
    result["add_function"] = args.add_function
    result["chunk_id"] = args.chunk_id
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load env + set model default
    dotenv.load_dotenv()
    if args.model is None:
        args.model = PROVIDER_DEFAULTS[args.provider]

    # ---- Load data ----
    print("Loading connectome data...")
    base_path = args.base_path

    global inprop, meta, cell_type_to_function, type_to_sign, idx_to_type

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

    idx_to_type = dict(zip(meta.idx, meta.cell_type))
    type_to_sign = dict(zip(meta.cell_type, meta.sign))

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

    if not chunk:
        print("All cells in this chunk already processed. Exiting.")
        return

    # ---- Start llama-server if needed ----
    llama_proc = None
    llama_port = None
    if args.provider == "llama":
        llama_port = args.llama_port or find_free_port()
        llama_proc = start_llama_server(args.llama_bin, args.gguf, llama_port)
        try:
            wait_for_llama_server(llama_port)
        except TimeoutError as e:
            print(f"ERROR: {e}")
            llama_proc.terminate()
            raise
        # Warmup: poll until the model actually generates tokens
        print("[llama-server] Warming up model...")
        warmup_deadline = time.time() + 120
        warmed = False
        while time.time() < warmup_deadline:
            try:
                call_llama("You are helpful.", "Say OK.", llama_port, max_tokens=512)
                print("[llama-server] Warmup complete.")
                warmed = True
                break
            except Exception as e:
                print(f"[llama-server] Warmup attempt failed ({e}), retrying in 10s...")
                time.sleep(10)
        if not warmed:
            llama_proc.terminate()
            raise RuntimeError("llama-server failed to generate tokens after 120s warmup")

    # ---- Process cells ----
    try:
        with open(out_file, "a") as f:
            for cell_type in tqdm(chunk, desc=f"Chunk {args.chunk_id}"):
                # Check server health before each cell; restart if needed
                if args.provider == "llama":
                    if not is_llama_server_healthy(llama_port):
                        print(f"  [llama-server] Unhealthy before {cell_type}, restarting...")
                        llama_proc = restart_llama_server(
                            args.llama_bin, args.gguf, llama_port, llama_proc
                        )
                try:
                    result = process_cell(cell_type, args, llama_port=llama_port)
                    f.write(json.dumps(result) + "\n")
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