"""
function_extraction_hpc.py — Cell type function extraction from PDFs for HPC SLURM array jobs.

Usage:
    python function_extraction_hpc.py --chunk-id 0 --n-chunks 4 [options]

The script:
  1. Loads a CSV of papers with PDF paths (discovered_papers_with_local.csv)
  2. Splits the paper list into N equal chunks
  3. Processes chunk --chunk-id, extracting cell types and functions via LLM
  4. Optionally starts a local llama-server (llama.cpp) per job
  5. Saves results incrementally to results/results_chunk_{chunk_id}.jsonl

Provider options:
  --provider llama     : local llama-server (llama.cpp OpenAI-compatible API)
  --provider openai    : OpenAI API (needs OPENAI_API_KEY)
  --provider anthropic : Anthropic API (needs ANTHROPIC_API_KEY)

Extraction modes:
  --mode brief         : Extract cell type + 6-word function summary
  --mode experimental  : Extract full experimental details (stimulus, activity, behaviour, etc.)

Example SLURM array job:
  #SBATCH --array=0-7
  python function_extraction_hpc.py \
      --chunk-id $SLURM_ARRAY_TASK_ID \
      --n-chunks 8 \
      --provider llama \
      --gguf /path/to/model.gguf \
      --papers-csv /path/to/discovered_papers_with_local.csv \
      --mode experimental
"""

import argparse
import json
import os
import re
import socket
import subprocess
import time
import traceback
from pathlib import Path

import dotenv
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Cell type function extraction batch job")
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
    p.add_argument("--papers-csv", type=str,default=None,
                   help="Path to CSV with paper metadata and PDF paths")
    p.add_argument("--papers-dir", type=str, default=None,
                    help="Base directory for papers in papers-csv")
    p.add_argument("--output-dir", type=str, default="./results",
                   help="Directory to save output JSONL files")
    p.add_argument("--mode", type=str, default="experimental",
                   choices=["brief", "experimental"],
                   help="Extraction mode: 'brief' (6-word summary) or 'experimental' (full details)")
    p.add_argument("--use-rag", action=argparse.BooleanOptionalAction, default=False,
                   help="Use RAG (ChromaDB + OpenAI embeddings) instead of full-text. "
                        "Requires OPENAI_API_KEY even with non-OpenAI LLM providers.")
    p.add_argument("--chroma-dir", type=str, default=None,
                   help="If set, persist ChromaDB embeddings here (one subdir per chunk). "
                        "Speeds up re-runs by caching embeddings. Only used with --use-rag. "
                        "If unset, an in-memory (ephemeral) client is used instead.")
    p.add_argument("--ctx-size", type=int, default=65536,
                   help="llama-server context window size in tokens (default 65536). "
                        "or use --max-chars to trim input instead.")
    p.add_argument("--max-chars", type=int, default=None,
                   help="Truncate paper text to this many characters (None = no limit). "
                        "~4 chars per token, so 120000 ≈ 30k tokens.")
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="Max tokens for LLM response")
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="Anthropic extended thinking budget (tokens)")
    p.add_argument("--reasoning-effort", type=str, default=None,
                   choices=["low", "medium", "high"],
                   help="OpenAI reasoning effort")
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable Qwen3 thinking mode (llama provider only). "
                        "Prepends /think to the prompt; strips <think> blocks when disabled (default). "
                        "Use --no-think to explicitly suppress thinking (default behaviour).")
    p.add_argument("--llama-port", type=int, default=None,
                   help="Port for llama-server (auto-assigned if None)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip already-processed papers (--no-resume to disable)")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Delay (seconds) between API calls for rate limiting")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Provider defaults & model config
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS = {
    "llama":     "qwen3.5:35b",   # label only; actual model is the GGUF
    "openai":    "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}

RESPONSES_API_MODELS = {"gpt-5", "gpt-5.4"}

MODEL_DEFAULTS = {
    "gpt-4o":      {"temperature": 1.0},
    "gpt-4-turbo": {"temperature": 1.0},
    "gpt-4":       {"temperature": 1.0},
    "gpt-5":       {"temperature": 1.0},
    "gpt-5.4":     {"temperature": 1.0, "reasoning_effort": "medium", "verbosity": "low"},
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


def start_llama_server(llama_bin, gguf_path, port, ctx_size=65536):
    """Start llama-server as a background process. Returns the process."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/public/gcc/12_2_0/lib64:" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        llama_bin,
        "-m", gguf_path,
        "--port", str(port),
        "-ngl", "99",
        "--ctx-size", str(ctx_size),
        "--no-mmap",
        "-np", "1",
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
    """Poll until llama-server is ready."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                data = json.loads(r.read())
                if data.get("status") == "ok":
                    print(f"[llama-server] Health check passed on port {port}, waiting for model load...")
                    time.sleep(30)
                    print(f"[llama-server] Ready on port {port}")
                    return True
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"llama-server did not start within {timeout}s")


def restart_llama_server(llama_bin, gguf_path, port, old_proc=None, ctx_size=65536):
    """Terminate old server and start a fresh one."""
    if old_proc is not None:
        print("[llama-server] Restarting...")
        old_proc.terminate()
        try:
            old_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            old_proc.kill()
    proc = start_llama_server(llama_bin, gguf_path, port, ctx_size=ctx_size)
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

def call_llama(system_prompt, user_prompt, port, max_tokens=8000, think=False, retries=3):
    """
    Call local llama-server via OpenAI-compatible API.

    think=False (default): prepend /no_think to suppress Qwen3 reasoning and
        strip any <think>...</think> blocks that slip through anyway.
    think=True: prepend /think to enable reasoning; the model emits a
        <think>...</think> block then the final answer. Both are returned
        so the caller can log or inspect the reasoning if desired.
    """
    from openai import OpenAI
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

    # Qwen3 soft-switches: /think and /no_think in the user turn control reasoning mode.
    # Prepend to user_prompt rather than system_prompt — the model checks the user turn.
    prefix = "/think\n" if think else "/no_think\n"
    user_prompt_prefixed = prefix + user_prompt

    for attempt in range(retries):
        response = client.chat.completions.create(
            model="",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt_prefixed},
            ],
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        if not think:
            # Strip any <think>...</think> that slipped through despite /no_think
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if text:
            return text, response
        print(f"  [llama] Empty response on attempt {attempt + 1}/{retries}, retrying...")
        time.sleep(5)
    raise ValueError(f"llama-server returned empty response after {retries} attempts")


def call_openai(system_prompt, user_prompt, model_name, max_tokens=8000,
                reasoning_effort=None, verbosity=None, temperature=None):
    from openai import OpenAI
    client = OpenAI()

    if _use_responses_api(model_name):
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
        api_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None and "o1" not in model_name.lower():
            api_params["temperature"] = temperature
        if reasoning_effort and "o1" in model_name.lower():
            api_params["reasoning_effort"] = reasoning_effort
        response = client.chat.completions.create(**api_params)
        return response.choices[0].message.content, response


def call_anthropic(system_prompt, user_prompt, model_name, max_tokens=8000,
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
             llama_port=None, think=False, thinking_budget=None, reasoning_effort=None,
             verbosity=None, temperature=None):
    if provider == "llama":
        return call_llama(system_prompt, user_prompt, llama_port, max_tokens, think=think)
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
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BRIEF = """You are a neuroscience research assistant specialising in Drosophila.

Extract cell type names and behavioral/sensory functions from experimental papers.
Focus on what neurons DO (behaviors triggered, sensory inputs processed), not molecular mechanisms.
Be precise and concise — use technical terminology from the field."""

SYSTEM_PROMPT_EXPERIMENTAL = """You are an expert neuroscience research analyst specialising in Drosophila neuroscience.

Your task is to extract factual experimental findings about neural cell types from scientific papers.
CRITICAL: Extract ONLY what is explicitly stated in the paper. Do not infer, speculate, or
incorporate interpretations from other papers cited.

Be systematic: for each cell type, ensure you capture:
- Exact stimulus conditions tested
- Measured neural activity (include technique)
- Behavioral outcomes
- How this relates to natural ecological contexts

Do NOT:
- Add general knowledge or speculation
- Cite findings from other papers
- Interpret beyond what the experiments show
- Make high-level functional inferences

DO:
- Quote or closely paraphrase exact findings
- Mention experimental techniques used
- Include multiple aspects and cell types if the paper tested them
- Note ecological relevance only if explicitly discussed"""

EXTRACTION_QUERY_BRIEF = """Extract cell types and their experimentally-tested functions from this paper.

RULES:
- Only include neurons tested experimentally in THIS paper (not cited from others)
- Focus on behavioral/sensory functions, not generic responses like "action potentials"
- Keep known_function extremely concise: MAX 6 words
- Return empty if no experimental function testing found

GOOD EXAMPLES of known_function format:
- "visual_chromatic"
- "oviposition; egg_laying"
- "courtship promotion"
- "grooming"
- "walking"
- "turning; flight"
- "olfactory; aversive"
- "visual_loom; backwards; freezing"
- "pheromone"
- "neck_control; neck_motor"
- "appetitive_learning; appetitive"

BAD EXAMPLES (too generic):
- "produces action potentials"
- "responds to stimulation"
- "synaptic transmission"

Return JSON format:
{
  "cell_types": [
    {"cell_type": "AOTU019", "known_function": "steer object towards midline"},
    {"cell_type": "DNp10",   "known_function": "landing"}
  ]
}

If no experimental data: return empty object {}
"""

EXTRACTION_QUERY_EXPERIMENTAL = """Extract cell types and their experimentally-determined functions from this paper.

Focus ONLY on what the paper experimentally tested and found. Extract facts from the paper, not speculation.

For each cell type, include:
1. Cell type name
2. Experimental stimulus (if tested): what stimulus was presented
3. Neural activity: how the neuron responded (firing, calcium, electrophysiology, etc.)
4. Behavioral consequence: what behavior changed when the neuron was activated/inactivated
5. Ecological context: when/why this might matter in the fly's life
6. Summary: a concise summary of the above, omitting what's not tested.

RULES:
- ONLY extract findings explicitly described in THIS paper
- Do NOT infer, speculate, or cite other papers' interpretations
- Cite which experimental technique was used (imaging, optogenetics, electrophysiology, etc.)
- Keep descriptions concise (generally no need for precise numbers)
- Include multiple behavioral/sensory aspects if the paper tested them
- If an aspect is not tested or described, leave it blank
- Cell type naming: Use the exact name from the connectome if the paper makes this clear (e.g. "LC4", "PPL106", "DA1_lPN"). If the paper studies neurons only by a GAL4 driver line that does not correspond to a single cell type, use the driver line name. If multiple cell types are studied together without being separable, list them all separately. Never invent or infer a connectome name that isn't stated or clearly implied by the paper.

EXAMPLES of what to extract:
- "Motion detection: LC4 neurons respond to ON-motion with increased firing (two-photon imaging). Optogenetic activation biases turning towards the direction of motion (behavioral arena); inactivation reduces turns. Likely useful for course control during navigation."
- "Walking: DNp10 activation increases forward walking speed in tethered flight. Essential for motor control transitions."

Return JSON format:
{
  "cell_types": [
    {
      "cell_type": "LC4",
      "experimental_stimulus": "motion stimuli (ON-motion)",
      "neural_activity": "increased firing rate upon ON-motion",
      "behavioral_consequence": "optogenetic activation biases turning towards motion direction; inactivation reduces turning",
      "ecological_context": "course control during visual navigation and object tracking",
      "techniques_used": ["two-photon imaging", "optogenetics"],
      "summary": "LC4 neurons respond to ON-motion with increased firing, and optogenetic activation biases turning towards the motion direction."
    }
  ]
}

If no experimental data found: return empty object {}
"""


# ---------------------------------------------------------------------------
# JSON parsing (robust multi-strategy, from neuron_interpretation.py)
# ---------------------------------------------------------------------------

def _try_parse(text):
    return json.loads(text)


def _strip_markdown_fences(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


def _fix_common_syntax_errors(text):
    text = _strip_markdown_fences(text)
    text = re.sub(r'\*(\w)', r'\1', text)
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    return text


def _merge_json_blocks(text):
    """Merge multiple top-level JSON objects if the model split its output."""
    merged = {}
    depth, start = 0, None
    candidates = []
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                merged.update(parsed)
        except json.JSONDecodeError:
            pass
    return merged if merged else None


def _recover_truncated_json(text):
    """Extract whatever complete key-value pairs survived truncation."""
    result = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        result[m.group(1)] = m.group(2)
    for m in re.finditer(r'"(\w+)"\s*:\s*(true|false|null|-?\d+\.?\d*)', text):
        v = m.group(2)
        if v == 'true':     v = True
        elif v == 'false':  v = False
        elif v == 'null':   v = None
        else:
            try: v = float(v) if '.' in v else int(v)
            except ValueError: pass
        result[m.group(1)] = v
    return result if result else None


def parse_llm_response(text):
    """
    Multi-strategy JSON parser:
    1. Direct parse
    2. Strip markdown fences
    3. Fix common syntax errors
    4. Merge multiple JSON blocks
    5. Regex match first {...}
    6. Recover from truncation
    """
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


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def load_pdf_text(pdf_path: str, max_chars: int = None) -> str:
    """Load full text from a PDF using PyMuPDF via llama_index."""
    from llama_index.readers.file import PyMuPDFReader
    documents = PyMuPDFReader().load_data(pdf_path)
    if not documents:
        return ""
    full_text = "\n\n".join(doc.text for doc in documents)
    if max_chars is not None and len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[... text truncated ...]"
    return full_text


def load_pdf_rag(pdf_path: str, paper_id: str, mode: str, extraction_query: str,
                 n_results: int = 8, chroma_dir: str = None) -> str:
    """
    Load PDF via RAG: chunk → embed (OpenAI) → query top chunks.
    Requires OPENAI_API_KEY regardless of LLM provider.

    chroma_dir: if provided, use a PersistentClient so embeddings are cached
                across re-runs (one subdir per SLURM chunk). If None, use an
                in-memory EphemeralClient (safe for parallel jobs, no caching).
    """
    import chromadb
    import tiktoken
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.readers.file import PyMuPDFReader

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_EMBED = "text-embedding-3-small"

    documents = PyMuPDFReader().load_data(pdf_path)
    if not documents:
        return ""

    # Chunk
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
    chunks, doc_idxs = [], []
    for doc_idx, doc in enumerate(documents):
        cur = splitter.split_text(doc.text)
        chunks.extend(cur)
        doc_idxs.extend([doc_idx] * len(cur))
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        return ""

    embed_fn = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=MODEL_EMBED)

    # ChromaDB client: persistent (with embedding cache) or ephemeral (safe for parallel jobs)
    if chroma_dir:
        chroma_path = Path(chroma_dir)
        chroma_path.mkdir(parents=True, exist_ok=True)
        db_client = chromadb.PersistentClient(path=str(chroma_path))
    else:
        db_client = chromadb.EphemeralClient()

    col_name = f"paper_{paper_id[:40]}_{mode}"  # chroma collection names have length limits
    try:
        # If persisting: reuse existing collection if it already has embeddings
        existing = db_client.get_collection(name=col_name, embedding_function=embed_fn)
        if chroma_dir and existing.count() > 0:
            # Embeddings already cached — skip re-embedding
            results = existing.query(query_texts=[extraction_query], n_results=n_results)
            context_docs  = results["documents"][0]
            context_dists = results["distances"][0]
            context = ""
            for doc, dist in zip(context_docs, context_dists):
                context += f"{doc}\nCosine distance: {dist:.2f}\n{'-' * 10}\n"
            return context or "No relevant context found."
    except Exception:
        pass  # Collection doesn't exist yet — create it below

    try:
        db_client.delete_collection(col_name)
    except Exception:
        pass
    collection = db_client.create_collection(
        name=col_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch-add respecting token budget
    enc = tiktoken.encoding_for_model(MODEL_EMBED)
    TOKEN_BUDGET = 7500
    i, counter = 0, 0
    while i < len(chunks):
        budget, j = 0, i
        while j < len(chunks):
            t = len(enc.encode(chunks[j]))
            if t >= 8192:
                j += 1
                continue
            if j > i and budget + t > TOKEN_BUDGET:
                break
            budget += t
            j += 1
        batch = chunks[i:j]
        collection.add(
            documents=batch,
            metadatas=[{"doc_idx": doc_idxs[k + i]} for k in range(len(batch))],
            ids=[f"chunk_{counter + k}" for k in range(len(batch))],
        )
        counter += len(batch)
        i = j

    # Query
    results = collection.query(query_texts=[extraction_query], n_results=n_results)
    context_docs = results["documents"][0]
    context_dists = results["distances"][0]
    context = ""
    for doc, dist in zip(context_docs, context_dists):
        context += f"{doc}\nCosine distance: {dist:.2f}\n{'-' * 10}\n"
    return context or "No relevant context found."


# ---------------------------------------------------------------------------
# Per-paper extraction
# ---------------------------------------------------------------------------

def format_extraction_results(result_json: dict, paper_info: dict,
                               mode: str, model_name: str) -> list[dict]:
    """Convert parsed LLM JSON into a list of flat result dicts."""
    if not result_json or "error" in result_json:
        return []

    cell_types = result_json.get("cell_types", [])
    extracted = []
    for ct in cell_types:
        if not isinstance(ct, dict):
            continue

        entry = {
            "paper_id":    paper_info["paper_id"],
            "paper_title": paper_info.get("title", ""),
            "cell_type":   (ct.get("cell_type") or "").strip(),
            "mode":        mode,
            "model":       model_name,
        }

        if mode == "brief":
            entry["known_function"] = (ct.get("known_function") or "").strip()
        elif mode == "experimental":
            # Guard against None values from the model
            entry["experimental_stimulus"]  = (ct.get("experimental_stimulus")  or "").strip()
            entry["neural_activity"]        = (ct.get("neural_activity")        or "").strip()
            entry["behavioral_consequence"] = (ct.get("behavioral_consequence") or "").strip()
            entry["ecological_context"]     = (ct.get("ecological_context")     or "").strip()
            entry["techniques_used"]        = ct.get("techniques_used", []) or []
            entry["summary"]                = (ct.get("summary")                or "").strip()

        extracted.append(entry)
    return extracted


def process_paper(paper_info: dict, args, llama_port=None) -> list[dict]:
    """
    Extract cell types and functions from a single paper PDF.

    Returns a list of result dicts (one per cell type found).
    Raises on unrecoverable errors so the caller can log and continue.
    """
    pdf_path = paper_info.get("path", "")
    if not pdf_path or not Path(str(pdf_path)).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path!r}")

    # Select prompt pair
    if args.mode == "brief":
        system_prompt     = SYSTEM_PROMPT_BRIEF
        extraction_query  = EXTRACTION_QUERY_BRIEF
    else:
        system_prompt     = SYSTEM_PROMPT_EXPERIMENTAL
        extraction_query  = EXTRACTION_QUERY_EXPERIMENTAL

    # Build context (RAG or full text)
    if args.use_rag:
        context = load_pdf_rag(
            pdf_path, paper_info["paper_id"], args.mode, extraction_query,
            chroma_dir=args.chroma_dir,
        )
        user_prompt = f"Query: {extraction_query}\n\nContext from paper:\n{context}"
    else:
        full_text = load_pdf_text(pdf_path, max_chars=args.max_chars)
        if not full_text.strip():
            return []
        user_prompt = f"Paper content:\n\n{full_text}\n\nQuery:\n{extraction_query}"

    # Resolve model config
    config           = _resolve_model_config(args.model)
    temperature      = config.get("temperature", None)
    reasoning_effort = config.get("reasoning_effort", args.reasoning_effort)
    verbosity        = config.get("verbosity", None)

    # LLM call
    response_text, _ = call_llm(
        system_prompt, user_prompt,
        provider         = args.provider,
        model_name       = args.model,
        max_tokens       = args.max_tokens,
        llama_port       = llama_port,
        think            = args.think,
        thinking_budget  = args.thinking_budget,
        reasoning_effort = reasoning_effort,
        verbosity        = verbosity,
        temperature      = temperature,
    )

    result_json = parse_llm_response(response_text)
    return format_extraction_results(result_json, paper_info, args.mode, args.model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dotenv.load_dotenv()

    if args.model is None:
        args.model = PROVIDER_DEFAULTS[args.provider]

    # ---- Load paper list ----
    print(f"Loading papers from: {args.papers_dir}")
    papers_df = pd.read_csv(os.path.expanduser(args.papers_csv))
    # combine folder and file_name 
    papers_df["path"] = [os.path.join(args.papers_dir, fname) for fname in papers_df['file_name']]

    # Keep only rows that have a real PDF path
    has_pdf = papers_df["path"].notna() & (papers_df["path"].astype(str) != "nan")
    papers_df = papers_df[has_pdf].reset_index(drop=True)
    print(f"Papers with PDFs: {len(papers_df)}")

    # ---- Split into chunks ----
    n = len(papers_df)
    chunk_size = (n + args.n_chunks - 1) // args.n_chunks
    start = args.chunk_id * chunk_size
    end   = min(start + chunk_size, n)
    chunk_df = papers_df.iloc[start:end].copy()
    print(f"Chunk {args.chunk_id}/{args.n_chunks}: papers {start}–{end} ({len(chunk_df)} papers)")

    # ---- Output setup ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"results_chunk_{args.chunk_id}.jsonl"

    # ---- Resume: skip already-done papers ----
    done_ids: set[str] = set()
    if args.resume and out_file.exists():
        with open(out_file) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # A paper is 'done' if it produced at least one non-error result,
                    # or if we logged it as processed (even with 0 cell types found).
                    pid = record.get("paper_id")
                    if pid:
                        done_ids.add(pid)
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} papers already done, skipping.")

    remaining = chunk_df[~chunk_df["paper_id"].isin(done_ids)]
    print(f"Papers to process: {len(remaining)}")

    if remaining.empty:
        print("All papers in this chunk already processed. Exiting.")
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

        # Warmup
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

    # ---- Process papers ----
    try:
        with open(out_file, "a") as f:
            for _, row in tqdm(remaining.iterrows(), total=len(remaining),
                               desc=f"Chunk {args.chunk_id}"):
                paper_info = row.to_dict()
                paper_id   = paper_info["paper_id"]

                # Check server health before each paper; restart if needed
                if args.provider == "llama":
                    if not is_llama_server_healthy(llama_port):
                        print(f"  [llama-server] Unhealthy before {paper_id[:12]}..., restarting...")
                        llama_proc = restart_llama_server(
                            args.llama_bin, args.gguf, llama_port, llama_proc
                        )

                try:
                    results = process_paper(paper_info, args, llama_port=llama_port)

                    if results:
                        for entry in results:
                            f.write(json.dumps(entry) + "\n")
                    else:
                        # Log an empty result so resume knows this paper was attempted
                        f.write(json.dumps({
                            "paper_id":    paper_id,
                            "paper_title": paper_info.get("title", ""),
                            "cell_types_found": 0,
                            "mode":  args.mode,
                            "model": args.model,
                        }) + "\n")

                    f.flush()

                    # Rate-limit delay (only relevant for cloud APIs)
                    if args.provider != "llama":
                        time.sleep(args.delay)

                except Exception as e:
                    print(f"  ERROR processing {paper_info.get('title', paper_id)[:60]}: {e}")
                    traceback.print_exc()
                    error_record = {
                        "paper_id":    paper_id,
                        "paper_title": paper_info.get("title", ""),
                        "error":       str(e),
                        "chunk_id":    args.chunk_id,
                    }
                    f.write(json.dumps(error_record) + "\n")
                    f.flush()
                    continue

    finally:
        if llama_proc is not None:
            print("[llama-server] Shutting down...")
            llama_proc.terminate()
            llama_proc.wait()

    print(f"\nDone. Results saved to {out_file}")


if __name__ == "__main__":
    main()