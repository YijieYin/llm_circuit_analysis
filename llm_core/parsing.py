"""Robust JSON parsing and key validation for LLM outputs.

Models routinely emit JSON wrapped in markdown fences, with trailing commas,
with extra prose, with double-emitted blocks, or truncated mid-string.
parse_llm_response() runs through six fallback strategies before giving up.

validate_response() then checks the dict has the required keys and only
those keys plus declared metadata.
"""

import json
import re


# ---------------------------------------------------------------------------
# Parse strategies
# ---------------------------------------------------------------------------

def _try_parse(text):
    return json.loads(text)


def _strip_markdown_fences(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


def _merge_json_blocks(text):
    """If the model emitted multiple {...} objects, merge their keys."""
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


def _fix_common_syntax_errors(text):
    """Strip fences, kill leading asterisks on keys, drop trailing commas."""
    text = _strip_markdown_fences(text)
    text = re.sub(r'\*(\w)', r'\1', text)
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    return text


def _recover_truncated_json(text):
    """Last-ditch: pull out any complete key-value pairs that survived."""
    result = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        result[m.group(1)] = m.group(2)
    for m in re.finditer(r'"(\w+)"\s*:\s*(true|false|null|-?\d+\.?\d*)', text):
        v = m.group(2)
        if v == 'true':    v = True
        elif v == 'false': v = False
        elif v == 'null':  v = None
        else:
            try: v = float(v) if '.' in v else int(v)
            except ValueError: pass
        result[m.group(1)] = v
    return result if result else None


def parse_llm_response(text):
    """Multi-strategy JSON parser. Always returns a dict.

    On success: the parsed object.
    On partial recovery: parsed keys plus 'error': 'truncated_response_partial_recovery'.
    On total failure: {'error': 'JSON parse failed', 'raw_response': text}.
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
# Validation
# ---------------------------------------------------------------------------

# Always-allowed metadata keys that scripts attach post-hoc to the parsed dict.
# Scripts add their own task-specific meta via extra_meta_keys.
BASE_META_KEYS = frozenset({
    "model", "provider", "chunk_id", "validation_error",
})


def validate_response(parsed, required_keys, extra_meta_keys=None):
    """Check parsed dict has exactly required_keys (plus allowed metadata).

    Returns (ok, reason).

    required_keys: iterable of strings — keys that MUST be present.
    extra_meta_keys: iterable of strings — additional keys allowed but not required
        (e.g. for circuit step 1: {"source", "target", "net_excit", "net_inhib",
        "critic_flag", "critic_rounds", "critic_approved", "critic_history"};
        for neuron interpretation: {"cell_type", "add_function"}).
    """
    required = set(required_keys)

    # If parsing failed AND we got none of the required keys back, fail fast.
    if "error" in parsed and not required.intersection(parsed.keys()):
        return False, f"parse error: {parsed.get('error')}"

    allowed_meta = BASE_META_KEYS | set(extra_meta_keys or [])
    missing = required - parsed.keys()
    extra = parsed.keys() - required - allowed_meta

    if missing:
        return False, f"missing keys: {missing}"
    if extra:
        return False, f"extra keys present: {extra}"
    return True, ""