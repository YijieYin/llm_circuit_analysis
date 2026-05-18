"""LLM call with retry + JSON validation.

Replaces the per-script retry loops in neuron_interpretation, function_extraction,
and circuit_step1/2/3 with one shared implementation.
"""

from .providers import call_llm, resolve_model_config
from .parsing import parse_llm_response, validate_response


MAX_RETRIES = 3


def call_llm_with_retry(system_prompt, user_prompt, required_keys, args,
                        llama_port=None, extra_meta_keys=None,
                        max_retries=MAX_RETRIES):
    """Call LLM, validate response shape, retry on failure.

    Reads from args: provider, model, max_tokens. Optionally reads
    think, thinking_budget, reasoning_effort, verbosity (uses getattr with
    None/False defaults).

    Returns the parsed dict. If all retries failed validation, the last
    parsed dict is returned with a 'validation_error' field attached.

    extra_meta_keys: passed through to validate_response — per-step keys
    that callers may attach to results without triggering 'extra keys' errors.
    """
    config = resolve_model_config(args.model)
    reasoning_effort = config.get("reasoning_effort",
                                   getattr(args, "reasoning_effort", None))
    verbosity = config.get("verbosity", getattr(args, "verbosity", None))
    temperature = config.get("temperature", None)
    think = getattr(args, "think", False)

    last_result = None
    last_reason = ""
    for attempt in range(max_retries):
        response_text, _ = call_llm(
            system_prompt, user_prompt,
            provider=args.provider,
            model_name=args.model,
            max_tokens=args.max_tokens,
            llama_port=llama_port,
            think=think,
            thinking_budget=getattr(args, "thinking_budget", None),
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            temperature=temperature,
        )
        result = parse_llm_response(response_text)
        ok, reason = validate_response(result, required_keys, extra_meta_keys)
        if ok:
            return result
        print(f"  [retry {attempt+1}/{max_retries}] {reason}")
        last_result = result
        last_reason = reason

    result = last_result if last_result is not None else {}
    result["validation_error"] = last_reason
    return result