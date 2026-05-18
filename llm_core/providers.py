"""Provider-agnostic LLM call wrappers.

Supported providers:
  - llama:     local llama-server (llama.cpp) via OpenAI-compatible API
  - openai:    OpenAI API (auto-routes gpt-5* to the Responses API)
  - anthropic: Anthropic API
"""

import re
import time


# ---------------------------------------------------------------------------
# Provider & model defaults
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS = {
    "llama":     "qwen3.5:35b",  # just a label; actual model is the GGUF on disk
    "openai":    "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}

# OpenAI models that should use the Responses API (supports verbosity, reasoning)
RESPONSES_API_MODELS = {"gpt-5", "gpt-5.4"}

# Model-specific default kwargs applied by resolve_model_config()
MODEL_DEFAULTS = {
    "gpt-4o":      {"temperature": 1.0},
    "gpt-4-turbo": {"temperature": 1.0},
    "gpt-4":       {"temperature": 1.0},
    "gpt-5":       {"temperature": 1.0},
    "gpt-5.4":     {"temperature": 1.0, "reasoning_effort": "medium", "verbosity": "low"},
}


def resolve_model_config(model_name, extra_config=None):
    """Return a kwargs dict for the model: defaults then any overrides."""
    config = {"model": model_name}
    if model_name in MODEL_DEFAULTS:
        config.update(MODEL_DEFAULTS[model_name])
    if extra_config:
        config.update(extra_config)
    return config


def _use_responses_api(model_name):
    return model_name in RESPONSES_API_MODELS


# ---------------------------------------------------------------------------
# Per-provider call wrappers
# ---------------------------------------------------------------------------

def call_llama(system_prompt, user_prompt, port, max_tokens=4096,
               think=False, retries=3):
    """Call local llama-server via its OpenAI-compatible /v1 endpoint.

    Qwen3 'think' mode:
      think=False (default): prepend /no_think and strip any <think>...</think>
                             blocks that slip through.
      think=True:            prepend /think; <think> blocks are preserved.
                             Both are returned so callers can inspect reasoning.
    """
    from openai import OpenAI
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

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


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def call_llm(system_prompt, user_prompt, provider, model_name, max_tokens,
             llama_port=None, think=False, thinking_budget=None,
             reasoning_effort=None, verbosity=None, temperature=None):
    """Dispatch to the right provider. Returns (text, raw_response)."""
    if provider == "llama":
        return call_llama(system_prompt, user_prompt, llama_port, max_tokens,
                          think=think)
    elif provider == "openai":
        return call_openai(system_prompt, user_prompt, model_name, max_tokens,
                           reasoning_effort=reasoning_effort, verbosity=verbosity,
                           temperature=temperature)
    elif provider == "anthropic":
        return call_anthropic(system_prompt, user_prompt, model_name, max_tokens,
                              thinking_budget=thinking_budget)
    else:
        raise ValueError(f"Unknown provider: {provider}")