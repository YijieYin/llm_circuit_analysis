"""llm_core — provider-agnostic LLM utilities for the pipeline.

Submodules:
  providers     — call_llama / call_openai / call_anthropic / call_llm + defaults
  llama_server  — local llama-server (llama.cpp) lifecycle
  parsing       — robust JSON parsing + key validation
  retry         — call_llm_with_retry (validation + retry loop)
  cli           — add_llm_args (shared argparse options)

Typical usage in a pipeline script:

    from llm_core import (
        PROVIDER_DEFAULTS, add_llm_args,
        call_llm_with_retry, setup_llama_server, restart_llama_server,
        is_llama_server_healthy,
    )
"""
from .providers import (
    PROVIDER_DEFAULTS,
    MODEL_DEFAULTS,
    RESPONSES_API_MODELS,
    resolve_model_config,
    call_llama,
    call_openai,
    call_anthropic,
    call_llm,
)
from .llama_server import (
    find_free_port,
    start_llama_server,
    wait_for_llama_server,
    is_llama_server_healthy,
    restart_llama_server,
    setup_llama_server,
    DEFAULT_CTX_SIZE,
    DEFAULT_N_PARALLEL,
)
from .parsing import (
    parse_llm_response,
    validate_response,
    BASE_META_KEYS,
)
from .retry import (
    call_llm_with_retry,
    MAX_RETRIES,
)
from .cli import (
    add_llm_args,
)
from .dump import (
    write_prompt_file,
    safe_filename,
)