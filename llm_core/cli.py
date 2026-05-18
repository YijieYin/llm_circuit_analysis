"""Argparse helper for LLM-related CLI args shared across pipeline scripts."""

import argparse
import os


def add_llm_args(parser):
    """Add LLM-related CLI args. Idempotent and safe to call from any script.

    Adds:
        --provider {llama,openai,anthropic}
        --model NAME
        --gguf PATH                 (llama provider)
        --llama-bin PATH            (llama provider)
        --llama-port PORT           (llama provider)
        --max-tokens N
        --thinking-budget N         (anthropic)
        --reasoning-effort {low,medium,high}  (openai gpt-5*, o1*)
        --verbosity {low,medium,high}         (openai gpt-5*)
        --think / --no-think        (llama: Qwen3 reasoning toggle)
        --resume / --no-resume

    Connectome-specific args (--base-path, --known-types-csv, --side, --n-steps,
    --threshold, --hypotheses-csv, --types-file, etc.) live in circuit_utils.
    """
    parser.add_argument("--provider", type=str, default="llama",
                        choices=["llama", "openai", "anthropic"])
    parser.add_argument("--model", type=str, default=None,
                        help="Model name. If unset, uses PROVIDER_DEFAULTS[provider].")

    # llama provider
    parser.add_argument("--gguf", type=str,
                        default="/cephfs2/yyin/huggingface/hub/qwen35_gguf/"
                                "Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf",
                        help="Path to GGUF file (llama provider only)")
    parser.add_argument("--llama-bin", type=str,
                        default=os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
                        help="Path to llama-server binary")
    parser.add_argument("--llama-port", type=int, default=None,
                        help="Override port; default picks a free one")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable Qwen3 reasoning mode (llama provider). "
                             "Off by default: prepends /no_think and strips <think> blocks.")

    # Generation
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Anthropic extended thinking budget (tokens)")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="OpenAI reasoning effort (gpt-5*, o1*)")
    parser.add_argument("--verbosity", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="OpenAI verbosity (Responses API models)")

    # Job control
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser