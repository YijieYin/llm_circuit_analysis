"""llama-server (llama.cpp) lifecycle management.

start/wait/health/restart all share the same configurable ctx_size and
n_parallel, so callers from any pipeline step can use the same module
with their own settings.
"""

import json
import os
import socket
import subprocess
import time

from .providers import call_llama


# Sensible defaults. Override per-script as needed.
# - circuit/neuron interpretation typically uses 24576 ctx, 1 parallel slot
# - function extraction uses 65536 ctx (long PDFs), 1 parallel slot
# - paper discovery uses 8192 ctx, 16 parallel slots (short title+abstract evals)
DEFAULT_CTX_SIZE = 24576
DEFAULT_N_PARALLEL = 1


def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_llama_server(llama_bin, gguf_path, port,
                       ctx_size=DEFAULT_CTX_SIZE,
                       n_parallel=DEFAULT_N_PARALLEL):
    """Start llama-server as a background subprocess. Returns Popen handle."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/public/gcc/12_2_0/lib64:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        llama_bin,
        "-m", gguf_path,
        "--port", str(port),
        "-ngl", "99",
        "--ctx-size", str(ctx_size),
        "--no-mmap",
        "-np", str(n_parallel),
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
    """Quick health check — does /health return ok?"""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def wait_for_llama_server(port, timeout=300):
    """Poll /health until ok, then sleep 30s for model load."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    print(f"[llama-server] Health check passed on port {port}, "
                          f"waiting for model load...")
                    time.sleep(30)
                    print(f"[llama-server] Ready on port {port}")
                    return True
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"llama-server did not start within {timeout}s")


def restart_llama_server(llama_bin, gguf_path, port, old_proc=None,
                         ctx_size=DEFAULT_CTX_SIZE,
                         n_parallel=DEFAULT_N_PARALLEL):
    """Tear down old server and start a fresh one with a warmup ping."""
    if old_proc is not None:
        print("[llama-server] Restarting...")
        old_proc.terminate()
        try:
            old_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            old_proc.kill()
    proc = start_llama_server(llama_bin, gguf_path, port,
                              ctx_size=ctx_size, n_parallel=n_parallel)
    wait_for_llama_server(port)
    print("[llama-server] Warmup after restart...")
    try:
        call_llama("You are helpful.", "Say OK.", port, max_tokens=512)
    except Exception:
        pass
    return proc


def setup_llama_server(args, ctx_size=None, n_parallel=None):
    """Start + warm up llama-server from CLI args. Returns (proc, port).

    ctx_size/n_parallel: explicit overrides. If None, will look for
    args.ctx_size / args.n_parallel; if those don't exist either, falls back
    to module defaults. Lets each pipeline step pass its own preferred sizing.
    """
    if ctx_size is None:
        ctx_size = getattr(args, "ctx_size", DEFAULT_CTX_SIZE)
    if n_parallel is None:
        n_parallel = getattr(args, "n_parallel", DEFAULT_N_PARALLEL)

    port = args.llama_port or find_free_port()
    proc = start_llama_server(
        args.llama_bin, args.gguf, port,
        ctx_size=ctx_size, n_parallel=n_parallel,
    )
    try:
        wait_for_llama_server(port)
    except TimeoutError:
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