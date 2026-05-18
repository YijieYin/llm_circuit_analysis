import pandas as pd
import os

# meta data from connectomes 
manc_meta = pd.read_csv('../interpret_connectome/data/MANC/manc_meta.csv', index_col=0)
mcns_meta = pd.read_csv('../interpret_connectome/data/maleCNS/mcns_all_neuron_meta.csv', index_col=0)
fafb_meta = pd.read_csv('../interpret_connectome/data/fafb_all_neuron/fafb_all_neuron_meta.csv')

extracted = pd.read_csv("function_extraction_from_papers/extraction_results/extracted.csv", encoding="utf-8")
hypotheses = pd.read_csv("neuron_interpretation/hypotheses.csv", encoding="utf-8")

hypotheses['add_function'] = hypotheses['add_function'].replace({True: 'inferred_computation_func', False: 'inferred_computation_nofunc'})
check = (hypotheses[['cell_type', 'inferred_computation', 'add_function']].pivot(index='cell_type', columns='add_function', values='inferred_computation').reset_index()
.merge(extracted[['cell_type', 'summary']], on='cell_type')
)

## Extract the function from the poorly parsed 'inferred_computation_func' column 
import ast

def extract_computations(val):
    # Return as-is if not a string
    if not isinstance(val, str):
        return val
    
    try:
        parsed = ast.literal_eval(val)
        # If it's a list of dicts with 'computation' keys, extract them
        if isinstance(parsed, list) and all(isinstance(d, dict) and 'computation' in d for d in parsed):
            return [d['computation'] for d in parsed]
        else:
            return val  # Keep intact if structure doesn't match
    except (ValueError, SyntaxError):
        return val  # Keep intact if it can't be parsed

check['inferred_computation_func'] = check['inferred_computation_func'].apply(extract_computations)

# ── Embeddings via local llama-server ──────────────────────────────────────
import numpy as np
import subprocess, time, json, urllib.request
from openai import OpenAI

LLAMA_BIN  = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
GGUF       = "/cephfs2/yyin/huggingface/hub/qwen35_gguf/Qwen_Qwen3.5-35B-A3B-Q6_K_L.gguf"
EMBED_PORT = 8081   # different from the inference server's 8080

def start_embed_server(llama_bin, gguf, port):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/public/gcc/12_2_0/lib64:" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.Popen(
        [llama_bin, "-m", gguf, "--port", str(port), "-ngl", "99",
         "--embeddings", "--ctx-size", "8192", "--no-mmap"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    for _ in range(100):          # wait up to ~300s
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    time.sleep(30); break   # extra wait for weights to load
        except Exception:
            pass
        time.sleep(3)
    return proc

embed_proc = start_embed_server(LLAMA_BIN, GGUF, EMBED_PORT)
client = OpenAI(base_url=f"http://localhost:{EMBED_PORT}/v1", api_key="not-needed")

def get_embeddings(texts, batch_size=32):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model="", input=batch)
        all_vecs.extend(item.embedding for item in sorted(resp.data, key=lambda x: x.index))
    return np.array(all_vecs, dtype=np.float32)

cols = ["inferred_computation_nofunc", "inferred_computation_func", "summary"]
embeddings = {col: get_embeddings(check[col].fillna("").tolist()) for col in cols}

embed_proc.terminate()   # shut down when done

# ── Pairwise cosine similarities (per row) ─────────────────────────────────
from sklearn.preprocessing import normalize

# L2-normalise so dot product == cosine similarity
normed = {col: normalize(embeddings[col]) for col in cols}

check["sim_summary_vs_nofunc"] = (normed["summary"] * normed["inferred_computation_nofunc"]).sum(axis=1)
check["sim_summary_vs_func"]   = (normed["summary"] * normed["inferred_computation_func"]).sum(axis=1)

# ── Save ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "embedding_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV: check df with similarity columns appended
check.to_csv(f"{OUTPUT_DIR}/check_with_similarities.csv", index=False)

# NPZ: raw embedding arrays + similarity scores
np.savez(
    f"{OUTPUT_DIR}/embeddings.npz",
    **{col: embeddings[col] for col in cols},
    sim_summary_vs_nofunc=check["sim_summary_vs_nofunc"].values,
    sim_summary_vs_func=check["sim_summary_vs_func"].values,
)

print(check[["cell_type", "sim_summary_vs_nofunc", "sim_summary_vs_func"]].describe())