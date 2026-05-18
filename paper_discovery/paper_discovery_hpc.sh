#!/bin/bash
#SBATCH --job-name=paper_discovery
#SBATCH --partition=ml
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/discovery_%j.out
#SBATCH --error=logs/discovery_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yy432@cam.ac.uk

# ---- Environment (mirrors function_extraction_from_paper.sh) ----
module load cuda/12.4
module load compilers/gcc/12.2.0
export LD_LIBRARY_PATH=/public/gcc/12_2_0/lib64:$LD_LIBRARY_PATH
export HF_HOME=/cephfs2/yyin/huggingface
export HUGGINGFACE_HUB_CACHE=/cephfs2/yyin/huggingface/hub

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

# requests_cache is needed; install once if missing
pip install -q requests-cache 2>/dev/null || true

# ---- Config ----
SCRIPT="$HOME/llm_circuit_analysis/paper_discovery/paper_discovery_hpc.py"
SEEDS="$HOME/llm_circuit_analysis/paper_discovery/seeds.txt"
OUTPUT_DIR="$HOME/llm_circuit_analysis/paper_discovery/output"

# 4B model is enough for relevance filtering. Adjust path to your actual GGUF.
GGUF="/cephfs2/yyin/huggingface/hub/qwen35_gguf/Qwen_Qwen3.5-4B-Q6_K.gguf"
LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"

mkdir -p "$HOME/llm_circuit_analysis/paper_discovery/logs"
mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"
echo "=============================="

python "$SCRIPT" \
    --seeds              "$SEEDS" \
    --output-dir         "$OUTPUT_DIR" \
    --gguf               "$GGUF" \
    --llama-bin          "$LLAMA_BIN" \
    --max-depth          2 \
    --max-papers-per-seed 2000 \
    --n-parallel-slots   16 \
    --n-llm-workers      16 \
    --n-fetch-workers    8 \
    --save-every         10 \
    --required-keywords  "Drosophila|fruit fly|melanogaster|D. melanogaster"

echo "Done."

# ---- Notes ----
# If S2 rate-limits you (HTTP 429), reduce parallelism: --s2-rate 0.5
# Get a free S2 API key (https://www.semanticscholar.org/product/api#api-key-form),
#   export SEMANTIC_SCHOLAR_API_KEY=... before sbatch, and the script will use it.
# If your compute node lacks internet, run discovery from a login-node tmux
#   session instead, or split into a non-GPU fetcher job + a GPU filter job.