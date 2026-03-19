#!/bin/bash
#SBATCH --job-name=func_extract
#SBATCH --array=0-7                  # 8 jobs (chunks 0..7) — one per A100
#SBATCH --partition=ml               # GPU partition
#SBATCH --gres=gpu:1                 # 1 GPU per job
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yy432@cam.ac.uk

# ---- Environment ----
module load cuda/12.4
module load compilers/gcc/12.2.0
export LD_LIBRARY_PATH=/public/gcc/12_2_0/lib64:$LD_LIBRARY_PATH
export HF_HOME=/cephfs2/yyin/huggingface
export HUGGINGFACE_HUB_CACHE=/cephfs2/yyin/huggingface/hub

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

# ---- Config ----
N_CHUNKS=8                           # Must match --array upper bound + 1
CHUNK_ID=$SLURM_ARRAY_TASK_ID

EXTRACT_SCRIPT="$HOME/llm_circuit_analysis/function_extraction_from_paper.py"

PAPERS_CSV="$HOME/llm_circuit_analysis/extraction_results/papers_with_names.csv"
PAPERS_DIR="/cephfs2/yyin/llm_circuit_analysis/pdfs" # will be used in combination with file_name in PAPERS_CSV to locate each PDF
GGUF="/cephfs2/yyin/huggingface/hub/qwen35_gguf/Qwen_Qwen3.5-35B-A3B-Q6_K_L.gguf"
LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"
OUTPUT_DIR="$HOME/llm_circuit_analysis/extraction_results/results"

mkdir -p "$HOME/llm_circuit_analysis/logs"
mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Chunk:     $CHUNK_ID / $N_CHUNKS"
echo "=============================="

python "$EXTRACT_SCRIPT" \
    --chunk-id   "$CHUNK_ID" \
    --n-chunks   "$N_CHUNKS" \
    --provider   llama \
    --gguf       "$GGUF" \
    --llama-bin  "$LLAMA_BIN" \
    --papers-csv "$PAPERS_CSV" \
    --papers-dir "$PAPERS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode       experimental \
    --max-tokens 8000
    # --mode brief             # for 6-word summaries instead
    # --no-resume              # force reprocess all papers
    # --max-chars 80000        # truncate very long PDFs (chars, not tokens)
    # --think                  # enable Qwen3 reasoning (default: off via /no_think)

echo "Chunk $CHUNK_ID complete."


# ---- To use OpenAI instead, replace the python call above with: ----
# python "$EXTRACT_SCRIPT" \
#     --chunk-id   "$CHUNK_ID" \
#     --n-chunks   "$N_CHUNKS" \
#     --provider   openai \
#     --model      gpt-4o \
#     --papers-csv "$PAPERS_CSV" \
#     --output-dir "$OUTPUT_DIR" \
#     --mode       experimental \
#     --delay      0.5          # rate-limit between API calls


# ---- To use RAG (ChromaDB) instead of full text: ----
# Requires OPENAI_API_KEY even with non-OpenAI LLM providers (for embeddings).
# If you want embeddings to persist across runs, point --chroma-dir at a
# shared path on cephfs so all 8 array jobs share one vector store:
#
# CHROMA_DIR="/cephfs2/yyin/chroma_paper_db"
# mkdir -p "$CHROMA_DIR"
#
# python "$EXTRACT_SCRIPT" \
#     --chunk-id    "$CHUNK_ID" \
#     --n-chunks    "$N_CHUNKS" \
#     --provider    llama \
#     --gguf        "$GGUF" \
#     --llama-bin   "$LLAMA_BIN" \
#     --papers-csv  "$PAPERS_CSV" \
#     --output-dir  "$OUTPUT_DIR" \
#     --mode        experimental \
#     --use-rag \
#     --chroma-dir  "$CHROMA_DIR"
#
# Note: concurrent writes from 8 jobs to the same ChromaDB are not safe.
# Either (a) pre-build the DB in a separate serial job first, then run
# the array jobs read-only, or (b) give each chunk its own --chroma-dir.