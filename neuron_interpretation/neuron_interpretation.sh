#!/bin/bash
#SBATCH --job-name=neuro_interp
#SBATCH --array=0-7                  # 8 jobs (chunks 0,1,2,3,4,5,6,7) — one per A100
#SBATCH --partition=ml               # GPU partition
#SBATCH --gres=gpu:1                 # 1 GPU per job
#SBATCH --output=logs/interp_fun_%A_%a.out
#SBATCH --error=logs/interp_fun_%A_%a.err
#SBATCH --mail-type=END,FAIL                # Email if job fails
#SBATCH --mail-user=yy432@cam.ac.uk

# ---- Environment ----
module load cuda/12.4
module load compilers/gcc/12.2.0
export LD_LIBRARY_PATH=/public/gcc/12_2_0/lib64:$LD_LIBRARY_PATH
export HF_HOME=/cephfs2/yyin/huggingface
export HUGGINGFACE_HUB_CACHE=/cephfs2/yyin/huggingface/hub

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

# ---- Config ----
N_CHUNKS=8                           # Must match --array upper bound + 1
CHUNK_ID=$SLURM_ARRAY_TASK_ID

INTERPRET_SCRIPT="$HOME/llm_circuit_analysis/neuron_interpretation/neuron_interpretation.py"

# Update these paths for your HPC setup:
BASE_PATH="$HOME/interpret_connectome/"
KNOWN_TYPES_CSV="$HOME/known_types_snapshots/known_types_140326.csv"
GGUF="/cephfs2/yyin/huggingface/hub/qwen35_gguf/Qwen_Qwen3.5-35B-A3B-Q6_K_L.gguf"
LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"
OUTPUT_DIR="$HOME/llm_circuit_analysis/neuron_interpretation/results"

mkdir -p "$HOME/llm_circuit_analysis/logs"
mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Chunk:     $CHUNK_ID / $N_CHUNKS"
echo "=============================="

python "$INTERPRET_SCRIPT" \
    --chunk-id "$CHUNK_ID" \
    --n-chunks "$N_CHUNKS" \
    --provider llama \
    --gguf "$GGUF" \
    --llama-bin "$LLAMA_BIN" \
    --base-path "$BASE_PATH" \
    --known-types-csv "$KNOWN_TYPES_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --n-steps 4 \
    --top-n 15 \
    --max-tokens 20000 \
    --no-resume 
    # --no-add-function # whether to map cell types to known functions

echo "Chunk $CHUNK_ID complete."

# ---- To use OpenAI instead, replace the python call above with: ----
# python "$INTERPRET_SCRIPT" \
#     --chunk-id "$CHUNK_ID" \
#     --n-chunks "$N_CHUNKS" \
#     --provider openai \
#     --model gpt-4o \
#     --base-path "$BASE_PATH" \
#     --known-types-csv "$KNOWN_TYPES_CSV" \
#     --output-dir "$OUTPUT_DIR"