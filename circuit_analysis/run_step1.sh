#!/bin/bash
#SBATCH --job-name=circuit_s1
#SBATCH --array=0-7                  # 8 jobs — adjust to match N_CHUNKS
#SBATCH --partition=ml
#SBATCH --gres=gpu:1
#SBATCH --output=logs/circuit_s1_%A_%a.out
#SBATCH --error=logs/circuit_s1_%A_%a.err
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
N_CHUNKS=8
CHUNK_ID=$SLURM_ARRAY_TASK_ID

SCRIPT_DIR="$HOME/llm_circuit_analysis/circuit_analysis"
BASE_PATH="$HOME/interpret_connectome/"
KNOWN_TYPES_CSV="$HOME/known_types_snapshots/known_types_140326.csv"
HYPOTHESES_CSV="$HOME/llm_circuit_analysis/neuron_interpretation/hypotheses.csv"
TYPES_FILE="$SCRIPT_DIR/circuit_types.json"
GGUF="/cephfs2/yyin/huggingface/hub/qwen35_gguf/Qwen_Qwen3.5-35B-A3B-Q6_K_L.gguf"
LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"
OUTPUT_DIR="$SCRIPT_DIR/results_step1"

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "STEP 1: Pairwise pathway analysis"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Chunk:     $CHUNK_ID / $N_CHUNKS"
echo "=============================="

python "$SCRIPT_DIR/circuit_step1_pairwise.py" \
    --chunk-id "$CHUNK_ID" \
    --n-chunks "$N_CHUNKS" \
    --provider llama \
    --gguf "$GGUF" \
    --llama-bin "$LLAMA_BIN" \
    --base-path "$BASE_PATH" \
    --known-types-csv "$KNOWN_TYPES_CSV" \
    --hypotheses-csv "$HYPOTHESES_CSV" \
    --types-file "$TYPES_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --side right \
    --n-steps 3 \
    --threshold 0.01 \
    --max-tokens 4096 \
    --resume
    # --actor-critic --max-critic-rounds 2  # uncomment to enable iterative critic-revision loop

echo "Step 1 chunk $CHUNK_ID complete."

# ---- OpenAI alternative ----
# python "$SCRIPT_DIR/circuit_step1_pairwise.py" \
#     --chunk-id "$CHUNK_ID" \
#     --n-chunks "$N_CHUNKS" \
#     --provider openai \
#     --model gpt-4o \
#     --base-path "$BASE_PATH" \
#     --known-types-csv "$KNOWN_TYPES_CSV" \
#     --hypotheses-csv "$HYPOTHESES_CSV" \
#     --types-file "$TYPES_FILE" \
#     --output-dir "$OUTPUT_DIR" \
#     --side right \
#     --n-steps 3 \
#     --resume

# ---- Anthropic alternative ----
# python "$SCRIPT_DIR/circuit_step1_pairwise.py" \
#     --chunk-id "$CHUNK_ID" \
#     --n-chunks "$N_CHUNKS" \
#     --provider anthropic \
#     --model claude-sonnet-4-6 \
#     --base-path "$BASE_PATH" \
#     --known-types-csv "$KNOWN_TYPES_CSV" \
#     --hypotheses-csv "$HYPOTHESES_CSV" \
#     --types-file "$TYPES_FILE" \
#     --output-dir "$OUTPUT_DIR" \
#     --side right \
#     --n-steps 3 \
#     --resume