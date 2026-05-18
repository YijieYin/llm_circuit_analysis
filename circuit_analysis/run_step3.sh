#!/bin/bash
#SBATCH --job-name=circuit_s3
#SBATCH --partition=ml
#SBATCH --gres=gpu:1
#SBATCH --output=circuit_analysis/logs/circuit_s3_%j.out
#SBATCH --error=circuit_analysis/logs/circuit_s3_%j.err
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
SCRIPT_DIR="$HOME/llm_circuit_analysis/circuit_analysis"
HYPOTHESES_CSV="$HOME/llm_circuit_analysis/neuron_interpretation/hypotheses.csv"
TYPES_FILE="$SCRIPT_DIR/circuit_types.json"
GGUF="/cephfs2/yyin/huggingface/hub/qwen35_gguf/Qwen_Qwen3.5-35B-A3B-Q6_K_L.gguf"
LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"
STEP2_DIR="$SCRIPT_DIR/results_step2"
STEP1_DIR="$SCRIPT_DIR/results_step1"
OUTPUT_DIR="$SCRIPT_DIR/results_step3"

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "STEP 3: Circuit synthesis"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "=============================="

python "$SCRIPT_DIR/circuit_step3_synthesis.py" \
    --provider llama \
    --gguf "$GGUF" \
    --llama-bin "$LLAMA_BIN" \
    --hypotheses-csv "$HYPOTHESES_CSV" \
    --types-file "$TYPES_FILE" \
    --step2-dir "$STEP2_DIR" \
    --step1-dir "$STEP1_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --side right \
    --max-tokens 20000

echo "Step 3 complete. Results in $OUTPUT_DIR/circuit_synthesis.json"