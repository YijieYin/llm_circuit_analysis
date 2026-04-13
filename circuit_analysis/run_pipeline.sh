#!/bin/bash
# run_pipeline.sh — Submit the full 3-step circuit analysis pipeline.
#
# Step 1 runs first (array job), step 2 starts after ALL step 1 jobs finish,
# step 3 starts after ALL step 2 jobs finish.
#
# Usage:
#   bash run_pipeline.sh
#   bash run_pipeline.sh --dry-run   # print commands without submitting

set -euo pipefail

SCRIPT_DIR="$HOME/llm_circuit_analysis/circuit_analysis"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Commands will be printed but not executed."
fi

# Ensure logs directory exists
mkdir -p "$SCRIPT_DIR/logs"

# Check that circuit_types.json exists
if [[ ! -f "$SCRIPT_DIR/circuit_types.json" ]]; then
    echo "ERROR: $SCRIPT_DIR/circuit_types.json not found."
    echo "Run select_types.py first to define your sources and targets."
    exit 1
fi

echo "=== Circuit Analysis Pipeline ==="
echo "Types file: $SCRIPT_DIR/circuit_types.json"
cat "$SCRIPT_DIR/circuit_types.json"
echo ""

# ---- Step 1: Pairwise pathway analysis (array job) ----
CMD1="sbatch $SCRIPT_DIR/run_step1.sh"
if $DRY_RUN; then
    echo "[Step 1] $CMD1"
    JOB1="123456"
else
    JOB1=$(sbatch "$SCRIPT_DIR/run_step1.sh" | awk '{print $4}')
    echo "[Step 1] Submitted array job $JOB1"
fi

# ---- Step 2: Per-target integration (depends on step 1) ----
CMD2="sbatch --dependency=afterok:$JOB1 $SCRIPT_DIR/run_step2.sh"
if $DRY_RUN; then
    echo "[Step 2] $CMD2"
    JOB2="123457"
else
    JOB2=$(sbatch --dependency=afterok:"$JOB1" "$SCRIPT_DIR/run_step2.sh" | awk '{print $4}')
    echo "[Step 2] Submitted array job $JOB2 (depends on $JOB1)"
fi

# ---- Step 3: Circuit synthesis (depends on step 2) ----
CMD3="sbatch --dependency=afterok:$JOB2 $SCRIPT_DIR/run_step3.sh"
if $DRY_RUN; then
    echo "[Step 3] $CMD3"
else
    JOB3=$(sbatch --dependency=afterok:"$JOB2" "$SCRIPT_DIR/run_step3.sh" | awk '{print $4}')
    echo "[Step 3] Submitted job $JOB3 (depends on $JOB2)"
fi

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $SCRIPT_DIR/logs/circuit_s*"