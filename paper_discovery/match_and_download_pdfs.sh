#!/bin/bash
#SBATCH --job-name=pdf_match_dl
#SBATCH --partition=cpu              # no GPU needed — pure CPU/IO
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=paper_discovery/logs/pdf_%j.out
#SBATCH --error=paper_discovery/logs/pdf_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yy432@cam.ac.uk

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

# rapidfuzz + PyMuPDF should already be installed (used by extraction);
# install quietly if missing
pip install -q rapidfuzz PyMuPDF requests 2>/dev/null || true

# ---- Config ----
SCRIPT="$HOME/llm_circuit_analysis/paper_discovery/match_and_download_pdfs.py"

DISCOVERED="$HOME/llm_circuit_analysis/paper_discovery/output/discovered_papers.csv"
PDFS_DIR="/cephfs2/yyin/llm_circuit_analysis/pdfs"                      # existing PDFs
OUTPUT="$HOME/llm_circuit_analysis/paper_discovery/output/discovered_papers_with_local.csv"

mkdir -p "$HOME/llm_circuit_analysis/paper_discovery/logs"

echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Discovered: $DISCOVERED"
echo "PDFs dir:   $PDFS_DIR"
echo "Output:     $OUTPUT"
echo "=============================="

python "$SCRIPT" \
    --discovered-csv         "$DISCOVERED" \
    --pdfs-dir               "$PDFS_DIR" \
    --output-csv             "$OUTPUT" \
    --similarity-threshold   85 \
    --n-workers              8

echo "Done."

# ---- Notes ----
# This step is CPU/IO-bound. You can also run it interactively on a login node
# inside tmux — it doesn't need to queue:
#   conda activate llm
#   python match_and_download_pdfs.py --discovered-csv ... --pdfs-dir ... ...
#
# PDF scan is cached by mtime in pdf_scan_cache.json — re-runs only re-process
# changed files.
#
# To match only (no downloads): add --skip-download
# To limit downloads (testing): add --max-downloads 20