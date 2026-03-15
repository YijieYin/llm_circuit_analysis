# llm_circuit_analysis

LLM-based neuron function hypothesis generation from connectome circuit data, designed to run as SLURM array jobs on a GPU cluster.

## Scripts

| File | Description |
|------|-------------|
| `interpret.py` | Main script — extracts circuit connectivity, queries LLM, saves JSONL |
| `submit_array.sh` | SLURM array submission (edit paths before use) |
| `merge_results.py` | Combines per-chunk outputs into one file |

## Setup

```bash
conda env create -f environment.yml
conda activate llm_circuit
pip install connectome_interpreter  # install separately from its own repo
```

API keys in `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

For local inference with llama.cpp, see [`docs/hpc_llama_setup.md`](docs/hpc_llama_setup.md).

## Usage

```bash
# Edit paths in submit_array.sh, then:
mkdir -p logs
sbatch submit_array.sh

# After jobs complete:
python merge_results.py --results-dir results --output hypotheses.jsonl
```

Resume is on by default — resubmit failed jobs and already-processed cells are skipped.

## Providers

| `--provider` | Notes |
|---|---|
| `llama` | Local GGUF via llama.cpp, free, no rate limits |
| `openai` | gpt-4o, gpt-5, gpt-5.4 (Responses API with `--reasoning-effort`, `--verbosity`) |
| `anthropic` | Extended thinking via `--thinking-budget` |