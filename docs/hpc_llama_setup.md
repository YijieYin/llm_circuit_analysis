# HPC Setup: llama.cpp with CUDA

Tested on: GLIBC 2.28, CUDA driver 12.2, A100 (sm_80).

## 1. Load modules

```bash
module load cuda/12.4           # must be ≤ driver version; supports GCC ≤ 12
module load compilers/gcc/12.2.0
export LD_LIBRARY_PATH=/public/gcc/12_2_0/lib64:$LD_LIBRARY_PATH
```

Add to `~/.bashrc` to persist.

## 2. Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DCMAKE_C_COMPILER=/public/gcc/12_2_0/bin/gcc \
  -DCMAKE_CXX_COMPILER=/public/gcc/12_2_0/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES=80    # A100=80, RTX4090=89, RTX3090=86, RTX2080Ti=75
cmake --build build --config Release -j$(nproc)
```

**Common issues:**
- `CUDA Toolkit not found` → set `-DCUDAToolkit_ROOT` explicitly
- `unsupported GNU version` → use GCC ≤ 12 for CUDA 12.x
- `std::filesystem` linker errors → GCC too old or `LD_LIBRARY_PATH` not set
- `device kernel image is invalid` → wrong `CMAKE_CUDA_ARCHITECTURES` for your GPU

## 3. Download model

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='bartowski/Qwen_Qwen3.5-35B-A3B-GGUF',
    allow_patterns='*Q4_K_M*',
    local_dir='/path/to/models/',
    token='YOUR_HF_TOKEN'
)
"
```

Set `HF_HOME` to a large storage partition to avoid filling your home directory quota.

## 4. Test

```bash
# Start server
~/llama.cpp/build/bin/llama-server \
    -m /path/to/model.gguf -ngl 99 --port 8080

# Test in another terminal
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}'
```