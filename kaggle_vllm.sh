#!/usr/bin/env bash
set -euo pipefail

CONDA="/kaggle/working/miniconda/bin/conda"
ENV_DIR="/kaggle/working/miniconda/envs/mcot-vllm"
PYTHON="${ENV_DIR}/bin/python"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.9.2/vllm-0.9.2%2Bcu126-cp38-abi3-manylinux1_x86_64.whl#sha256=d47bda5d71e03d975db5e0939fdb639dc8b9cd3e458ad4ab1f72b65ef630140d"
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

export PYTHONPATH=""
export PYTHONNOUSERSITE=1

case "${1:-}" in
    install)
        if [[ ! -x "$CONDA" ]]; then
            echo "Miniconda was not found. Run the Kaggle setup.sh first." >&2
            exit 1
        fi
        if [[ ! -x "$PYTHON" ]]; then
            "$CONDA" create -y -p "$ENV_DIR" -c conda-forge python=3.10 pip
        fi

        # Select the xFormers wheel built for the same CUDA runtime as PyTorch.
        "$PYTHON" -m pip install --only-binary=:all: --no-deps \
            --index-url "$TORCH_INDEX" "xformers==0.0.30"
        "$PYTHON" -m pip install --extra-index-url "$TORCH_INDEX" \
            "$VLLM_WHEEL" "torch==2.7.0+cu126" "torchvision==0.22.0+cu126" \
            "torchaudio==2.7.0+cu126" "transformers==4.53.3"
        "$PYTHON" -m pip check
        ;;
    serve)
        if [[ ! -x "${ENV_DIR}/bin/vllm" ]]; then
            echo "Run bash kaggle_vllm.sh install first." >&2
            exit 1
        fi

        export CUDA_VISIBLE_DEVICES="${MCOT_VLLM_GPU:-0}"
        export VLLM_USE_V1=0
        export VLLM_ATTENTION_BACKEND=XFORMERS
        "$PYTHON" -c '
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5):
    raise SystemExit("Select a CUDA GPU with compute capability 7.5 or newer, such as a Kaggle T4.")
print(f"vLLM GPU: {torch.cuda.get_device_name(0)} | CUDA runtime: {torch.version.cuda}", flush=True)
'

        exec "${ENV_DIR}/bin/vllm" serve "$MODEL" \
            --host 127.0.0.1 --port 8001 --dtype half \
            --max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization 0.85 \
            --limit-mm-per-prompt '{"image":1,"video":0}' \
            --mm-processor-kwargs '{"max_pixels":262144}' \
            --disable-mm-preprocessor-cache
        ;;
    *)
        echo "Usage: bash kaggle_vllm.sh {install|serve}" >&2
        exit 2
        ;;
esac
