#!/bin/bash
# =============================================================
# setup.sh - Cài đặt môi trường MCoT-MVS trên Kaggle
# Chạy trong notebook:
#   !wget -q -O setup.sh https://raw.githubusercontent.com/<USER>/<REPO>/main/setup.sh
#   !bash setup.sh
# =============================================================
set -e
export PYTHONPATH=""

CONDA_DIR="/kaggle/working/miniconda"
ENV_DIR="${CONDA_DIR}/envs/py310"
PIP="${ENV_DIR}/bin/pip"
PY="${ENV_DIR}/bin/python"
REPO_DIR="/kaggle/working/MCoT-MVS-Online"

echo "=========================================================="
echo "BƯỚC 1: MINICONDA & PYTHON 3.10"
echo "=========================================================="
if [ ! -d "$CONDA_DIR" ]; then
    echo "• Tải và cài đặt Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $CONDA_DIR -u > /dev/null
    rm miniconda.sh
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main > /dev/null 2>&1 || true
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r > /dev/null 2>&1 || true
fi

if [ ! -d "$ENV_DIR" ]; then
    echo "• Tạo môi trường py310 (Python 3.10)..."
    $CONDA_DIR/bin/conda create -n py310 -c conda-forge python=3.10 -y -q
fi

echo "=========================================================="
echo "BƯỚC 2: CLONE / CẬP NHẬT MÃ NGUỒN"
echo "=========================================================="
cd /kaggle/working
if [ ! -d "MCoT-MVS-Online" ]; then
    git clone https://github.com/LMQ-Ptit/MCoT-MVS-Online.git
else
    (cd MCoT-MVS-Online && git pull)
fi

echo "=========================================================="
echo "BƯỚC 3: PYTORCH 2.3.1 (CUDA 12.1) & DEPENDENCIES"
echo "=========================================================="
echo "• Cài PyTorch 2.3.1..."
$PIP install -q torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

echo "• Cài requirements.txt..."
$PIP install -q -r ${REPO_DIR}/requirements.txt

echo "• Cài thư viện cho Segment & Qwen2.5-VL..."
$PIP install -q \
    transformers==4.49.0 \
    accelerate==1.4.0 \
    qwen-vl-utils==0.0.8 \
    timm==0.9.16 \
    fairscale==0.4.13 \
    pycocotools \
    git+https://github.com/xinyu1205/recognize-anything.git

echo "• Cài thư viện cho API server (env py310)..."
$PIP install -q fastapi "uvicorn[standard]" pyjwt python-multipart

echo "• Cài pyngrok (Python mặc định của notebook)..."
pip install -q pyngrok

echo "=========================================================="
echo "BƯỚC 4: LIÊN KẾT TRỌNG SỐ TỪ DATASET KAGGLE"
echo "=========================================================="
mkdir -p ${REPO_DIR}/segment/sam2/checkpoints
mkdir -p ${REPO_DIR}/segment/ram_models
mkdir -p ${REPO_DIR}/src/checkpoints

link_weight () {   # $1 = mẫu tên file, $2 = đường dẫn đích
    local src
    src=$(find /kaggle/input -name "$1" 2>/dev/null | head -n 1)
    if [ -z "$src" ]; then
        echo "❌ Không tìm thấy '$1' trong /kaggle/input. Hãy Add Input dataset chứa trọng số."
        exit 1
    fi
    ln -sf "$src" "$2"
}

link_weight "sam2.1_hiera_large.pt"      ${REPO_DIR}/segment/sam2/checkpoints/sam2.1_hiera_large.pt
link_weight "*ram_plus_swin_large*"      ${REPO_DIR}/segment/ram_models/ram_plus_swin_large_14m.pth
link_weight "dress.pt"                   ${REPO_DIR}/src/checkpoints/dress.pt
link_weight "shirt.pt"                   ${REPO_DIR}/src/checkpoints/shirt.pt

echo "Kiểm tra liên kết trọng số:"
ls -lhL ${REPO_DIR}/segment/sam2/checkpoints/sam2.1_hiera_large.pt
ls -lhL ${REPO_DIR}/segment/ram_models/ram_plus_swin_large_14m.pth
ls -lhL ${REPO_DIR}/src/checkpoints/dress.pt
ls -lhL ${REPO_DIR}/src/checkpoints/shirt.pt

echo "=========================================================="
echo "BƯỚC 5: KIỂM TRA MÔI TRƯỜNG"
echo "=========================================================="
$PY -c "import sys; print('Python:', sys.version.split()[0])"
$PY -c "import transformers; print('Transformers:', transformers.__version__)"
$PY -c "import torch; print('GPU sẵn có:', torch.cuda.is_available(), '| Số GPU:', torch.cuda.device_count())"
$PY -c "import fastapi, uvicorn, jwt; print('FastAPI', fastapi.__version__, '| PyJWT', jwt.__version__)"

echo "=========================================================="
echo "🎉 SETUP HOÀN TẤT. Tiếp theo chạy cell khởi động server."
echo "=========================================================="
