#!/bin/bash
# =============================================================
# setup.sh – Cài đặt môi trường MCoT-MVS trên Kaggle / Linux
#
# Cách dùng trên Kaggle notebook:
#   BRANCH=quylm_v2 bash setup.sh
#
# Biến môi trường có thể override:
#   BRANCH   – nhánh git cần dùng          (mặc định: quylm_v2)
#   SKIP_ENV – bỏ qua tạo conda env nếu =1 (mặc định: 0)
# =============================================================
set -euo pipefail
export PYTHONPATH=""

# ── Cấu hình đường dẫn ────────────────────────────────────────
CONDA_DIR="/kaggle/working/miniconda"
ENV_DIR="${CONDA_DIR}/envs/py310"
PIP="${ENV_DIR}/bin/pip"
PY="${ENV_DIR}/bin/python"
REPO_DIR="/kaggle/working/MCoT-MVS-Online"
BRANCH="${BRANCH:-quylm_v2}"
SKIP_ENV="${SKIP_ENV:-0}"

# ── Màu cho log ───────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${GREEN}══════════════════════════════════════════${NC}"; \
         echo -e "${GREEN} $1${NC}"; \
         echo -e "${GREEN}══════════════════════════════════════════${NC}"; }
info() { echo -e "  ${YELLOW}▶${NC} $1"; }
ok()   { echo -e "  ${GREEN}✔${NC} $1"; }
err()  { echo -e "  ${RED}✘${NC} $1"; exit 1; }

# =============================================================
step "BƯỚC 1: MINICONDA & PYTHON 3.10"
# =============================================================
if [ ! -d "$CONDA_DIR" ]; then
    info "Tải Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR" -u > /dev/null
    rm /tmp/miniconda.sh
    # Chấp nhận ToS (không chặn script)
    "$CONDA_DIR/bin/conda" tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main > /dev/null 2>&1 || true
    "$CONDA_DIR/bin/conda" tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/r    > /dev/null 2>&1 || true
    ok "Miniconda đã cài xong."
else
    ok "Miniconda đã tồn tại, bỏ qua."
fi

if [ "$SKIP_ENV" = "0" ] && [ ! -d "$ENV_DIR" ]; then
    info "Tạo môi trường py310 (Python 3.10)..."
    "$CONDA_DIR/bin/conda" create -n py310 -c conda-forge python=3.10 -y -q > /dev/null
    ok "Môi trường py310 đã tạo xong."
else
    ok "Môi trường py310 đã tồn tại, bỏ qua."
fi

# =============================================================
step "BƯỚC 2: CLONE / CẬP NHẬT MÃ NGUỒN (nhánh: ${BRANCH})"
# =============================================================
cd /kaggle/working
if [ ! -d "MCoT-MVS-Online/.git" ]; then
    info "Clone repo nhánh ${BRANCH}..."
    git clone --depth=1 -b "${BRANCH}" https://github.com/LMQ-Ptit/MCoT-MVS-Online.git
else
    info "Repo đã tồn tại, cập nhật nhánh ${BRANCH}..."
    (cd MCoT-MVS-Online \
        && git fetch --depth=1 origin \
        && git checkout "${BRANCH}" \
        && git reset --hard "origin/${BRANCH}")
fi
ok "Đang dùng nhánh: $(cd MCoT-MVS-Online && git rev-parse --abbrev-ref HEAD) | commit: $(cd MCoT-MVS-Online && git rev-parse --short HEAD)"

# =============================================================
step "BƯỚC 3: PYTORCH 2.3.1 (CUDA 12.1) & DEPENDENCIES"
# =============================================================
info "Cài PyTorch 2.3.1 + torchvision..."
$PIP install -q \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

info "Cài requirements.txt (đã bao gồm transformers, accelerate, qwen-vl-utils, fastapi...)..."
$PIP install -q -r "${REPO_DIR}/requirements.txt"

info "Cài recognize-anything (RAM++) từ GitHub..."
$PIP install -q git+https://github.com/xinyu1205/recognize-anything.git

info "Cài bitsandbytes 0.45.5 để nạp Qwen2.5-VL NF4..."
"$PY" -m pip install -q --no-deps bitsandbytes==0.45.5

info "Cài pyngrok vào Python mặc định notebook..."
pip install -q pyngrok

# =============================================================
step "BƯỚC 4: LIÊN KẾT TRỌNG SỐ TỪ DATASET KAGGLE"
# =============================================================
mkdir -p "${REPO_DIR}/segment/sam2/checkpoints"
mkdir -p "${REPO_DIR}/segment/ram_models"
mkdir -p "${REPO_DIR}/src/checkpoints"

# Thư mục chứa toàn bộ trọng số (đã upload sẵn trên Kaggle)
WEIGHTS_DIR="/kaggle/input/datasets/b22dckh099lngminhqu/mcot-mvs-weights/weights"

# Kiểm tra thư mục weights tồn tại
[ -d "$WEIGHTS_DIR" ] || err "Không tìm thấy thư mục weights tại '$WEIGHTS_DIR'. Hãy Add dataset 'mcot-mvs-weights' vào notebook."

# Hàm symlink weight với đường dẫn tường minh
link_weight() {   # $1=tên file trong WEIGHTS_DIR  $2=dest
    local src="${WEIGHTS_DIR}/$1"
    [ -f "$src" ] || err "Không tìm thấy file '$1' trong $WEIGHTS_DIR"
    ln -sf "$src" "$2"
    ok "$(basename "$2")  →  $src"
}

link_weight "sam2.1_hiera_large.pt"       "${REPO_DIR}/segment/sam2/checkpoints/sam2.1_hiera_large.pt"
link_weight "ram_plus_swin_large_14m.pth" "${REPO_DIR}/segment/ram_models/ram_plus_swin_large_14m.pth"
link_weight "dress.pt"                    "${REPO_DIR}/src/checkpoints/dress.pt"
link_weight "shirt.pt"                    "${REPO_DIR}/src/checkpoints/shirt.pt"

# =============================================================
step "BƯỚC 5: KIỂM TRA MÔI TRƯỜNG"
# =============================================================
$PY - <<'PYCHECK'
import sys, importlib
checks = {
    "Python"       : sys.version.split()[0],
    "torch"        : __import__("torch").__version__,
    "transformers" : __import__("transformers").__version__,
    "open_clip"    : __import__("open_clip").__version__,
    "fastapi"      : __import__("fastapi").__version__,
}
for k, v in checks.items():
    print(f"  ✔ {k:<16} {v}")

import torch
print(f"  ✔ {'CUDA':16} available={torch.cuda.is_available()} | GPUs={torch.cuda.device_count()}")
PYCHECK

echo ""
echo -e "${GREEN}🎉 SETUP HOÀN TẤT – sẵn sàng chạy pipeline!${NC}"
