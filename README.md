<div align="center">
	
# MCoT-MVS: Multi-level Vision Selection by Multi-modal Chain-of-Thought Reasoning for Composed Image Retrieval (WWW2026 Oral)

[![arXiv](https://img.shields.io/badge/arXiv-2603.17360-b31b1b.svg)](https://arxiv.org/pdf/2603.17360)
[![ACM](https://img.shields.io/badge/ACM-Digital%20Library-007A5E.svg)](https://dl.acm.org/doi/10.1145/3774904.3792287)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>

## Installation

Use Python version **3.10.0**:

```bash
conda create -n MCoT-MVS python=3.10.0
conda activate MCoT-MVS
```

Then install the requirements:

```bash
pip install -r requirements.txt
```

---

## ⚡ Hướng Dẫn Setup & Test Nhanh (Kaggle / Linux)

### 1. Setup môi trường trên Kaggle (Chạy 1 lần trong Notebook trắng)
> **Lưu ý trước khi chạy:** 
> - Bật **GPU T4 x2** hoặc **P100** trong phần *Settings -> Accelerator*.
> - Đã Add Dataset chứa weights tại: `/kaggle/input/datasets/b22dckh099lngminhqu/mcot-mvs-weights/weights`

Chạy cell bash sau trên Kaggle:
```bash
%%bash
wget -q -O setup.sh https://raw.githubusercontent.com/LMQ-Ptit/MCoT-MVS-Online/quylm_v2/setup.sh
BRANCH=quylm_v2 bash setup.sh
```

---

### 2. Thiết lập đường dẫn import (Chạy đầu mỗi Session Python)
Trong notebook, thêm cell Python này trước khi import các module:
```python
import sys
sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/src")
sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/segment")
```

---

### 3. Test MCoT VLLM Extractor (Qwen2.5-VL-3B-Instruct)
Cell Python kiểm tra trích xuất đặc trưng đa phương thức (Target, Retained, Removed) từ ảnh và câu lệnh chỉnh sửa:
```python
import sys
sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/src")
sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/segment")

import os
import time
from PIL import Image
from vllm_extractor import MCoTExtractor

# 1. Khởi tạo mô hình
# - mode="local": Chạy trực tiếp model Qwen2.5-VL trên GPU (yêu cầu GPU VRAM >= 6GB)
# - mode="mock": Test nhanh pipeline logic mà không cần tải model
extractor = MCoTExtractor(
    mode="local",
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    device="cuda:0"   # Sử dụng cuda:0 (hoặc cuda:1 nếu chạy song song)
)

# 2. Warmup kernel GPU
extractor.warmup()

# 3. Chuẩn bị ảnh và câu lệnh test
sample_img_path = "/kaggle/working/MCoT-MVS-Online/data/fiq/fashionIQ_dataset/images/B005X4PL1G.png"
if os.path.exists(sample_img_path):
    img = Image.open(sample_img_path).convert("RGB")
else:
    print("⚠ Không tìm thấy ảnh mẫu, sử dụng ảnh synthetic (test logic)...")
    img = Image.new("RGB", (336, 336), color=(200, 70, 70))

modification_text = "is shiny and blue with longer sleeves and flared skirt"

# 4. Trích xuất MCoT
t0 = time.perf_counter()
result = extractor.extract(img, modification_text)
elapsed = time.perf_counter() - t0

print(f"\n⏱ Thời gian inference: {elapsed:.2f}s")
print(f"📝 Câu lệnh gốc: {modification_text}\n")
print(f"🎯 Target description : {result['target']}")
print(f"✅ Retained attributes : {result['retained']}")
print(f"❌ Deleted/Changed     : {result['deleted']}")
```

---

### 4. Test Segment Pipeline (RAM++ + Grounding DINO + SAM 2.1)
Cell Python kiểm tra toàn bộ pipeline tagging, detect bounding boxes và phân đoạn mặt nạ (masks):
```python
import sys
sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/src")
sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/segment")

import os
import time
from PIL import Image
from get_masks import load_segment_pipeline

# 1. Khởi tạo Segment Pipeline
# Tự động tải weights RAM++ và SAM 2.1 đã liên kết từ thư mục checkpoints
segment_pipe = load_segment_pipeline(
    device="cuda:0",
    load_clip=False    # Đặt True nếu muốn test luôn việc pool OpenCLIP features
)

# 2. Chuẩn bị ảnh đầu vào
sample_img_path = "/kaggle/working/MCoT-MVS-Online/data/fiq/fashionIQ_dataset/images/B005X4PL1G.png"
if os.path.exists(sample_img_path):
    img = Image.open(sample_img_path).convert("RGB")
else:
    print("⚠ Không tìm thấy ảnh mẫu, sử dụng ảnh synthetic...")
    img = Image.new("RGB", (384, 384), color=(50, 120, 200))

# 3. Chạy phân đoạn
t0 = time.perf_counter()
output = segment_pipe.process_image(
    image_input=img,
    save_dir="./test_segment_output",
    is_draw=True,
    compute_clip_features=False
)
elapsed = time.perf_counter() - t0

print(f"\n⏱ Thời gian xử lý Segment: {elapsed:.2f}s")
print(f"🏷 Tags nhận diện (RAM++)     : {output['tags']}")
print(f"📦 Bounding boxes (DINO)       : {len(output['boxes'])} boxes")
print(f"🎭 Masks tạo ra (SAM 2.1)      : {len(output['masks'])} masks")
if output.get("annotated_image_path"):
    print(f"🖼 Ảnh trực quan hóa lưu tại   : {output['annotated_image_path']}")
```

---

---

## Segment

If you want to generate masks and segmentation features manually, follow the steps below:

### 1. Download Checkpoints

* **RAM++**: [ram\_plus\_swin\_large\_14m.pth](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth)
  → Put it in `./segment/ram_models/` directory

* **SAM2**: [sam2.1\_hiera\_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
  → Put it in `./segment/sam2/checkpoints/` directory

### 2. Generate Masks

```bash
cd segment
python get_mask.py
```

### 3. Extract Segmentation Features

```bash
python extract_seg_feature_cirr.py
python extract_seg_feature_fiq.py
```

Or download the pre-extracted segmentation features directly:

* **CIRR**: [Download](https://drive.google.com/file/d/1MV8ITivi-Ik-qehj5Ud3aw11pG-cIp9V/view?usp=drive_link)
* **FashionIQ**: [Download](https://drive.google.com/file/d/19eBDQEFDpzSEALGSnm6lgSY31MJiYsLz/view?usp=drive_link)


---

## Data Structure

### FashionIQ

```
data/
└── fiq/
    ├── fashionIQ_dataset/
    │   ├── correction_dict_*.json
    │   ├── keywords_in_mods_*.json
    │   ├── captions/
    │   ├── images/
    │   ├── resized_image/
    │   └── image_splits/
    ├── llm_data/
    └── segment/seg_features_vit-h_patch/
```

> 📥 Download FashionIQ images from [Yandex Disk](https://disk.yandex.com/d/Z2E54WCwvrQA3A)
> Then run the [resize\_images.py](https://github.com/XiaoxiaoGuo/fashion-iq/blob/master/start_kit/resize_images.py) script.
> Resized images should be saved to `./data/fiq/fashionIQ_dataset/resized_image/`.

---

### CIRR

```
data/
└── cirr_dataset/
    ├── train/
    ├── dev/
    ├── test1/
    ├── cirr/
    │   ├── captions/
    │   └── image_splits/
    ├── llm_data/
    └── segment/seg_features_vit-h_patch/
```

> 📥 Follow the [official CIRR repository](https://github.com/Cuberick-Orion/CIRR) to download and unzip the dataset.
> Place the files under `./data/cirr_dataset/`.

---

## Training

```bash
cd src
# cirr
CUDA_VISIBLE_DEVICES=<device_id> python train.py --dataset cirr --loss_weight 100.0

# fiq
CUDA_VISIBLE_DEVICES=<device_id> python train.py --dataset dress --loss_weight 10.0

CUDA_VISIBLE_DEVICES=<device_id> python train.py --dataset shirt --loss_weight 10.0

CUDA_VISIBLE_DEVICES=<device_id> python train.py --dataset toptee --loss_weight 10.0
```


---

## Inference

Download pre-trained checkpoints:

* **CIRR checkpoint**: [Download](https://drive.google.com/file/d/1OJRFIk1TTwd4iRgNNcpBESrNMJuUVo2h/view?usp=drive_link)
* **dress checkpoint**: [Download](https://drive.google.com/file/d/1tNMtH1SABWU7HsREc297STgkj52gtsFL/view?usp=drive_link)
* **shirt checkpoint**: [Download](https://drive.google.com/file/d/1r526gq_lHyDxm1fhrTxDwsFzChDNNTf3/view?usp=drive_link)
* **toptee checkpoint**: [Download](https://1drv.ms/u/c/a76cc9a14c451e3b/EatrK0L9tc9IqboXxSN76DcBwIoXtuDosRsMkn7XRWiFeg?e=t2ZxuS)
  → Place in the `checkpoints/` directory

### CIRR Inference

```bash
cd src
CUDA_VISIBLE_DEVICES=<device_id> python cirr_test_submission.py --model_path checkpoints/your_model.pt --submission_name test
```

### FashionIQ Inference

```bash
cd src
CUDA_VISIBLE_DEVICES=<device_id> python fiq_validate.py --model_path checkpoints/your_model.pt --dress_type dress
```