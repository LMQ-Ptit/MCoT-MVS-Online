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
> - Bật Internet và chọn **GPU T4 x2** trong *Settings* để thử Qwen2.5-VL NF4. Cell test chọn một GPU theo VRAM còn trống.
> - Đã Add Dataset chứa weights tại: `/kaggle/input/datasets/b22dckh099lngminhqu/mcot-mvs-weights/weights`

Chạy cell bash sau trên Kaggle:
```bash
%%bash
wget -q -O setup.sh https://raw.githubusercontent.com/LMQ-Ptit/MCoT-MVS-Online/quylm_v2/setup.sh
BRANCH=quylm_v2 bash setup.sh
```

`setup.sh` cài `bitsandbytes==0.45.5` trong môi trường `py310` và giữ các phiên bản PyTorch 2.3.1 (CUDA 12.1), torchvision 0.18.1, Transformers 4.49.0 và accelerate 1.4.0.

---

### 2. Thiết lập đường dẫn import (Chạy đầu mỗi Session Python)
Trong notebook, thêm cell Python này trước khi import các module:
```python
import sys
sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/src")
sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/segment")
```

---

### 3. Test Qwen2.5-VL FP16 và NF4 trên Kaggle

Chạy cell dưới đây với cùng `image_url` và câu lệnh chỉnh sửa. Chỉ cần đổi `model_name` giữa `qwen-vl-3b-4bit`, `qwen-vl-7b-4bit` và checkpoint gốc 3B FP16 để so sánh.

Hai preset 4-bit tải checkpoint gốc từ Hugging Face rồi lượng tử hóa bằng bitsandbytes NF4 khi nạp, với phần xử lý ảnh và lớp đầu ra giữ FP16. Có thể truyền ID checkpoint gốc cùng `load_in_4bit=True` vào `MCoTExtractor` để có cùng cấu hình. Lần đầu cần tải đầy đủ trọng số gốc.

Mỗi lượt dùng một tiến trình `py310` riêng để giải phóng model local khi cell kết thúc. Cell gọi `extractor.warmup()` cho model local và chạy thử ảnh test trước khi đo.

```python
%%script /kaggle/working/miniconda/envs/py310/bin/python
import sys
import time
from io import BytesIO
from statistics import median
from urllib.request import urlopen

import torch
from PIL import Image

sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/src")
from vllm_extractor import MCoTExtractor

mode = "local"  # Dùng "api" với server vLLM ở mục 4
model_name = "qwen-vl-3b-4bit"
# model_name = "qwen-vl-7b-4bit"
# model_name = "Qwen/Qwen2.5-VL-3B-Instruct"  # Mốc FP16 và tên model trên vLLM
gpu_index = None  # Đặt 0 để so sánh FP16 trên cùng T4 mặc định của vLLM
repetitions = 3
image_url = "https://raw.githubusercontent.com/open-mmlab/mmdetection/master/demo/demo.jpg"
modification_text = "is shiny and blue with longer sleeves and flared skirt"

with urlopen(image_url, timeout=30) as response:
    with Image.open(BytesIO(response.read())) as source_image:
        img = source_image.convert("RGB")
img.thumbnail((512, 512), Image.Resampling.LANCZOS)
print(f"Kích thước ảnh test: {img.width}×{img.height}")

if mode == "local":
    if not torch.cuda.is_available():
        raise RuntimeError("Bật GPU trong Kaggle trước khi chạy cell này.")
    free_by_gpu = {}
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        free_by_gpu[index] = free
        print(f"cuda:{index} ({torch.cuda.get_device_name(index)}): {free / 2**30:.2f}/{total / 2**30:.2f} GiB trống")
    if gpu_index is None:
        gpu_index = max(free_by_gpu, key=free_by_gpu.get)
    device = f"cuda:{gpu_index}"
    extractor = MCoTExtractor(mode=mode, model_name=model_name, device=device)
    extractor.warmup()
elif mode == "api":
    extractor = MCoTExtractor(
        mode=mode, model_name=model_name, device="cpu",
        api_base="http://127.0.0.1:8001/v1", api_key="EMPTY",
    )
else:
    raise ValueError("Chọn mode='local' hoặc mode='api'.")

# Warm up with the benchmark's image and prompt in either mode.
extractor.extract(img, modification_text)
if mode == "local":
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

latencies = []
for _ in range(repetitions):
    t0 = time.perf_counter()
    result = extractor.extract(img, modification_text)
    if mode == "local":
        torch.cuda.synchronize(device)
    latencies.append(time.perf_counter() - t0)

print(f"\n{mode} / {model_name}")
print("Inference sau warmup: " + ", ".join(f"{t:.2f}s" for t in latencies))
print(f"Latency trung vị: {median(latencies):.2f}s")
if mode == "local":
    print(f"Peak VRAM sau warmup (PyTorch): {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
print(f"📝 Câu lệnh gốc: {modification_text}\n")
print(f"🎯 Target description : {result['target']}")
print(f"✅ Retained attributes : {result['retained']}")
print(f"❌ Deleted/Changed     : {result['deleted']}")
```

---

### 4. Chạy Qwen2.5-VL qua vLLM trên Kaggle

Sau bước 1, cài vLLM vào môi trường `mcot-vllm` riêng. Server dùng vLLM 0.9.2, CUDA runtime 12.6 và backend xFormers trên T4. Client và các cell Transformers vẫn dùng `py310`. [Wheel CUDA 12.6](https://github.com/vllm-project/vllm/releases/tag/v0.9.2)

```bash
%%bash
bash /kaggle/working/MCoT-MVS-Online/kaggle_vllm.sh install
```

Chạy cell Python sau để khởi động Qwen2.5-VL-3B FP16 trên GPU 0. Server lắng nghe ở `127.0.0.1:8001` trong notebook. Lần đầu khởi động cần tải checkpoint.

```python
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

def stop_vllm(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

with socket.socket() as connection:
    if connection.connect_ex(("127.0.0.1", 8001)) == 0:
        raise RuntimeError("Cổng 8001 đang được sử dụng. Dừng server cũ trước khi chạy lại.")

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
log_path = Path("/kaggle/working/mcot-vllm.log")
with log_path.open("w", encoding="utf-8") as log:
    vllm_process = subprocess.Popen(
        ["bash", "/kaggle/working/MCoT-MVS-Online/kaggle_vllm.sh", "serve"],
        env={**os.environ, "MCOT_VLLM_GPU": "0"},
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )

try:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if vllm_process.poll() is not None:
            raise RuntimeError(f"vLLM đã dừng với mã {vllm_process.returncode}.")
        try:
            with urlopen("http://127.0.0.1:8001/v1/models", timeout=2) as response:
                served_models = json.load(response)["data"]
            if any(model["id"] == model_id for model in served_models):
                break
        except (URLError, TimeoutError):
            pass
        time.sleep(2)
    else:
        raise TimeoutError("Server vLLM chưa sẵn sàng sau 15 phút.")
except BaseException:
    stop_vllm(vllm_process)
    print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
    raise

print("vLLM đã sẵn sàng tại http://127.0.0.1:8001/v1")
```

Để đo vLLM, chạy lại cell mục 3 với `mode = "api"` và `model_name = "Qwen/Qwen2.5-VL-3B-Instruct"`. Cell gửi một request ảnh để warmup rồi đo ba request tuần tự. Sau khi đo, dừng server bằng `stop_vllm(vllm_process)`. Tiếp theo chạy cell mục 3 với `mode = "local"`, cùng `model_name` và `gpu_index = 0` để lấy mốc Transformers FP16 trên cùng GPU.

Giữ cùng ảnh và prompt. Số đo API bao gồm mã hóa ảnh và HTTP localhost. Số token đầu ra có thể khác nhau giữa hai engine và ảnh hưởng tổng latency. Bài đo này kiểm tra latency từng request. Để đánh giá lợi ích batching của vLLM, cần đo throughput với nhiều request đồng thời.

---

### 5. Test Segment Pipeline (RAM++ + Grounding DINO + SAM 2.1)
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
