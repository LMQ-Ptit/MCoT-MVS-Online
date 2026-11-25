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

## Hướng dẫn chạy trên Kaggle

Mặc định, hai bước Qwen và backend dùng **Qwen/Qwen2.5-VL-3B-Instruct FP16**. Mỗi cell benchmark tự tải cùng ảnh váy dạ hội xanh, chỉnh hướng ảnh theo EXIF, chuyển sang RGB và thu về tối đa 512 px mỗi chiều trước khi warmup và đo. Để thử ảnh khác, đổi `load_test_image()` thành `load_test_image("https://...")` trong cell đang chạy. Dùng cùng URL nếu muốn so sánh latency giữa các bước. [Ảnh Public Domain của The Met](https://www.metmuseum.org/art/collection/search/159159)

### 1. Setup môi trường

Bật Internet và chọn **GPU T4 x2**. Thêm dataset trọng số `mcot-mvs-weights` để có thư mục `/kaggle/input/datasets/b22dckh099lngminhqu/mcot-mvs-weights/weights`.

```bash
%%bash
set -e
wget -q -O setup.sh https://raw.githubusercontent.com/LMQ-Ptit/MCoT-MVS-Online/quylm_v2/setup.sh
time env BRANCH=quylm_v2 bash setup.sh
```

Dòng `time` ở bước này đo thời gian setup. Các cell inference bên dưới sử dụng môi trường `py310` với PyTorch 2.3.1 (CUDA 12.1), torchvision 0.18.1, Transformers 4.49.0 và accelerate 1.4.0.

### 2. Qwen local qua Transformers

Cell dùng `MCoTExtractor(mode="local")` trên GPU 0. Model được giải phóng khi cell kết thúc. Có thể đổi `model_name` thành `qwen-vl-3b-4bit` hoặc `qwen-vl-7b-4bit` để thử NF4 riêng. Giữ checkpoint FP16 mặc định khi so với vLLM ở mục 3.

```python
%%script /kaggle/working/miniconda/envs/py310/bin/python
import sys

import torch

sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/src")
from benchmark_latency import load_test_image, measure_inference
from vllm_extractor import MCoTExtractor

device = "cuda:0"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
img = load_test_image()
text = "make it black"

extractor = MCoTExtractor(mode="local", model_name=model_name, device=device)
extractor.warmup()
extractor.extract(img, text)
torch.cuda.synchronize(device)
torch.cuda.reset_peak_memory_stats(device)
result = measure_inference(
    lambda: extractor.extract(img, text),
    synchronize=lambda: torch.cuda.synchronize(device),
)
print(f"Peak VRAM (PyTorch): {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
print(result)
```

### 3. Qwen qua API vLLM

Cài vLLM vào môi trường riêng `mcot-vllm`. Launcher dùng vLLM 0.9.2, CUDA runtime 12.6 và xFormers trên T4. Client vẫn chạy trong `py310`. [Wheel CUDA 12.6](https://github.com/vllm-project/vllm/releases/tag/v0.9.2)

```bash
%%bash
bash /kaggle/working/MCoT-MVS-Online/kaggle_vllm.sh install
```

Chạy cell sau bằng kernel Python mặc định để khai báo hai hàm quản lý vLLM. Hai hàm này cũng được dùng để host toàn pipeline ở mục 5.

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
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def start_vllm(action="serve"):
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", 8001)) == 0:
            raise RuntimeError("Dừng server đang dùng cổng 8001 trước khi chạy lại.")

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    log_path = Path("/kaggle/working/mcot-vllm.log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["bash", "/kaggle/working/MCoT-MVS-Online/kaggle_vllm.sh", action],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )

    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM đã dừng với mã {process.returncode}.")
            try:
                with urlopen("http://127.0.0.1:8001/v1/models", timeout=2) as response:
                    models = json.load(response)["data"]
                if any(model["id"] == model_id for model in models):
                    print("vLLM đã sẵn sàng tại http://127.0.0.1:8001/v1")
                    return process
            except (URLError, TimeoutError):
                pass
            time.sleep(2)
        raise TimeoutError("Server vLLM chưa sẵn sàng sau 15 phút.")
    except BaseException:
        stop_vllm(process)
        print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
        raise


```

Để benchmark riêng Qwen qua API, chạy cell này để khởi động 3B FP16 trên GPU 0 và đợi API `127.0.0.1:8001` sẵn sàng. Nếu chỉ muốn host toàn pipeline, chuyển thẳng sang mục 5 sau khi khai báo hai hàm.

```python
vllm_process = start_vllm()
```

Gửi một request ảnh để warmup, rồi đo ba request qua API. Số đo bao gồm mã hóa ảnh và HTTP localhost. Hai engine có thể sinh số token khác nhau, vì vậy đây là phép so latency tổng trên cùng đầu vào.

```python
%%script /kaggle/working/miniconda/envs/py310/bin/python
import sys

sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/src")
from benchmark_latency import load_test_image, measure_inference
from vllm_extractor import MCoTExtractor

extractor = MCoTExtractor(
    mode="api",
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    device="cpu",
    api_base="http://127.0.0.1:8001/v1",
    api_key="EMPTY",
)
img = load_test_image()
text = "make it black"

extractor.extract(img, text)
result = measure_inference(lambda: extractor.extract(img, text))
print(result)
```

### 4. Segment

Đo RAM++, Grounding DINO, SAM 2.1 và pooling OpenCLIP trên **GPU 1**. vLLM benchmark ở mục 3 có thể tiếp tục chạy trên **GPU 0**. Cell tự tải ảnh test, chạy một lượt warmup và đồng bộ GPU 1 khi đo.

```python
%%script /kaggle/working/miniconda/envs/py310/bin/python
import sys

import torch

sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/src")
sys.path.insert(0, "/kaggle/working/MCoT-MVS-Online/segment")
from benchmark_latency import load_test_image, measure_inference
from get_masks import load_segment_pipeline

device = "cuda:1"
img = load_test_image()

with torch.cuda.device(device):
    segment = load_segment_pipeline(device=device, load_clip=True)

    @torch.inference_mode()
    def infer_segment():
        return segment.process_image(img, is_draw=False, compute_clip_features=True)

    infer_segment()
    result = measure_inference(infer_segment, synchronize=lambda: torch.cuda.synchronize(device))

print(f"Tags: {result['tags']}")
print(f"Boxes: {len(result['boxes'])} | Masks: {len(result['masks'])}")
print(f"Segment features: {tuple(result['seg_features'].shape)}")
```

### 5. Backend và host toàn bộ pipeline

**Chuẩn bị.** Thêm dataset gallery `mcot-vectors` chứa `gallery_dress_vectors.pt` và tạo hai Kaggle Secrets `API_KEY`, `JWT_SECRET`.

Để chạy thẳng full pipeline, chạy setup ở mục 1, rồi cell cài vLLM và cell khai báo `start_vllm()`/`stop_vllm()` ở mục 3. Sau đó chạy ba bước bên dưới theo thứ tự.

**Bước 1. Khởi động vLLM cho toàn pipeline.** Chạy cell này trong kernel mặc định. Nếu đã chạy vLLM để benchmark ở mục 3, cell sẽ dừng tiến trình đó trước.

```python
if "vllm_process" in globals():
    stop_vllm(vllm_process)
vllm_process = start_vllm("serve-pipeline")
```

Cấu hình pipeline đặt segmentation trên **GPU 0**, còn **GPU 1** chạy vLLM và `CIRModel`. vLLM dùng ngân sách **60% VRAM**, một sequence và eager mode để chừa bộ nhớ cho CIR. Nếu thiếu VRAM khi chạy chung, kiểm tra GPU còn tiến trình cũ bằng `nvidia-smi` và đọc hai log startup. Có thể chỉnh `MCOT_VLLM_GPU_MEMORY_UTILIZATION` trước khi khởi động vLLM. Latency của cấu hình chạy chung có thể khác bài đo vLLM riêng ở mục 3.

**Bước 2. Khởi động FastAPI.** Chạy cell Python này trong kernel notebook mặc định. Uvicorn sẽ được khởi động bằng Python `py310`.

Mặc định, FastAPI gọi Qwen qua vLLM trong lúc GPU 0 chạy Segment. CIR bắt đầu khi cả hai bước đã trả kết quả. `MCOT_PARALLEL_BRANCHES=1` bật chế độ song song. Dùng `0` để đo chế độ tuần tự với cùng server vLLM.

```python
import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from kaggle_secrets import UserSecretsClient

repo = Path("/kaggle/working/MCoT-MVS-Online")
search_python = "/kaggle/working/miniconda/envs/py310/bin/python"
search_base = "http://127.0.0.1:8000"
log_path = Path("/kaggle/working/mcot-search.log")

with socket.socket() as connection:
    if connection.connect_ex(("127.0.0.1", 8000)) == 0:
        raise RuntimeError("Dừng server đang dùng cổng 8000 trước khi chạy lại.")

secrets_client = UserSecretsClient()
search_api_key = secrets_client.get_secret("API_KEY")
env = {
    **os.environ,
    "PYTHONPATH": "",
    "MCOT_SEARCH_API_KEY": search_api_key,
    "MCOT_SEARCH_JWT_SECRET": secrets_client.get_secret("JWT_SECRET"),
    "MCOT_EXTRACTOR_MODE": "api",
    "MCOT_PARALLEL_BRANCHES": "1",
    "MCOT_VLLM_BASE_URL": "http://127.0.0.1:8001/v1",
    "MCOT_GALLERY_PATH": "/kaggle/input/datasets/thichuy/mcot-vectors/gallery_dress_vectors.pt",
}
with log_path.open("w", encoding="utf-8") as log:
    search_process = subprocess.Popen(
        [search_python, "-m", "uvicorn", "search_api:app",
         "--app-dir", str(repo / "src"), "--host", "127.0.0.1", "--port", "8000"],
        cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
    )

try:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if search_process.poll() is not None:
            raise RuntimeError(f"FastAPI đã dừng với mã {search_process.returncode}.")
        try:
            with urlopen(f"{search_base}/openapi.json", timeout=2):
                break
        except (URLError, TimeoutError):
            time.sleep(2)
    else:
        raise TimeoutError("FastAPI chưa sẵn sàng sau 15 phút.")
except BaseException:
    if search_process.poll() is None:
        search_process.terminate()
        search_process.wait(timeout=30)
    print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
    raise

print(f"FastAPI đã sẵn sàng tại {search_base}")
```

`MCOT_CIR_CHECKPOINT` có thể trỏ đến checkpoint `dress.pt` riêng. Mặc định backend dùng checkpoint mà `setup.sh` đã liên kết tại `src/checkpoints/dress.pt`.

**Bước 3. Lấy JWT, warmup rồi gọi tìm kiếm.** Cell chạy một request đầy đủ để warmup và đo ba request tiếp theo. Header `Server-Timing` ghi các mốc tính bằng mili giây. `segment` bao gồm I/O của nhánh Segment, `mcot` là thời gian bước Qwen qua API, `branches` là thời gian thực từ khi bắt đầu hai nhánh đến khi cả hai xong, còn `total` là tổng xử lý bên trong endpoint. Khi chạy song song, dùng `branches` để xem thời gian chờ thực tế của hai nhánh.

```python
import sys
from io import BytesIO

import requests

sys.path.insert(0, str(repo / "src"))
from benchmark_latency import load_test_image, measure_inference

token_response = requests.post(
    f"{search_base}/api/token", headers={"X-API-KEY": search_api_key}, timeout=30,
)
token_response.raise_for_status()
authorization = {"Authorization": f"Bearer {token_response.json()['access_token']}"}
img = load_test_image()
with BytesIO() as image_buffer:
    img.save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()

def search_once():
    response = requests.post(
        f"{search_base}/api/list",
        headers=authorization,
        files={"image": ("query.png", image_bytes, "image/png")},
        data={"text": "make it black", "top_k": 5},
        timeout=120,
    )
    response.raise_for_status()
    return response

search_once()  # Warmup toàn bộ pipeline
server_timings = []

def infer_search():
    response = search_once()
    server_timings.append(response.headers.get("Server-Timing", ""))
    return response

result = measure_inference(infer_search)
for index, timing in enumerate(server_timings, start=1):
    print(f"Request {index} | {timing}")
print(result.json())
```

**So với local.** Dừng FastAPI bằng `search_process.terminate()` rồi `search_process.wait(timeout=30)`. Dừng vLLM bằng `stop_vllm(vllm_process)` để giải phóng GPU 1. Chạy lại bước 2 với `MCOT_EXTRACTOR_MODE = "local"`, rồi chạy bước 3 với cùng ảnh, prompt và gallery.

**Đo A/B song song và tuần tự với vLLM.** Sau lượt đo mặc định, dừng FastAPI bằng `search_process.terminate()` và `search_process.wait(timeout=30)`. Giữ vLLM đang chạy, đổi `MCOT_PARALLEL_BRANCHES` thành `"0"` trong bước 2 rồi chạy lại bước 2 và 3 với cùng ảnh, prompt và gallery. So sánh latency trung vị của client, `branches` và `total` giữa hai lượt. Cả hai lượt giữ `MCOT_EXTRACTOR_MODE="api"`.

**Gọi từ bên ngoài Kaggle.** Tạo Kaggle Secret `NGROK_AUTHTOKEN`, chạy cell sau rồi dùng `public_base` thay cho `search_base` ở client. Client vẫn lấy JWT và gọi `/api/list` theo bước 3. Cổng **8001** phục vụ kết nối nội bộ giữa FastAPI và vLLM.

```python
from pyngrok import ngrok

ngrok.set_auth_token(secrets_client.get_secret("NGROK_AUTHTOKEN"))
tunnel = ngrok.connect(8000)
public_base = tunnel.public_url
print(public_base)
```

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
