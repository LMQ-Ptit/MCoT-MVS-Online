"""FastAPI backend for the FashionIQ dress gallery on Kaggle."""

import hmac
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.error import URLError
from urllib.request import Request, urlopen

import jwt
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from PIL import Image

from vllm_extractor import MCoTExtractor

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_GALLERY = "/kaggle/input/datasets/thichuy/mcot-vectors/gallery_dress_vectors.pt"
TOKEN_TTL = 3600
JWT_ALG = "HS256"
gpu_lock = Lock()
logger = logging.getLogger(__name__)


def load_mcot():
    mode = os.environ.get("MCOT_EXTRACTOR_MODE", "api").lower()
    model_name = os.environ.get("MCOT_MODEL_NAME", DEFAULT_MODEL)
    if mode == "local":
        return MCoTExtractor(mode="local", model_name=model_name, device="cuda:1")
    if mode != "api":
        raise RuntimeError("MCOT_EXTRACTOR_MODE must be api or local.")

    api_base = os.environ.get("MCOT_VLLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
    api_key = os.environ.get("MCOT_VLLM_API_KEY", "EMPTY")
    request = Request(f"{api_base}/models", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(request, timeout=5) as response:
            models = json.load(response)["data"]
    except (URLError, TimeoutError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Cannot read models at {api_base}. Start vLLM before FastAPI.") from exc
    if not any(model["id"] == model_name for model in models):
        raise RuntimeError(f"The API at {api_base} does not serve {model_name}.")
    return MCoTExtractor(mode="api", model_name=model_name, device="cpu", api_base=api_base, api_key=api_key)


def load_cir(checkpoint_path):
    from model import CIRModel

    with torch.cuda.device(1):
        cir = CIRModel()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        weights = checkpoint.get("CIRModel") or checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
        cir.load_state_dict(weights, strict=False)
        return cir.to("cuda:1").eval()


@asynccontextmanager
async def lifespan(app):
    app.state.api_key = os.environ.get("MCOT_SEARCH_API_KEY") or os.environ.get("API_KEY")
    app.state.jwt_secret = os.environ.get("MCOT_SEARCH_JWT_SECRET") or os.environ.get("JWT_SECRET")
    if not app.state.api_key or not app.state.jwt_secret:
        raise RuntimeError("Set MCOT_SEARCH_API_KEY and MCOT_SEARCH_JWT_SECRET before starting FastAPI.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This backend uses two CUDA GPUs. Select GPU T4 x2 on Kaggle.")
    parallel_flag = os.environ.get("MCOT_PARALLEL_BRANCHES", "1")
    if parallel_flag not in ("0", "1"):
        raise RuntimeError("MCOT_PARALLEL_BRANCHES must be 0 or 1.")

    gallery_path = Path(os.environ.get("MCOT_GALLERY_PATH", DEFAULT_GALLERY))
    checkpoint_path = Path(os.environ.get("MCOT_CIR_CHECKPOINT", REPO_DIR / "src/checkpoints/dress.pt"))
    for path in (gallery_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required backend artifact is missing: {path}")

    app.state.mcot = load_mcot()
    sys.path.insert(0, str(REPO_DIR / "segment"))
    from get_masks import load_segment_pipeline

    with torch.cuda.device(0):
        app.state.seg = load_segment_pipeline(device="cuda:0", load_clip=True)
    app.state.cir = load_cir(checkpoint_path)
    gallery = torch.load(gallery_path, map_location="cpu")
    app.state.gallery_features = gallery["features"].float()
    app.state.gallery_ids = gallery["ids"]
    if len(app.state.gallery_ids) == 0 or len(app.state.gallery_features) != len(app.state.gallery_ids):
        raise RuntimeError("Gallery features and IDs must have the same nonzero length.")
    use_parallel = app.state.mcot.mode == "api" and parallel_flag == "1"
    executor_context = ThreadPoolExecutor(max_workers=1) if use_parallel else nullcontext()
    with executor_context as executor:
        app.state.mcot_executor = executor
        branch_mode = "parallel" if use_parallel else "sequential"
        print(
            f"READY: models + gallery loaded (MCoT mode={app.state.mcot.mode}, branches={branch_mode})",
            flush=True,
        )
        yield


app = FastAPI(title="MCoT Search API", lifespan=lifespan)


def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    try:
        jwt.decode(authorization[7:], app.state.jwt_secret, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


@app.post("/api/token")
def get_token(x_api_key: str = Header(None)):
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), app.state.api_key.encode()):
        raise HTTPException(401, "Invalid API key")
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {"sub": "manh", "iat": now, "exp": now + timedelta(seconds=TOKEN_TTL)},
        app.state.jwt_secret,
        algorithm=JWT_ALG,
    )
    return {"access_token": token, "token_type": "bearer", "expires_in": TOKEN_TTL}


def run_mcot(extractor, image, text):
    started = perf_counter()
    try:
        result = extractor.extract(image, text)
    except Exception as exc:
        if extractor.mode != "api":
            raise
        logger.exception("MCoT inference endpoint failed")
        raise HTTPException(502, "MCoT inference endpoint failed") from exc
    return result, (perf_counter() - started) * 1000


@app.post("/api/list", dependencies=[Depends(verify_token)])
def list_results(
    response: Response,
    text: str = Form(...),
    image: UploadFile = File(...),
    top_k: int = Form(10),
):
    started = perf_counter()
    try:
        with Image.open(image.file) as source:
            pil = source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise HTTPException(400, "Invalid image file") from exc

    lock_started = perf_counter()
    with gpu_lock, torch.inference_mode():
        lock_ms = (perf_counter() - lock_started) * 1000
        branches_started = perf_counter()
        executor = app.state.mcot_executor
        future = None
        if executor is not None:
            # PIL encoders can update image metadata while saving.
            future = executor.submit(run_mcot, app.state.mcot, pil.copy(), text)
        try:
            segment_started = perf_counter()
            with tempfile.TemporaryDirectory() as temp, torch.cuda.device(0):
                path = Path(temp) / "query.png"
                pil.save(path)
                seg = app.state.seg.process_image(
                    str(path), save_dir=temp, is_draw=False, compute_clip_features=True
                )
            segment_ms = (perf_counter() - segment_started) * 1000
            if future is None:
                with torch.cuda.device(1):
                    mcot, mcot_ms = run_mcot(app.state.mcot, pil, text)
            else:
                mcot, mcot_ms = future.result()
        finally:
            # Keep the lock until an already-running vLLM call has finished, even if Segment failed.
            if future is not None and not future.done() and not future.cancel():
                wait((future,))
        branches_ms = (perf_counter() - branches_started) * 1000

        cir_started = perf_counter()
        with torch.cuda.device(1):
            seg_features = seg["seg_features"].to("cuda:1")
            query = app.state.cir.extract_query(
                modified_text=[text],
                reference_image=[pil],
                reference_seg_feature_list=[seg_features],
                llm_info_dict={field: [mcot[field]] for field in ("target", "retained", "deleted")},
            )
            query = query.float().cpu()
        cir_ms = (perf_counter() - cir_started) * 1000

    rank_started = perf_counter()
    ids = app.state.gallery_ids
    top_k = max(1, min(top_k, 50, len(ids)))
    scores, indices = torch.topk((query @ app.state.gallery_features.T).squeeze(0), top_k)
    results = [
        {"rank": rank, "image_id": str(ids[int(index)]), "score": round(float(score), 4)}
        for rank, (score, index) in enumerate(zip(scores, indices), start=1)
    ]
    finished = perf_counter()
    rank_ms = (finished - rank_started) * 1000
    total_ms = (finished - started) * 1000
    response.headers["Server-Timing"] = (
        f"lock_wait;dur={lock_ms:.2f}, segment;dur={segment_ms:.2f}, mcot;dur={mcot_ms:.2f}, "
        f"branches;dur={branches_ms:.2f}, cir;dur={cir_ms:.2f}, "
        f"rank;dur={rank_ms:.2f}, total;dur={total_ms:.2f}"
    )
    return {"query_text": text, "results": results}
