"""Image input and wall-clock timing for the Kaggle inference examples."""

from io import BytesIO
from statistics import median
from time import perf_counter
from urllib.request import urlopen

from PIL import Image, ImageOps


DEFAULT_TEST_IMAGE_URL = "https://collectionapi.metmuseum.org/api/collection/v1/iiif/159159/331375/main-image"


def load_test_image(image_url=DEFAULT_TEST_IMAGE_URL):
    """Download a test image and return it as RGB with a maximum side of 512 px."""
    with urlopen(image_url, timeout=30) as response:
        with Image.open(BytesIO(response.read())) as source:
            img = ImageOps.exif_transpose(source).convert("RGB")
    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    return img


def measure_inference(infer, *, repeats=3, synchronize=None):
    """Measure a warmed-up callable and return its last result."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if synchronize is not None:
        synchronize()

    latencies = []
    for _ in range(repeats):
        started = perf_counter()
        result = infer()
        if synchronize is not None:
            synchronize()
        latencies.append(perf_counter() - started)

    print("Inference sau warmup: " + ", ".join(f"{elapsed:.2f}s" for elapsed in latencies))
    print(f"Latency trung vị: {median(latencies):.2f}s")
    return result
