"""Wall-clock timing for the Kaggle inference examples."""

from statistics import median
from time import perf_counter


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
