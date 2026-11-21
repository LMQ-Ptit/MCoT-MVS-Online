"""Install the patched AutoAWQ runtime into the current Kaggle environment."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve


AUTOAWQ_VERSION = "0.2.7"
SOURCE_URL = f"https://codeload.github.com/casper-hansen/AutoAWQ/tar.gz/refs/tags/v{AUTOAWQ_VERSION}"
SOURCE_SHA256 = "5ddb924bb0ef92d5b64990af90ae336571e9593bdb596c330685133412714a4c"


def replace_once(path: Path, original: str, replacement: str) -> None:
    content = path.read_text(encoding="utf-8")
    if content.count(original) != 1:
        raise RuntimeError(f"Unexpected AutoAWQ source in {path}: {original!r}")
    path.write_text(content.replace(original, replacement, 1), encoding="utf-8")


def patch_source(source: Path) -> None:
    # Text-based calibration loads datasets on demand. Loading AWQ weights does not need it.
    setup = source / "setup.py"
    replace_once(setup, '    "datasets>=2.20",\n', "")
    replace_once(
        setup,
        '    extras_require={\n',
        '    extras_require={\n        "calibration": ["datasets>=2.20"],\n',
    )
    calibration = source / "awq/utils/calib_data.py"
    replace_once(calibration, "from datasets import load_dataset\n", "")
    replace_once(
        calibration,
        "    if isinstance(data, str):\n",
        "    if isinstance(data, str):\n        from datasets import load_dataset\n\n",
    )

    # Triton 2.3 (bundled with Torch 2.3) lacks the interleave used by these kernels.
    gemm = source / "awq/modules/linear/gemm.py"
    replace_once(
        gemm,
        "try:\n    from awq.modules.triton.gemm",
        "TRITON_AVAILABLE = False\ntry:\n"
        "    from triton.language import interleave\n"
        "    from awq.modules.triton.gemm",
    )
    replace_once(
        gemm,
        "            if not user_has_been_warned:",
        "            global user_has_been_warned\n            if not user_has_been_warned:",
    )


def smoke_test() -> None:
    import torch
    from awq.modules.linear.gemm import TRITON_AVAILABLE, WQLinear_GEMM, awq_ext

    device = torch.device("cuda", torch.cuda.current_device())
    layer = WQLinear_GEMM(
        w_bit=4, group_size=128, in_features=128, out_features=128, bias=False, dev=device
    ).eval()
    backend = "awq_ext" if awq_ext is not None else "Triton" if TRITON_AVAILABLE else "PyTorch"
    with torch.inference_mode():
        layer.qweight.fill_(0x11111111)
        layer.qzeros.zero_()
        layer.scales.fill_(1 / 128)
        for tokens in (1, 8):
            inputs = torch.ones((1, tokens, 128), dtype=torch.float16, device=device)
            torch.testing.assert_close(layer(inputs), inputs, rtol=0, atol=0)
    torch.cuda.synchronize(device)
    print(f"AutoAWQ {AUTOAWQ_VERSION} CUDA GEMM passed on {torch.cuda.get_device_name(device)} ({backend}).")


def main() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5):
        raise SystemExit("AutoAWQ requires a CUDA device with compute capability 7.5 or higher.")

    pip = [sys.executable, "-m", "pip"]
    installed = json.loads(subprocess.check_output([*pip, "list", "--format=json"], text=True))
    build_env = {**os.environ, "INSTALL_KERNELS": "0", "TORCH_VERSION": torch.__version__}
    with TemporaryDirectory(prefix="mcot-awq-") as directory:
        work = Path(directory)
        archive = work / "autoawq.tar.gz"
        urlretrieve(SOURCE_URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise RuntimeError("AutoAWQ source checksum mismatch.")
        shutil.unpack_archive(str(archive), str(work))
        source = work / f"AutoAWQ-{AUTOAWQ_VERSION}"
        patch_source(source)

        constraints = work / "constraints.txt"
        constraints.write_text(
            "".join(
                f"{item['name']}=={item['version']}\n"
                for item in installed
                if item["name"].lower() != "autoawq"
            ),
            encoding="utf-8",
        )
        install = [*pip, "install", "--constraint", str(constraints)]
        subprocess.run([*install, "setuptools", "wheel"], check=True)
        # Replace an unpatched wheel of the same version without forcing its dependencies to reinstall.
        subprocess.run(
            [*install, "--no-build-isolation", "--no-deps", "--force-reinstall", str(source)],
            check=True,
            env=build_env,
        )
        subprocess.run([*install, f"autoawq=={AUTOAWQ_VERSION}"], check=True)

    # Use a fresh interpreter so it sees the newly installed package and wheel metadata.
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--smoke-test"],
        check=True,
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke-test"]:
        smoke_test()
    else:
        main()
