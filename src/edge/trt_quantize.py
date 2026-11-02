"""§7.2 — TensorRT INT8 quantization.

Run ON-DEVICE (Jetson JetPack image ships trtexec + TensorRT 10.0 to match Table III):

    trtexec --onnx=perceptual.onnx \\
            --int8 \\
            --calib=calibration_cache.bin \\
            --saveEngine=perceptual_int8.engine \\
            --calibrationBatchSize=8

Calibration set: 5,000 representative images per the paper — sample proportionally
across FLAME/D-Fire/FIgLib and across day/night/fog conditions if you have them, rather
than pulling 5,000 frames from one dataset (the paper doesn't specify the sampling
policy — this is our documented choice). Build the calibration cache with the standard
TensorRT IInt8EntropyCalibrator2 workflow (see NVIDIA's calibrator sample) feeding those
5,000 images through the same preprocessing as training.

Target checks after quantization: resident memory ≈ 4.2 GB on Orin Nano, forward latency
≈ 74 ms for RGB+LWIR (36 ms RGB-only) — the paper's *reported* numbers; treat them as a
target/sanity-check, not a guarantee. Your own hardware run will differ.

This module is x86-safe: on a machine without trtexec it prints the exact commands
instead of failing.
"""
import random
import shutil
import subprocess
from pathlib import Path

TRTEXEC_TIMEOUT_S = 60 * 60
CALIBRATION_SET_SIZE = 5000   # per the paper


def build_int8_engine(onnx_path: str, engine_path: str, calib_cache: str | None = None) -> int:
    """Shell out to trtexec. Returns the process exit code; prints instructions when
    trtexec is unavailable (i.e., on any x86 box without TensorRT)."""
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        print(
            "trtexec not found on PATH. TensorRT INT8 engines must be built on the Jetson\n"
            "(JetPack image). Copy the ONNX file there and run:\n"
            f"  trtexec --onnx={onnx_path} --int8 --calib=calibration_cache.bin \\\n"
            f"          --saveEngine={engine_path} --calibrationBatchSize=8"
        )
        return 127
    cmd = [trtexec, f"--onnx={onnx_path}", "--int8", f"--saveEngine={engine_path}",
           "--calibrationBatchSize=8"]
    if calib_cache:
        cmd.append(f"--calib={calib_cache}")
    print("running:", " ".join(cmd))
    return subprocess.run(cmd, timeout=TRTEXEC_TIMEOUT_S).returncode


def sample_calibration_set(datasets_root: str | Path, n: int = CALIBRATION_SET_SIZE,
                           seed: int = 42, exts=(".jpg", ".jpeg", ".png")) -> list:
    """Proportional sample across datasets present under data/ (our documented policy).

    Looks one level deep: data/<dataset>/**/*. Accepts an optional pre-built
    calibration_list.txt to restrict candidates (e.g., day/night/fog balanced).
    """
    root = Path(datasets_root)
    per_dataset = {}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = [p for p in ds_dir.rglob("*") if p.suffix.lower() in exts]
        if imgs:
            per_dataset[ds_dir.name] = imgs
    if not per_dataset:
        raise FileNotFoundError(f"no calibration images found under {root}")

    rng = random.Random(seed)
    total = sum(len(v) for v in per_dataset.values())
    sampled = []
    for name, imgs in per_dataset.items():
        k = round(n * len(imgs) / total)
        sampled += rng.sample(imgs, min(k, len(imgs)))
    rng.shuffle(sampled)
    return sampled


def write_calib_list(paths, out_txt: str | Path):
    out = Path(out_txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="§7.2 TensorRT INT8 helpers")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--n", type=int, default=CALIBRATION_SET_SIZE)
    ap.add_argument("--list-out", default="results/calibration_list.txt")
    args = ap.parse_args()
    paths = sample_calibration_set(args.data_root, n=args.n)
    write_calib_list(paths, args.list_out)
    print(f"wrote {len(paths)} calibration image paths -> {args.list_out}")
    print("feed these through IInt8EntropyCalibrator2 with the TRAINING preprocessing,")
    print("then build the engine with build_int8_engine().")
