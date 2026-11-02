# Training-loop engineering notes — RAM and throughput fixes for the dual-stream trainer

This document explains the two problems hit while training the custom dual-stream
model on the RTX 5070 Laptop GPU, their root causes, the fixes applied in
`src/train_perceptual.py` and `configs/default.yaml`, and the measured effect of each.
It is written so the changes can be defended in a paper's reproducible-implementation
section. Nothing here changes the model architecture, the loss, or the reported
metrics — only *how* data reaches the GPU.

## Problem 1 — system-RAM exhaustion (training crash)

**Symptom.** First launch of the dual-stream training died during the first
validation pass: `RuntimeError: Caught MemoryError in DataLoader worker process 0 —
Unable to allocate 9.38 MiB` (a tiny allocation that only fails when the *system* RAM
is exhausted). This is a Windows host-memory failure, **not** a GPU failure.

**Root cause — 8 persistent worker processes.** The first configuration used
`num_workers=4` with `persistent_workers=True` on **both** the train and the
validation DataLoader. On Windows, each spawned worker carries its own full
Python + PyTorch + Ultralytics + OpenCV runtime (~1 GB+ before doing any work), so
the process tree held **8 heavy interpreters alive at the same time**
(4 train workers, idle but resident, + 4 validation workers spinning up). Together
with the main training process this exceeded the machine's budget.

**Second finding — workers ballooned to ~9 GB each.** Halving the workers was not
enough: with `num_workers=2`, system RAM still hit **99% (0.3 GB free)**. Process
inspection showed the two *train* workers at **8.62 GB and 9.85 GB RSS each**
(~18.5 GB combined). Per-item, the dataset produced a
`640×640×6 float32` tensor = **9.38 MB** (plus letterbox intermediates); over the
~15,000 items of one epoch, Windows heap growth/fragmentation let each worker's
resident set climb to a ~9 GB high-water mark without returning memory to the OS.

### Fix 1 — uint8 pipeline + worker lifecycle (measured effect)

Three changes in `src/train_perceptual.py` / `configs/default.yaml`:

1. **Keep images `uint8` through the dataloader; normalize on the GPU.**
   The dataset now returns a `(6, H, W) uint8` tensor (3.1 MB per item instead of
   9.4 MB); normalization moved to the training step:
   `imgs = batch["img"].to(dev, non_blocking=True).float().div_(255.0)`.
   The benchmark's dual-stream eval path was updated identically.
2. **`cv2.setNumThreads(0)`** in the dataset — OpenCV's per-thread buffer pools were
   compounding worker RSS.
3. **Worker lifecycle:** `persistent_workers=False` on the train loader (workers
   respawn each epoch, capping heap growth to one epoch) and **transient** validation
   workers — train and validation workers never coexist, so the 8-process peak is
   structurally impossible. `num_workers` restored to 4 once the uint8 pipeline
   removed the balloon.

**Measured effect:** worker RSS ~8.6–9.9 GB → **~3 GB each**; system RAM **99% used
(0.3 GB free) → stable ~65% (10.6–11.3 GB free)** across a full epoch including the
validation pass — the exact point where both earlier crashes occurred.

## Problem 2 — epoch time (fp32 vs mixed precision)

**Symptom.** With memory fixed, epoch 1 still took **12+ minutes** → ~35 h projected
for 150 epochs, even though GPU utilization looked healthy at 83–98% during bursts.

**Root cause.** The custom dual-stream loop ran the forward/loss in **full float32**,
while Ultralytics' own trainer (used for the single-stream baseline, ~75 s/epoch)
trains with mixed precision. On the same GPU the custom loop was paying 2–4× in
compute for fp32 arithmetic.

### Fix 2 — bfloat16 autocast (speedup projected, run not yet completed)

Forward and loss in both the train and validation loops are wrapped in:

```python
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
    loss, items = criterion(model(imgs), tgt)
```

bfloat16 was chosen over float16 because it needs **no `GradScaler`** (no loss-scale
bookkeeping to reproduce) and is natively supported by the RTX 5070's Blackwell
architecture (sm_120). Combined with the restored `num_workers=4`, the projected
epoch time is ~4–6 min (**~10–15 h for 150 epochs**). *This speedup is projected from
the fp32↔AMP gap and Ultralytics' measured behavior; the end-to-end dual-stream run
was paused for the paper-artifact work before a full epoch could be timed, so no
measured AMP epoch number is claimed here.*

## Related pitfall found the same day (for completeness)

Calling `model.predict()` with the **entire 3,264-image test list as one source** on
this 8 GB GPU attempted a single ~19.9 GiB allocation and raised CUDA OOM. The
per-image prediction loop (as used by `benchmark/run_benchmark.py` and
`scripts/reviewer_analysis.py`) is the reliable pattern on this hardware.

## Correctness guardrails applied with every change

- Every pipeline change was smoke-tested end-to-end (1-epoch training incl.
  validation, checkpoint save, resume, and the benchmark's dual-stream eval path).
- The uint8 change required the matching normalization in the benchmark eval —
  both call sites updated together.
- The full test suite (55 tests) passes after every change.
- No change alters model architecture, loss definition, augmentation, split logic, or
  any reported metric; training numerics move fp32 → bf16, the standard practice used
  by the Ultralytics baseline itself (keeping both trainers comparable).

## Where the changes live

| Change | Location |
|---|---|
| uint8 dataset output + `cv2.setNumThreads(0)` | `src/train_perceptual.py` → `DualStreamYoloDataset.__getitem__` |
| GPU-side normalization | `src/train_perceptual.py` → train/val loops; `benchmark/run_benchmark.py` → `_dual_detections`, latency probe |
| worker counts / persistence | `configs/default.yaml` → `perceptual_training.num_workers`; DataLoader kwargs in `train_dual` |
| bf16 autocast | `src/train_perceptual.py` → train and validation forward/loss |
