# Limitations — what a reviewer will ask about

## Not measured (no estimates given anywhere)
- **Jetson Orin Nano / INT8 latency.** No Jetson device was available. The reported
  latency (11.7 ms mean, FP32, RTX 5070 Laptop GPU, 640 px) is a desktop-GPU
  measurement and must not be presented as edge-hardware performance. The ONNX export
  and TensorRT INT8 calibration-list tooling exist in `src/edge/`, but no engine was
  built or timed.
- **Dual-stream (RGB+LWIR) full training.** The dual-stream architecture trains
  end-to-end (validated by smoke runs) but was not fully trained within this working
  session; the trained model in this folder is the single-stream RGB detector.
- **Registered two-modal training.** On D-Fire the LWIR stream is a **grayscale
  proxy** of the RGB frame (documented placeholder). Real two-modal results require
  FLAME 2's registered RGB/IR pairs (IEEE DataPort license).

## Dataset caveats
- **D-Fire source-video IDs are not published.** Splits use 32-frame numeric-stem
  bucketing (673 groups, zero straddling verified). This is a documented approximation,
  not a true video-level partition; the official mirror partition itself leaks
  consecutive frames across splits.
- **D-Fire domain mix.** WEB/AoF/PublicDataset frames mix web images, night flare
  footage and public datasets — the model's fire recall (0.57–0.68 box-level) partly
  reflects night/flare ambiguity, not only model capacity.
- **FLAME subset is a single burn** (Sycan Marsh Kaggle subset); the full multi-burn
  FLAME 3 / FLAME 2 sets were not licensed.
- **FIgLib contains only fire sequences**; negatives follow the official CV splits and
  are kept sequence-level.

## Protocol caveats
- **Tables V and VI are synthetic replays** of the supervisory loop (Algorithm 1) over
  simulated confidence timelines — the same procedure as the source paper. They
  validate the *software and policies*, not field performance. N = 3 events; high
  variance is expected and bootstrap CIs are reported.
- **Contextual-tier operating point.** The contextual transformer is trained on
  windows derived from the perceptual model's embeddings of the *train* split
  (leakage-safe); its val accuracy (0.936) is a window-classification number, not an
  end-to-end alarm-rate measurement.
- **Calibration.** The decisional reliabilities/τ remain the paper defaults plus the
  validated synthetic calibration; recalibration on labeled real validation ticks is
  required before deployment claims.
- Full determinism is not guaranteed across GPUs/drivers; seeds are fixed (42) and
  recorded.
