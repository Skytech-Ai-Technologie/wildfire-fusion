# Contribution Statement — Multimodal Wildfire Early-Warning System (wildfire-fusion)

This document describes the project as a whole and states its scientific contributions:
what is faithfully reproduced from the source publication, what was incomplete in the
source and is completed here, and what methodological and engineering contributions this
work adds on top. Companion documents: [README.md](README.md) (usage),
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) (pre-reproduction gates),
[docs/index.html](docs/index.html) (interactive explainer with a live fusion playground).

---

## 1. Problem and context

Wildfire early warning needs three things at once: **detect** (smoke or flame in imagery),
**contextualize** (separate a weak but real signal from sensor noise over time), and
**decide** (raise an alarm with a controlled false-alarm rate, fusing heterogeneous
sensors that sometimes disagree). Single-modality detectors fail on exactly the cases
that matter most: small early-stage smoke plumes, night-time visibility, and cameras
with degraded views. The source publication proposed a three-tier, four-layer fusion
architecture for this problem and reported results on FLAME, D-Fire and FIgLib; it also
left a number of components under-specified (training objectives, fusion depth,
operational policies, evaluation protocol). This work is a complete, verified,
open re-implementation of that architecture, plus the methodological and engineering
contributions listed in §3.

## 2. System overview

```
Sensing        PTZ RGB+LWIR camera · on-demand UAV RGB · IIoT environmental nodes
               (temperature / humidity / particulate / gas)
Perceptual     dual-stream YOLOv8n (RGB + LWIR) with CBAM attention · IIoT
               isolation-forest anomaly score
Contextual     4-layer entropy-gated cross-modal transformer refining per-modality
               confidences over 32-frame windows
Decisional     conflict-aware Dempster–Shafer fusion (eq. 4–5) with calibrated
               reliabilities and alarm threshold τ
Edge/ops       ONNX → TensorRT INT8 (Jetson) · MQTT/TLS with ≤ 8 kB alarm payloads ·
               supervisory loop (Algorithm 1) with hysteresis, latching, cooldown
```

Data flow: images and IIoT telemetry → leakage-safe video-level splits → perceptual
detector confidences + embeddings → 32-frame event windows → contextual refinement →
Dempster–Shafer mass functions → fused belief → alarm decision → MQTT event +
supervisory queue (with UAV re-tasking suggestion).

Benchmarks: **D-Fire** (ground RGB, detection), **FLAME 3 CV subset** (UAV RGB+thermal,
classification), **FIgLib** (fixed-view PTZ, sequences).

## 3. Scientific contributions

### C1 — Faithful, fully-specified reproduction of the three-tier fusion architecture

Every component of the source architecture is implemented end-to-end and runnable:
dual-stream YOLOv8n + CBAM with the eq. (2) weighted loss; the entropy-gated
cross-modal transformer (eq. 3); conflict-aware Dempster–Shafer fusion with Jousselme
divergence (eq. 4–5); and the supervisory loop (Algorithm 1). Where the paper is
silent, every gap is filled with an **explicitly labeled assumption** (assumption
register, `configs/default.yaml` and inline `ASSUMPTION` tags) — e.g. the contextual
training objective (window-level BCE), the dual-stream fusion depth
(channel-concat + 1×1 conv after stage-2), the mapping of eq. (2) onto the Ultralytics
task-aligned assignment (CIoU+DFL ← λ1, focal/BCE ← λ2, objectness ← λ3), and the
reliability fallbacks. Nothing is silently invented; every assumption is swappable.

### C2 — Leakage-safe video-level splitting, with a demonstrated flaw in frame-level splitting

The paper flags that "random frame-level splitting can overestimate generalization"
but does not operationalize it. This work makes the guarantee concrete:

- Groups are split at the event level (FLAME: burn/flight ID; FIgLib: official
  sequence ID; D-Fire: numeric-stem bucketing, see below), stratified 70/15/15 by
  label rank, with a **hard zero-overlap assertion** at split time and in the test
  suite.
- For D-Fire — whose public release ships no source-video IDs — we show that the
  **official mirror partition itself leaks**: consecutive video frames land in
  different splits (e.g. `AoF00004` in train, `AoF00008` in val). The fix introduced
  here buckets the stem number line into 32-frame blocks so near-duplicate frames
  always share a group (673 groups over 21,527 images; verified zero straddling).
- The contextual tier is trained strictly on the manifest's train split, with val for
  early stopping and test held out entirely (a naive random window split was shown to
  leak test events through the encoder).

### C3 — Evaluation-protocol upgrades over the published evaluation

The reported evaluation is strengthened in four ways the source lacks:

1. **Bootstrap confidence intervals** (≥ 1,000 resamples) on mAP, image-level metrics
   and every reported delta (Tables IV–VI).
2. An **explicitly defined and logged FPR denominator** (false alarms / monitored
   no-fire windows).
3. **Image-level P/R/F1 and per-class confusion matrices**, absent from the paper.
4. **Raw per-call latency traces** saved alongside summaries, per the
   reproducibility checklist.

A custom VOC-style mAP implementation (`map_from_detections`) extends mAP@.5 to the
dual-stream checkpoints, which cannot pass through the stock Ultralytics validator.

### C4 — Dataset provenance and automated integrity verification

All three benchmarks are fetched by one script with per-dataset provenance markers
(timestamp, source, file count). D-Fire integrity is verified **automatically against
the published instance counts** (smoke 11,865 / fire 14,692 — exact match in this
checkout, confirming the canonical 0 = smoke, 1 = fire mapping that is commonly
reported both ways across forks), and `harmonize_labels.py` fails loudly on any class
id outside the mapping instead of silently training on wrong labels.

### C5 — Operational policies for the supervisory loop, formalized and unit-tested

The paper leaves deployment policies unspecified. This work defines, implements and
tests them: timestamp-sync tolerance (±2 s), packet-loss handling (modality marked
inactive after 2 missed heartbeats), UAV-availability policy, sensor-dropout hook,
alarm hysteresis (2 consecutive ticks), latching until human clear, cooldown before
re-arming — including the corner case that an alarm burst **during cooldown must not
re-latch** the loop (otherwise a single post-cooldown tick re-confirms an alarm,
defeating the hysteresis).

### C6 — Calibrated decisional tier with safety guards

Reliabilities r_c, r_d, r_s and the threshold τ are grid-searched on labeled
validation ticks under a recall constraint, and the calibrated file is auto-loaded by
the fusion loop. The calibrator **refuses to write** a configuration derived from
unlabeled or degenerate data (e.g. timelines without per-tick labels), because such a
file would silently override the paper's τ in every subsequent run.

### C7 — Edge deployment path and payload budget

The full inference stack targets Jetson-class hardware: ONNX export for both the
perceptual and contextual tiers, TensorRT INT8 quantization tooling (5,000-image
calibration list; engine built on-device), MQTT/TLS publishing with a hard **8 kB
payload budget** (drop-not-truncate policy), and a live end-to-end field simulation
replaying Algorithm 1 over full event timelines.

### C8 — Verified open implementation

- **55 automated tests** covering: the leakage assertion, Dempster–Shafer math
  (identity, reinforcement, total-conflict fallback), entropy gates, IIoT alignment,
  loop policies, payload budget, split heuristics, and the detection-based mAP.
- An **interactive single-file explainer** (`docs/index.html`, works offline) with a
  clickable architecture diagram, a step-through of Algorithm 1, a live
  Dempster–Shafer playground using the same math as the implementation, and an
  entropy-gate heatmap.
- All bugs found during verification were fixed and pinned by regression tests.

## 4. Empirical status (as of this writing)

| Item | Result |
|---|---|
| D-Fire integrity | 21,527 images (14,122/3,099/4,306 train/val/test); label histogram exactly matches published counts |
| FIgLib | 732 images across all 10 official split sequences (per-sequence counts match the CDN index) |
| FLAME 3 CV subset | 2,214 RGB+thermal images (fire 1,866 / no-fire 348) |
| Single-stream YOLOv8n (D-Fire, in progress) | val mAP50 0.733 @ epoch 49/150 (RTX 5070 Laptop, ~75 s/epoch) |
| Dual-stream smoke (2 epochs) | train loss 369 → 36, val 549 → 142; full pipeline validated end-to-end |

These are interim engineering numbers from an in-progress reproduction, not final
claims; the pre-reproduction gates in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) define
what must hold before results are compared against the publication.

## 5. Limitations (carried over and documented, not hidden)

- The dual-stream LWIR input on RGB-only datasets is a **grayscale proxy**
  (documented placeholder); a real two-modal run requires the FLAME 2 registered
  RGB/IR pairs.
- The FLAME data currently present is the no-login Kaggle CV subset (single burn);
  the full multi-burn FLAME sets require an IEEE DataPort license.
- Detection-horizon results are reported over N = 3 simulated events with bootstrap
  CIs — high variance is expected, and the field simulation is synthetic by design.
- Baseline parity (same splits/augmentation) holds only for baselines actually
  trained in this repository.
- The D-Fire event grouping is a documented approximation (numeric-stem bucketing),
  not a true source-video ID, which the public release does not provide.

## 6. Reproducibility

Python 3.11, PyTorch 2.14.1+cu130, Ultralytics ≥ 8.2, single consumer GPU
(RTX 5070 Laptop, 8 GB). Every artifact in the pipeline is regenerated by one command
chain from `README.md` §1–2 (unit tests, dataset fetch, splits, training, benchmark,
calibration, ablations). Interim checkpoints and all training hyperparameters are
logged per run under `runs/<experiment>/train_config.yaml`; raw metric traces are
written under `results/`.
