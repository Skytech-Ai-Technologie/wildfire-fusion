# Draft figure captions

**Figure 1 — Architecture (`architecture.png/.svg/.pdf`).** Implemented FORFIRE-AI
pipeline: RGB/LWIR cameras and IIoT nodes feed a dual-stream YOLOv8n+CBAM perceptual
tier and an IIoT isolation-forest branch; a 4-layer entropy-gated cross-modal
transformer refines per-modality confidences over 32-frame windows; a conflict-aware
Dempster–Shafer rule raises an alarm at threshold τ, published over MQTT within an
8 kB budget. The dashed lane marks training-only artifacts. On D-Fire the LWIR stream
uses a grayscale proxy.

**Figure 2 — Training curves (`training_curves.png`).** Box/classification/DFL losses
and validation precision, recall, mAP@0.5 and mAP@0.5:0.95 over 150 epochs on the
leakage-safe split. mAP@0.5 plateaus near 0.72 after ~epoch 60; `best.pt` is selected
on validation mAP by the framework.

**Figure 3 — Confusion matrix, validation (`confusion_matrix_validation.png` and
normalized variant).** Final-training-validation confusion matrix. Smoke is recovered
77% of the time and fire 68%; class-to-class confusion is negligible (<1%), misses
resolve to background.

**Figure 4 — Confusion matrix, test (`confusion_matrix_TEST.png` and normalized
variant).** Held-out test-split confusion matrix produced by the benchmark harness.

**Figure 5 — Precision–Recall curve (`precision_recall_curve.png`).** Per-class PR
curves at IoU 0.5 on the test split; smoke AP50 0.777, fire AP50 0.650.

**Figure 6 — F1 / precision / recall curves (`f1_curve.png`, `precision_curve.png`,
`recall_curve.png`).** Confidence-threshold sweeps on the test split; useful for
choosing an operating point for early warning (recall-favoring).

**Figure 7 — Sample predictions (`sample_batch_predictions.png` vs
`sample_batch_ground_truth.png`).** A validation batch with ground truth (left) and
model predictions (right) for qualitative inspection.

**Table I (`results_table.tex`, `table_iv_benchmark.csv`).** Detection and image-level
performance on the D-Fire test split with 95% bootstrap CIs and latency.

**Table II (`table_v_fieldsim_SYNTHETIC.csv`).** Field-simulation replay of the fusion
configurations (camera-only / +LWIR / +IIoT / IIoT-only). **Synthetic software
validation, not a field trial.**

**Table III (`table_vi_ablation_SYNTHETIC.csv`).** Ablation deltas with bootstrap CIs
from the same synthetic replay.
