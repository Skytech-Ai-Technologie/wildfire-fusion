# postdo — paper-ready artifacts (real measured results)

Everything in this folder comes from **actual runs** in this repository. Nothing is
estimated or hand-written. Synthetic items are clearly labeled; unmeasured items are
listed as *not measured*, never estimated.

## Headline result (D-Fire test split — 3,264 images, 3,990 instances)

YOLOv8n single-stream RGB, 150 epochs, leakage-safe 32-frame group split:

| Metric | Box-level | Image-level |
|---|---|---|
| Precision | 0.739 | 0.971 |
| Recall | 0.664 | 0.945 |
| F1 | 0.700 | 0.958 |
| mAP@0.5 | **0.719** | — |
| mAP@0.5:0.95 | 0.385 | — |
| Accuracy (95% bootstrap CI) | — | 0.955 [0.948, 0.962] |

Latency **11.7 ms mean / 17.1 ms p95** per image (RTX 5070 Laptop, FP32, 640 px).
Contextual tier (entropy-gated transformer on 15,063 real embeddings, 32-frame
leakage-safe windows): best val accuracy **0.936**.

## File index

| File | What it is |
|---|---|
| `metrics.json` | every number in one machine-readable file (training config, test metrics, per-class, image-level, CI, latency, contextual tier, hardware) |
| `results_table.tex` | LaTeX table for the paper (generated from `metrics.json` values) |
| `training_curves.png` | loss + P/R/mAP per epoch (150 epochs) |
| `training_history.csv` | the same data as logged per epoch |
| `confusion_matrix_validation.png` / `_normalized_` | val-split confusion matrix from the final training validation |
| `confusion_matrix_TEST.png` / `_normalized_` | **test-split** confusion matrix (from the benchmark run) |
| `precision_recall_curve.png`, `f1_curve.png`, `precision_curve.png`, `recall_curve.png` | per-class PR/F1/P/R curves |
| `per_class_metrics.csv`, `classification_report.txt`, `per_class_metrics.json` | per-class box-level metrics on the test split |
| `sample_batch_ground_truth.png` / `sample_batch_predictions.png` | visual sanity pairs (labels vs model predictions) |
| `table_iv_benchmark.csv` | the full benchmark row (source of the headline table) |
| `latency_raw_traces.json` | raw per-call latency traces (reproducibility checklist) |
| `table_v_fieldsim_SYNTHETIC.csv` | fusion-config comparison — **SYNTHETIC field replay, not a field trial** |
| `table_vi_ablation_SYNTHETIC.csv` | ablation deltas — **SYNTHETIC field replay** |
| `contextual_training_history.json` / `_config` | contextual-tier training log (real embeddings) |
| `architecture.png/.svg/.pdf` | implemented-pipeline diagram (training lane dashed) |
| `dataset_audit.json`, `dataset_summary.txt` | verified dataset counts, splits, leakage control, provenance |
| `environment.txt` | exact software/hardware versions |
| `figure_captions.md` | draft captions for every figure |
| `reproducibility.md` | the exact commands that produced all of this |
| `limitations.md` | what was NOT measured and the caveats a reviewer will ask about |
| `training_optimization_notes.md` | the RAM/throughput engineering story: 8 persistent workers → uint8 pipeline + bf16 autocast, with measured effects |
| `reviewer_response.md` | point-by-point answers to the five reviewer requests |
| `threshold_analysis.csv`, `threshold_tradeoff.png` | 18-point confidence sweep: fire/smoke P-R vs image-level false alarms |
| `operational_metrics.json` | latency, recall at FP budgets (0.5/1/2%), FPR-per-hour disclosure |
| `false_positives/` + `false_positives_index.csv` | all 50 background scenes detected as fire/smoke, annotated and ranked by confidence |
| `leakage_verification.json` | recomputed group-level split verification (673 groups, zero straddling) |
| `raw_test_predictions.csv` | every test-split detection — all metrics above regenerate from it |

## Integrity map (what is genuine vs synthetic vs not run)

- **Genuine, measured on the held-out test split:** detection metrics, per-class
  metrics, image-level P/R/F1 + CI + confusion matrix, latency (FP32, RTX 5070),
  training curves, contextual-tier training on real embeddings.
- **Synthetic (labeled):** Tables V and VI — field-simulation replays of Algorithm 1,
  the same procedure the source paper uses; they are software-validation results, not
  field-trial evidence.
- **Not measured (no estimate given):** Jetson Orin Nano INT8 latency; dual-stream
  full training; registered RGB/LWIR two-modal results (FLAME 2 pairs not licensed).
