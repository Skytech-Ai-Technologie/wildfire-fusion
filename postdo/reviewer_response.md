# Response to reviewer requests — artifacts and measured answers

All numbers below are measured on the held-out D-Fire test split (3,264 images,
3,990 instances, 1,504 background scenes) unless stated otherwise.

## R1 — Precision–recall curves and per-class AP ✅ (already in postdo)

- `precision_recall_curve.png` — per-class PR curves at IoU 0.5.
- `per_class_metrics.csv` / `per_class_metrics.json` — **smoke: AP50 0.777, AP50-95
  0.453, P 0.786, R 0.720, F1 0.752 · fire: AP50 0.650, AP50-95 0.326, P 0.699,
  R 0.572, F1 0.629.**
- All classes: **mAP@0.5 = 0.719, mAP@0.5:0.95 = 0.385** (`metrics.json`,
  `results_table.tex`).

## R2 — Confidence-threshold analysis ✅ (`threshold_analysis.csv`, `threshold_tradeoff.png`)

18 operating points swept on the test split. Fire false alarms **can** be cut
substantially, at a quantified recall cost:

| conf τ | fire P | fire R | smoke R | background scenes w/ fire alarm (of 1,504) |
|---|---|---|---|---|
| 0.15 | 0.511 | 0.698 | 0.781 | 27 (1.8%) |
| 0.25 (default) | 0.699 | 0.572 | 0.720 | 50 incl. any-class / see index |
| 0.30 | 0.674 | 0.592 | 0.733 | 12 (0.8%) |
| 0.40 | 0.756 | 0.528 | 0.696 | 7 (0.47%) |

Answer to the reviewer's question: raising τ from 0.15 to 0.40 removes **75% of the
background fire alarms** (27 → 7) while smoke recall stays above 0.69; fire box recall
drops 0.698 → 0.528. The recall-vs-FP curve is in `threshold_tradeoff.png`.

## R3 — False-positive analysis ✅ (`false_positives/` + `false_positives_index.csv`)

All **50 background scenes** that produce detections at conf ≥ 0.25 are exported as
annotated images, ranked by confidence. Dominant failure modes visible in the top
examples: **golden-hour sunlit clouds** behind treelines (fire 0.79), **storm/overcast
skies** (smoke 0.82 over a full city skyline) and **vertical man-made structures**
(fire 0.26 on a construction crane). This matches the reviewer's expectation
(sunlight, clouds, visually similar textures); fog/dust examples are rarer in D-Fire's
background set.

## R4 — Independent test-set verification ✅ (`leakage_verification.json`)

Recomputed programmatically from the split manifest: 21,527 frames grouped into **673
event groups (471 train / 100 val / 102 test), zero groups straddling splits** — the
build script hard-fails on any overlap. Method: D-Fire publishes no source-video IDs,
so groups are 32-frame numeric-stem buckets; this is *stricter* than the official
mirror partition, which itself places consecutive frames in different splits
(demonstrated in `contribution.md` C2). The approximation is disclosed as a limitation.

## R5 — Operational metrics ✅ partially (`operational_metrics.json`)

- **Detection latency (measured):** 11.7 ms mean / 17.1 ms p95 per image, including
  pre/post-processing (RTX 5070 Laptop, FP32, 640 px; raw traces in
  `latency_raw_traces.json`).
- **Recall at a specified false-alarm rate (measured):** fire box recall **0.528** at
  image-FP ≤ 0.5%, **0.592** at ≤ 1%, **0.698** at ≤ 2% (background scenes).
- **False alarms per hour: not measured** — it requires timestamped negative video,
  which the benchmarks do not contain. The field-sim replay
  (`table_v_fieldsim_SYNTHETIC.csv`) reports per-window FPR but is **synthetic** and
  labeled as such. This remains open until negative full-length footage is collected.
