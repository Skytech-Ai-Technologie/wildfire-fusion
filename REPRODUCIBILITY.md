# Reproducibility checklist (spec §12 — the paper's own §VI list)

Treat each item as a literal gate before calling any result "reproduced". Replace all
`ASSUMPTION:` markers with real, justified values before any external claim.

- [ ] **Exact dataset partitions committed** — `data/<ds>/split_manifest.csv` per dataset
      (§3.5, `scripts/build_splits.py`)
- [ ] **Class-label harmonization documented and verified against real label files** (§3.6,
      `scripts/harmonize_labels.py` — it fails loudly on unmapped class ids)
- [ ] **Leakage-control assertion passes (zero event-ID overlap across splits) as an
      automated test** — `pytest tests/test_splits_leakage.py`
- [ ] **Full training config logged per run** (optimizer, LR schedule, epochs, batch size,
      seed, augmentation) — `runs/<exp>/train_config.yaml`
- [ ] **Baselines (YOLOv5s / plain YOLOv8n / YOLOv7-tiny) retrained under identical
      splits/augmentation** — `src/train_perceptual.py --arch baseline`
- [ ] **Per-event confusion matrices + bootstrap confidence intervals reported** —
      `benchmark/run_benchmark.py` (>=1000 resamples)
- [ ] **Raw latency traces saved (not just mean/median)** —
      `results/latency_raw_<model>.json`
- [ ] **Timestamp-sync, packet-loss, UAV-unavailable, sensor-dropout, hysteresis policies
      implemented and documented** (§8 — `configs/default.yaml:operational`,
      `src/pipeline/fusion_loop.py`)
- [ ] **Field-log equivalents (or clearly-labeled synthetic sim §9) with denominators for
      every rate metric** — `scripts/run_field_sim.py`,
      `benchmark/metrics.py:fpr_over_intervals` (logs its denominator definition)
- [ ] **All `ASSUMPTION:` markers in the spec replaced with real, justified values before
      any external claim of "reproducing" the paper** — grep the spec; the key ones are
      registered in README.md
