# `data/` — the datasets used (not shipped)

This folder is **intentionally empty in the repository**. The three databases behind
every experiment in this project total ≈9.6 GB and are license-gated, so they are not
committed. Fetch them into this folder, then follow README §2:

```bat
python scripts\get_datasets.py --dataset dfire              :: 3.0 GB, no auth wall
python scripts\get_datasets.py --dataset figlib --subset 5  :: 377 MB (smoke test before the full pull)
python scripts\get_datasets.py --dataset flame              :: prints IEEE DataPort steps (free account + license)
```

## The three databases

| | **D-Fire** | **FLAME 3 CV subset** | **FIgLib** |
|---|---|---|---|
| **Task / modality** | Object detection — RGB stills, day + night | Classification — RGB **+ thermal** pairs, UAV aerial | Fire **sequences** — fixed-view PTZ cameras |
| **Samples** | 21,527 images (14,122 / 3,099 / 4,306 train-val-test) | 2,214 images (fire 1,866 / no-fire 348) | 732 images in 10 sequences |
| **Labels** | YOLO boxes: 0 = smoke, 1 = fire (verified against published counts) | Class folders `fire\` / `none\` | None (negatives from official CV splits) |
| **Source** | Kaggle mirror `sayedgamal99/smoke-fire-detection-yolo` of `gaiasd/DFireDataset` | Kaggle `brycehopkins/flame-3-cv-subset-sycan-marsh` | HPWREN CDN + official split manifest |
| **Size on disk** | 3.0 GB | 6.2 GB | 377 MB |
| **Land in** | `data/dfire/` | `data/flame/` | `data/figlib/` |

Every fetch writes a `.done` provenance marker (timestamp + source URL + file count)
into the dataset folder. Full documentation — split construction, label harmonization,
which pipeline step uses which database, and caveats — is in
[`DATASETS.md`](../DATASETS.md). The leakage-safe split builder is
`scripts/build_splits.py`; label verification is `scripts/harmonize_labels.py`.
