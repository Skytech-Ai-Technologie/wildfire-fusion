# Datasets — what each database is, and where it is used

All three databases live under `data\`, were fetched by `scripts\get_datasets.py`, and
carry a `.done` provenance marker (timestamp + source URL + file count). Verified
2026-10-04/05 on this machine.

## The three databases at a glance

| | **D-Fire** | **FLAME 3 CV subset** | **FIgLib** |
|---|---|---|---|
| **Task type** | Object **detection** (bounding boxes) | **Classification** (fire / no-fire) | **Sequences** (video frames, events) |
| **Modalities** | RGB stills, day + night | RGB **+ thermal** pairs, UAV aerial | RGB, fixed-view PTZ cameras |
| **Samples** | **21,527** images (14,122 train / 3,099 val / 4,306 test) | **2,214** images (fire 1,866 / no-fire 348) | **732** images in **10 fire sequences** (62–81 per sequence) |
| **Labels** | YOLO boxes: 0 = smoke, 1 = fire (11,865 + 14,692 instances — exact match to published counts) | Class folders: `fire\` / `none\` | None (frames only; negatives come from the official CV splits) |
| **Source** | Kaggle mirror `sayedgamal99/smoke-fire-detection-yolo` of `gaiasd/DFireDataset` | Kaggle `brycehopkins/flame-3-cv-subset-sycan-marsh` | HPWREN CDN + official split manifest |
| **Size on disk** | 3.0 GB | 6.2 GB | 377 MB |
| **Location** | `data\dfire\{images,labels}\{train,val,test}` | `data\flame\{fire,none}\...` | `data\figlib\images\<sequence>\` |

## Which step uses which database

| Pipeline step | D-Fire | FLAME | FIgLib |
|---|---|---|---|
| Perceptual detector training (`train_perceptual.py`) | ✅ **primary** (boxes) | optional (classification tree) | — |
| Dual-stream RGB+LWIR training | ✅ (LWIR = grayscale proxy) | ✅ thermal available | — |
| Table IV benchmark (test split) | ✅ **primary** | optional | — |
| Contextual tier (32-frame windows, `build_windows`) | ✅ (windows from the split manifest) | — | ✅ (sequence-level events) |
| Field simulation (`run_field_sim.py`) | — | — | informs sequence behavior (synthetic replay) |
| Edge export / TensorRT calibration list | ✅ | — | — |

## Notes and caveats

- **D-Fire class mapping is verified**: 11,865 smoke + 14,692 fire instances exactly
  match the published counts, confirming 0 = smoke, 1 = fire (commonly reported both
  ways across forks). `harmonize_labels.py` fails loudly on any foreign class id.
- **D-Fire ships no source-video IDs**, so splits use numeric-stem bucketing (32-frame
  blocks) to keep near-duplicate frames together — see `scripts\build_splits.py` and
  `contribution.md` C2. The official mirror partition itself leaks consecutive frames.
- **FLAME here is the no-login Kaggle subset** (single Sycan Marsh burn). The full
  multi-burn FLAME 3 / FLAME 2 sets need an IEEE DataPort account + license; FLAME 2's
  *registered* RGB/IR pairs are the right data for real two-modal (non-proxy) dual-stream
  training.
- **FIgLib images are fire sequences only** — negative windows must come from the
  official CV splits, kept sequence-level to prevent leakage (§3.5).
- No checksums are computed anywhere; verification is count- and histogram-based.
