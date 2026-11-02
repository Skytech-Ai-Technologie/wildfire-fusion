r"""§3.5 — Leakage-safe, video-level splitting.

Rule: split at the video/sequence/burn-event ID, never at the frame level. Group every
frame by its parent event (FLAME: burn/flight ID; D-Fire: numeric-stem bucket — the
public release ships no source-video ids, but stems are <prefix><number> and
consecutive numbers are temporally adjacent frames of one source video; FIgLib:
fire-sequence ID from the official manifest).

Splits 70/15/15 train/val/test, stratified at the GROUP level by the group's label rank
(fire > smoke > none) so rare classes aren't lost to one split, then asserts ZERO
event-ID overlap across splits — the concrete implementation of the paper's flagged
risk that "random frame-level splitting can overestimate generalization."

Outputs per dataset:
    data/<name>/split_manifest.csv   frame_path,event_id,split,label
    data/<name>/{train,val,test}.txt  ultralytics-compatible image lists
    data/<name>/dataset.yaml          ultralytics detect config

Usage:
    python scripts/build_splits.py --dataset dfire
    python scripts/build_splits.py --dataset dfire --group-regex '^(.*?)_(\d+)$'
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RANK = {"none": 0, "smoke": 1, "fire": 2}
# ASSUMPTION: D-Fire stems are <prefix><zero-padded number> (AoF00000, WEB00001,
# PublicDataset0001) and consecutive numbers are temporally adjacent frames of one
# source video — the official mirror split itself puts AoF00004 in train and AoF00008
# in val. Bucketing the number line into DFIRE_FRAME_BLOCK-stem blocks keeps
# near-duplicates inside one group instead of leaking them across splits.
DFIRE_STEM_RE = re.compile(r"(\D*?)(\d+)")
DFIRE_FRAME_BLOCK = 32
DEFAULT_GROUP_REGEXES = {
    "flame": r"^(.*?)_?[fF]rame",
    "dfire": None,   # numeric-stem bucketing in group_id_for; --group-regex overrides
    "figlib": r"^(.*?)[/\\]",
}
EXTS = (".jpg", ".jpeg", ".png")


def group_id_for(frame_path: Path, dataset: str, regex: str | None) -> str:
    stem = frame_path.stem
    if regex:
        m = re.search(regex, stem)
        if m and m.group(1):
            return m.group(1)
    if dataset == "dfire":
        m = DFIRE_STEM_RE.fullmatch(stem)
        if m:
            return f"{m.group(1)}#{int(m.group(2)) // DFIRE_FRAME_BLOCK:06d}"
    if dataset == "figlib" and str(frame_path.parent) not in ("", "."):
        return str(frame_path.parent)          # sequence dir = event id
    return stem                                # isolated still -> own group


def label_for_dfire(img: Path, labels_root: Path) -> str:
    # labels may sit flat in labels/ or mirrored per split (labels/<split>/<stem>.txt,
    # the ultralytics convention get_datasets.py produces)
    candidates = [labels_root / img.parent.name / (img.stem + ".txt"),
                  labels_root / (img.stem + ".txt")]
    lbl = next((c for c in candidates if c.exists()), None)
    if lbl is None:
        return "none"
    classes = set()
    for line in lbl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            classes.add(int(line.split()[0]))
    # ASSUMPTION per configs/datasets.yaml: 0 = smoke, 1 = fire (verify at download!)
    if 1 in classes:
        return "fire"
    if 0 in classes:
        return "smoke"
    return "none"


def collect_dfire(root: Path):
    imgs_root, lbl_root = root / "images", root / "labels"
    records = []
    for split_dir in ("train", "val", "test"):
        d = imgs_root / split_dir
        if not d.exists():
            continue
        for img in sorted(d.rglob("*")):
            if img.suffix.lower() in EXTS:
                records.append({"frame_path": img.resolve(), "label": label_for_dfire(img, lbl_root)})
    return records


def collect_class_dirs(root: Path):
    """FLAME-style classification tree: <root>/.../{Fire,Non-Fire}/... (case-insensitive
    match on 'fire' with a 'non' variant). Also accepts data/figlib images/ with
    smoke/none dirs."""
    records = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in EXTS:
            continue
        parts = [p.lower() for p in img.parts]
        if any(p == "fire" for p in parts):
            label = "fire"
        elif any("non" in p and "fire" in p for p in parts):
            label = "none"
        elif any("smoke" in p for p in parts):
            label = "smoke"
        elif any(p in ("none", "negative", "no_smoke", "nofire", "non_fire") for p in parts):
            label = "none"
        else:
            continue
        records.append({"frame_path": img.resolve(), "label": label})
    return records


def make_group_splits(records: list, seed: int = 42, train_frac=0.70, val_frac=0.15):
    """Stratified GROUP-level split. records: [{frame_path, label}] (+optional event_id).

    Returns {frame_path: 'train'|'val'|'test'}. Zero event overlap is asserted here and
    again by tests/test_splits_leakage.py.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    import numpy as np

    groups = defaultdict(list)
    for r in records:
        gid = r.get("event_id") or group_id_for(Path(r["frame_path"]), r.get("dataset", ""), r.get("regex"))
        groups[gid].append(r)

    gids, ranks = [], []
    for gid, rs in groups.items():
        gids.append(gid)
        ranks.append(max(RANK[r["label"]] for r in rs))

    gids = np.array(gids, dtype=object)
    ranks = np.array(ranks)
    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 0

    assignment = {}
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
    train_idx, temp_idx = next(sss1.split(gids, ranks))
    rel_test = test_frac / (val_frac + test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed + 1)
    val_idx, test_idx = next(sss2.split(gids[temp_idx], ranks[temp_idx]))
    val_idx, test_idx = temp_idx[val_idx], temp_idx[test_idx]

    for idx, split in ((train_idx, "train"), (val_idx, "val"), (test_idx, "test")):
        for gid in gids[idx]:
            for r in groups[gid]:
                assignment[str(r["frame_path"])] = split

    # LEAKAGE ASSERTION — must never fire
    overlap = (set(gids[train_idx]) & set(gids[val_idx])
               | set(gids[train_idx]) & set(gids[test_idx])
               | set(gids[val_idx]) & set(gids[test_idx]))
    if overlap:
        raise AssertionError(f"LEAKAGE: event-id overlap across splits: {sorted(overlap)[:5]}")
    return assignment


def main():
    ap = argparse.ArgumentParser(description="§3.5 leakage-safe splits")
    ap.add_argument("--dataset", required=True, choices=["flame", "dfire", "figlib"])
    ap.add_argument("--root", default=None, help="override data/<dataset> root")
    ap.add_argument("--group-regex", default=None,
                    help="regex whose group(1) is the event id extracted from the stem")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root) if args.root else REPO / "data" / args.dataset
    if not root.exists():
        sys.exit(f"{root} not found — run scripts/get_datasets.py --dataset {args.dataset} first")

    if args.dataset == "dfire":
        records = collect_dfire(root)
    else:
        records = collect_class_dirs(root)
    if not records:
        sys.exit(f"no labeled images found under {root}")
    for r in records:
        r["dataset"] = args.dataset
        r["regex"] = args.group_regex or DEFAULT_GROUP_REGEXES[args.dataset]

    assignment = make_group_splits(records, seed=args.seed)

    out = root
    rows = ["frame_path,event_id,split,label"]
    lists = {"train": [], "val": [], "test": []}
    for r in records:
        fp = str(r["frame_path"])
        split = assignment[fp]
        gid = r.get("event_id") or group_id_for(Path(fp), args.dataset, r.get("regex"))
        rows.append(f"{fp},{gid},{split},{r['label']}")
        lists[split].append(fp)
    (out / "split_manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for split, fps in lists.items():
        (out / f"{split}.txt").write_text("\n".join(fps) + "\n", encoding="utf-8")

    # Ultralytics detect yaml — detection datasets only (dfire). Classification
    # datasets (flame/figlib) train via the class-dir tree; see train_perceptual.py.
    if args.dataset == "dfire":
        (out / "dataset.yaml").write_text(yaml.safe_dump({
            "path": str(out.resolve()),
            "train": "train.txt", "val": "val.txt", "test": "test.txt",
            "names": {0: "smoke", 1: "fire"},
        }), encoding="utf-8")

    from collections import Counter
    print(f"[{args.dataset}] frames: {Counter(a for a in assignment.values())}")
    print(f"wrote {out/'split_manifest.csv'} (+ train/val/test lists"
          + (", dataset.yaml" if args.dataset == "dfire" else "") + ")")


if __name__ == "__main__":
    main()
