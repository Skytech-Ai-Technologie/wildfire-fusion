"""§3.6 — Unify class taxonomy to the canonical 2-class set [smoke, fire].

dfire:  remap YOLO class ids per configs/datasets.yaml ({0: smoke, 1: fire} ASSUMPTION),
        write data/dfire/labels_harm/ mirroring labels/, print the class histogram, and
        FAIL LOUDLY if ids outside the mapping appear — verify against the actual label
        files, don't trust the table blindly (per spec).
flame:  Fire/Non-Fire -> fire/none (classification; no boxes).
figlib: smoke/no-smoke -> smoke/none (classification; no boxes).

Usage: python scripts/harmonize_labels.py --dataset dfire
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "datasets.yaml").read_text(encoding="utf-8"))
CANONICAL = CFG["classes"]                      # [smoke, fire]
EXTS = (".jpg", ".jpeg", ".png")


def harmonize_dfire(root: Path) -> None:
    mapping = {int(k): v for k, v in CFG["mapping"]["dfire"].items()}
    labels_root = root / "labels"
    out_root = root / "labels_harm"
    if not labels_root.exists():
        sys.exit(f"{labels_root} missing — run get_datasets.py/build_splits.py first")

    hist = Counter()
    n_files = 0
    for lbl in sorted(labels_root.rglob("*.txt")):
        rel = lbl.relative_to(labels_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        lines_out = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(line.split()[0])
            if cid not in mapping:
                sys.exit(f"HARMONIZATION FAILURE: class id {cid} in {lbl} is not in the "
                         f"dfire mapping {mapping} — check gaiasd README + actual label "
                         "files (spec §3.6) before training.")
            new_id = CANONICAL.index(mapping[cid])
            rest = line.split()[1:]
            lines_out.append(" ".join([str(new_id)] + rest))
            hist[mapping[cid]] += 1
        dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
        n_files += 1
    print(f"[dfire] harmonized {n_files} label files -> {out_root}")
    print(f"[dfire] instance histogram: {dict(hist)}  (canonical ids: "
          f"{CANONICAL})")
    if n_files == 0:
        sys.exit("no label files found — wrong root?")


def report_classification_tree(root: Path, dataset: str) -> None:
    """flame/figlib: verify the class-dir tree matches the mapping and print counts."""
    mapping = CFG["mapping"][dataset]
    found = Counter()
    for img in root.rglob("*"):
        if img.suffix.lower() not in EXTS:
            continue
        parts = [p.lower() for p in img.parts]
        if dataset == "flame":
            if any(p == "fire" for p in parts):
                found["fire"] += 1
            elif any("non" in p and "fire" in p for p in parts):
                found["none"] += 1
        else:  # figlib
            if any("smoke" in p for p in parts) and not any("no" in p for p in parts):
                found["smoke"] += 1
            elif any(p in ("no_smoke", "nosmoke", "negative", "none") for p in parts):
                found["none"] += 1
    print(f"[{dataset}] class-dir histogram: {dict(found)} (mapping: {mapping})")
    if not found:
        print(f"[{dataset}] WARNING: no recognizable class directories under {root} — "
              "check the folder names against configs/datasets.yaml")


def main():
    ap = argparse.ArgumentParser(description="§3.6 label harmonization")
    ap.add_argument("--dataset", required=True, choices=["flame", "dfire", "figlib"])
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    root = Path(args.root) if args.root else REPO / "data" / args.dataset
    if not root.exists():
        sys.exit(f"{root} not found — download the dataset first")
    if args.dataset == "dfire":
        harmonize_dfire(root)
    else:
        report_classification_tree(root, args.dataset)


if __name__ == "__main__":
    main()
