"""§6 — Calibrate reliability coefficients r_c, r_d, r_s and alarm threshold τ.

Grid-searches on a validation CSV of per-tick per-modality raw confidences
(columns: modality (c|d|s), pi, label (0/1)), minimizing FPR subject to
recall >= --target-recall. Two-stage search (coarse grid -> local refinement) keeps
runtime sane — a full 19^3 x 10 grid re-fuses every tick 70k times.

Writes configs/decisional_calibration.yaml:
    reliability: {c: ..., d: ..., s: ...}
    fire_alarm_threshold: ...
    divergence_threshold: ...
    calibrated_on: {n_ticks, target_recall, achieved_recall, fpr, seed}

Refuses to run on data without both positive and negative labels (field-sim timelines
carry no per-tick labels; a calibration written against invented zeros would silently
override tau in every FusionLoop).

Usage:
    python src/calibrate_decisional.py --ticks-csv results/field_sim/labeled_ticks.csv
    python src/calibrate_decisional.py --synthetic   # self-contained demo, labeled as such
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.decisional.evidential_fusion import mass_function, conflict_aware_fuse  # noqa: E402

MODALITIES = ["c", "d", "s"]


def load_ticks(csv_path: Path):
    import csv as _csv
    ticks, labels = {}, {}
    with open(csv_path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            t = float(row["t"])
            ticks.setdefault(t, {})[row.get("modality", "c")] = float(row["pi"])
            labels[t] = int(float(row["label"]))
    return [ticks[t] for t in sorted(ticks)], [labels[t] for t in sorted(ticks)]


def synthetic_ticks(n=1200, ignition=600, seed=0):
    """Self-contained synthetic calibration data (clearly synthetic)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = lambda start, rise, ceil_: ceil_ / (1 + np.exp(-(t - start) / rise))  # noqa: E731
    pi = {
        "c": np.clip(0.03 + base(ignition + 45, 20, 0.9) + rng.normal(0, 0.02, n), 0, 0.97),
        "d": np.clip(0.04 + base(ignition + 30, 18, 0.92) + rng.normal(0, 0.02, n), 0, 0.97),
        "s": np.clip(0.05 + 0.8 * np.clip((t - (ignition - 120)) / 240, 0, 1)
                     + rng.normal(0, 0.03, n), 0, 0.97),
    }
    labels = (t >= ignition).astype(int)
    return [{m: float(pi[m][i]) for m in MODALITIES} for i in range(n)], list(labels)


def evaluate(ticks, labels, r, tau, div_thr):
    tp = fp = fn = tn = 0
    for tick, lab in zip(ticks, labels):
        masses = [mass_function(tick[m], r[m]) for m in MODALITIES if m in tick]
        if not masses:
            pred = 0
        else:
            fused, _ = conflict_aware_fuse(masses, divergence_threshold=div_thr)
            pred = int(fused["F"] > tau)
        if pred and lab:
            tp += 1
        elif pred and not lab:
            fp += 1
        elif lab:
            fn += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return recall, fpr


def calibrate(ticks, labels, target_recall=0.90, div_thr=0.30):
    best = None
    coarse = [0.2, 0.4, 0.6, 0.8, 0.95]
    taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for rc in coarse:
        for rd in coarse:
            for rs in coarse:
                for tau in taus:
                    rec, fpr = evaluate(ticks, labels, {"c": rc, "d": rd, "s": rs}, tau, div_thr)
                    feasible = rec >= target_recall
                    key = (0 if feasible else 1, fpr, -(rec - target_recall))
                    if best is None or key < best[0]:
                        best = (key, {"c": rc, "d": rd, "s": rs}, tau, rec, fpr)

    # local refinement around the coarse winner
    _key, r0, tau0, _, _ = best
    for m in MODALITIES:
        for d in np.arange(-0.08, 0.081, 0.02):
            r = dict(r0)
            r[m] = float(np.clip(r0[m] + d, 0.05, 1.0))
            for tau in np.arange(tau0 - 0.05, tau0 + 0.051, 0.01):
                tau = float(np.clip(tau, 0.3, 0.99))
                rec, fpr = evaluate(ticks, labels, r, tau, div_thr)
                feasible = rec >= target_recall
                key = (0 if feasible else 1, fpr, -(rec - target_recall))
                if key < best[0]:
                    best = (key, r, tau, rec, fpr)
    return best[1], best[2], best[3], best[4]


def main():
    ap = argparse.ArgumentParser(description="§6 decisional calibration")
    ap.add_argument("--csv", default=None,
                    help="DEPRECATED: field-sim timelines carry no per-tick labels — "
                         "calibration against them is refused; use --ticks-csv")
    ap.add_argument("--ticks-csv", default=None,
                    help="per-tick per-modality CSV with columns t, modality, pi, label")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--divergence-threshold", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "configs" / "decisional_calibration.yaml"))
    args = ap.parse_args()

    if args.synthetic or (not args.csv and not args.ticks_csv):
        print("SYNTHETIC calibration data (pipeline demo — recalibrate on real val data!)")
        ticks, labels = synthetic_ticks(seed=args.seed)
    elif args.ticks_csv:
        ticks, labels = load_ticks(Path(args.ticks_csv))
    else:
        sys.exit(
            "ERROR: field-sim timelines (--csv) carry no per-tick labels — calibrating "
            "against invented zeros would tune r/tau to garbage, and FusionLoop "
            "auto-loads that yaml. Export labeled validation ticks instead "
            "(--ticks-csv with columns t, modality, pi, label) or use --synthetic "
            "for the demo.")

    n_pos = int(sum(labels))
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        sys.exit(f"refusing to calibrate: degenerate labels ({n_pos} positive / "
                 f"{n_neg} negative of {len(labels)} ticks) — recall/FPR are meaningless "
                 "without both classes. Export labeled validation ticks (--ticks-csv).")
    print(f"calibrating on {len(ticks)} ticks ({n_pos} positive / {n_neg} negative)")

    r, tau, rec, fpr = calibrate(ticks, labels, args.target_recall, args.divergence_threshold)
    out = {
        "reliability": {k: round(v, 3) for k, v in r.items()},
        "fire_alarm_threshold": round(tau, 3),
        "divergence_threshold": args.divergence_threshold,
        "calibrated_on": {
            "n_ticks": len(ticks), "target_recall": args.target_recall,
            "achieved_recall": round(rec, 3), "fpr": round(fpr, 4),
            "SYNTHETIC": bool(args.synthetic),
        },
    }
    Path(args.out).write_text(yaml.safe_dump(out), encoding="utf-8")
    print(yaml.safe_dump(out))
    print(f"written -> {args.out}  (fusion_loop loads this file; never hardcode r_m)")


if __name__ == "__main__":
    main()
