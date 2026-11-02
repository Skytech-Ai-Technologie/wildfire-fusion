r"""Reviewer-requested analyses on the D-Fire test split (all measured, no estimates).

Produces, from ONE low-threshold prediction sweep over the held-out test split:
  postdo/threshold_analysis.csv        per-threshold fire/smoke box P/R + image-level FP
  postdo/threshold_tradeoff.png        fire recall vs FP rate + precision vs threshold
  postdo/operational_metrics.json      recall at FP budgets (0.5/1/2%), latency, notes
  postdo/false_positives/*.jpg         annotated background scenes detected as fire/smoke
  postdo/false_positives_index.csv     filename, classes, confidences
  postdo/leakage_verification.json     group-level split-overlap verification
  postdo/raw_test_predictions.csv      every detection (for regenerating all metrics)

Usage:
    python scripts/reviewer_analysis.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from benchmark.metrics import _iou_xyxy  # noqa: E402

RUN = REPO / "runs" / "single_rgb_dfire_1791130238"
OUT = REPO / "postdo"
THRESHOLDS = [round(0.05 + 0.05 * i, 2) for i in range(18)]      # 0.05 .. 0.90
FP_BUDGETS = [0.005, 0.01, 0.02]                                 # image-level FP rate
NAMES = {0: "smoke", 1: "fire"}


def gt_label_path(img: Path) -> Path:
    return Path(*("labels" if p == "images" else p for p in img.parts)).with_suffix(".txt")


def load_gt_boxes(lbl: Path, w: int, h: int) -> dict:
    """GT boxes in original-image xyxy pixels, per class id."""
    out = defaultdict(list)
    if not lbl.exists():
        return out
    for line in lbl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        c, cx, cy, bw, bh = line.split()[:5]
        cx, cy, bw, bh = (float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h)
        out[int(float(c))].append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return out


def match_tpfp(gt_boxes, det_boxes) -> tuple[int, int]:
    """Greedy highest-IoU matching at IoU>=0.5; returns (tp, fp)."""
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0
    for d in det_boxes:
        ious = _iou_xyxy(np.asarray(d)[None], np.asarray(gt_boxes) if gt_boxes
                         else np.zeros((0, 4)))[0] if gt_boxes else np.zeros(0)
        best = -1 if not len(ious) else int(np.argmax(ious))
        if best >= 0 and not matched[best] and ious[best] >= 0.5:
            matched[best] = True
            tp += 1
    return tp, len(det_boxes) - tp


def main():
    from ultralytics import YOLO
    OUT.mkdir(exist_ok=True)
    (OUT / "false_positives").mkdir(exist_ok=True)

    files = [Path(l.strip()) for l in
             open(REPO / "data" / "dfire" / "test.txt", encoding="utf-8") if l.strip()]
    model = YOLO(str(RUN / "weights" / "best.pt"))

    imgs = []            # per image: {"file", "gt": {cls: [xyxy]}, "dets": [(cls, conf, xyxy)]}
    for f in files:      # per-image predict: a 3,264-image list source OOM'd the GPU
        r = model.predict(str(f), conf=0.01, imgsz=640, verbose=False)[0]
        h, w = r.orig_shape
        dets = [(int(c), float(s), b.tolist())
                for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                   r.boxes.xyxy.cpu().numpy())] if r.boxes is not None else []
        imgs.append({"file": f, "gt": load_gt_boxes(gt_label_path(f), w, h), "dets": dets})

    # raw predictions dump (regenerates every metric below)
    with open(OUT / "raw_test_predictions.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["image", "cls", "conf", "x1", "y1", "x2", "y2"])
        for im in imgs:
            for c, s, b in im["dets"]:
                w.writerow([im["file"].name, NAMES.get(c, c), f"{s:.4f}",
                            *[f"{v:.1f}" for v in b]])

    # ---- threshold sweep --------------------------------------------------- #
    gt_count = {c: sum(len(im["gt"].get(c, [])) for im in imgs) for c in (0, 1)}
    n_neg = sum(1 for im in imgs if not im["gt"])
    rows, sweep = [], {}
    for t in THRESHOLDS:
        m = {}
        for c in (0, 1):
            tp = fp = 0
            for im in imgs:
                d = [b for cc, s, b in
                     sorted([(cc, s, b) for cc, s, b in im["dets"] if cc == c and s >= t],
                            key=lambda x: -x[1])]                     # greedy by confidence
                tp_c, fp_c = match_tpfp(im["gt"].get(c, []), d)
                tp += tp_c
                fp += fp_c
            m[c] = (tp, fp, tp / gt_count[c] if gt_count[c] else 0.0,
                    tp / (tp + fp) if tp + fp else 0.0)
        fp_fire = sum(1 for im in imgs if not im["gt"]
                      and any(cc == 1 and s >= t for cc, s, _ in im["dets"]))
        fp_any = sum(1 for im in imgs if not im["gt"]
                     and any(s >= t for _, s, _ in im["dets"]))
        rows.append([t, m[0][2], m[0][3], m[1][2], m[1][3],
                     fp_fire, fp_fire / n_neg, fp_any / n_neg])
        sweep[t] = m

    with open(OUT / "threshold_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conf_threshold", "smoke_recall", "smoke_precision",
                    "fire_recall", "fire_precision",
                    "background_images_with_fire_det", "image_fp_rate_fire",
                    "image_fp_rate_any_class"])
        w.writerows([[f"{r[0]:.2f}", f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
                      f"{r[4]:.4f}", r[5], f"{r[6]:.4f}", f"{r[7]:.4f}"] for r in rows])

    # recall at FP budgets (image-level, fire class)
    op_points = {}
    for budget in FP_BUDGETS:
        cands = [r for r in rows if r[6] <= budget]
        if cands:
            best = max(cands, key=lambda r: r[3])       # max fire recall under budget
            op_points[f"fp_rate<={budget:.1%}"] = {
                "conf_threshold": best[0], "fire_recall_box": round(best[3], 4),
                "fire_precision_box": round(best[4], 4),
                "smoke_recall_box": round(best[1], 4),
                "image_fp_rate_fire": round(best[6], 4),
                "background_images_with_fire_det": best[5], "n_background": n_neg}

    # ---- false-positive examples (standard operating conf 0.25) ------------- #
    fps = []
    for im in imgs:
        if im["gt"]:
            continue
        d = [(c, s, b) for c, s, b in im["dets"] if s >= 0.25]
        if d:
            fps.append((max(s for _, s, _ in d), im, d))
    fps.sort(key=lambda x: -x[0])
    index = []
    for rank, (top_conf, im, dets) in enumerate(fps, 1):
        img = cv2.imread(str(im["file"]))
        for c, s, b in dets:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255) if c == 1 else (255, 160, 0), 2)
            cv2.putText(img, f"{NAMES.get(c, c)} {s:.2f}", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        name = f"fp_{rank:02d}_{im['file'].stem}_conf{top_conf:.2f}.jpg"
        cv2.imwrite(str(OUT / "false_positives" / name), img)
        index.append([name, im["file"].name,
                      "; ".join(f"{NAMES.get(c, c)}:{s:.2f}" for c, s, _ in dets)])
    with open(OUT / "false_positives_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["annotated_file", "source_image", "detections(conf>=0.25)"])
        w.writerows(index)

    # ---- leakage verification (recomputed from the manifest) ---------------- #
    manifest = list(csv.DictReader(open(REPO / "data" / "dfire" / "split_manifest.csv",
                                        encoding="utf-8")))
    ev_split = defaultdict(set)
    for r in manifest:
        ev_split[r["event_id"]].add(r["split"])
    straddle = {k: v for k, v in ev_split.items() if len(v) > 1}
    leakage = {
        "method": "32-frame numeric-stem bucketing (no source-video ids published); "
                  "each group assigned to exactly one of train/val/test",
        "event_groups": len(ev_split),
        "frames": len(manifest),
        "groups_straddling_splits": len(straddle),
        "examples_straddling": sorted(straddle)[:5],
        "per_split_groups": {s: sum(1 for v in ev_split.values() if s in v and len(v) == 1)
                             for s in ("train", "val", "test")},
        "assertion": "ZERO overlap — passed" if not straddle else "FAILED",
    }
    json.dump(leakage, open(OUT / "leakage_verification.json", "w", encoding="utf-8"),
              indent=2)

    # ---- trade-off figure --------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot([r[7] * 100 for r in rows], [r[3] for r in rows], "o-",
             color="#cc334d", label="fire recall")
    ax1.plot([r[7] * 100 for r in rows], [r[1] for r in rows], "s-",
             color="#1373ba", label="smoke recall")
    for b in FP_BUDGETS:
        ax1.axvline(b * 100, color="#999", ls=":", lw=1)
    ax1.set_xlabel("image-level false-positive rate on background scenes (%)")
    ax1.set_ylabel("box recall (IoU 0.5)")
    ax1.set_title("Recall vs false alarms across thresholds")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ts, [r[4] for r in rows], "o-", color="#cc334d", label="fire precision")
    ax2.plot(ts, [r[3] for r in rows], "o--", color="#cc334d", alpha=0.6, label="fire recall")
    ax2.plot(ts, [r[2] for r in rows], "s-", color="#1373ba", label="smoke precision")
    ax2.set_xlabel("confidence threshold"); ax2.set_ylabel("box precision / recall")
    ax2.set_title("Operating-point sweep (test split)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_tradeoff.png", dpi=300, bbox_inches="tight")

    # ---- operational metrics ------------------------------------------------ #
    lat = json.load(open(OUT / "latency_raw_traces.json", encoding="utf-8"))
    import statistics
    ops = {
        "detection_latency_ms": {"mean": round(statistics.mean(lat["raw_ms"]), 1),
                                 "p95": round(sorted(lat["raw_ms"])[int(0.95 * len(lat["raw_ms"]))], 1),
                                 "n": len(lat["raw_ms"]),
                                 "hardware": "RTX 5070 Laptop GPU, FP32, 640 px, includes pre+post"},
        "recall_at_image_fp_budget": op_points,
        "background_scenes": n_neg,
        "background_with_fire_det_at_conf_0.25": len(fps),
        "false_alarms_per_hour": "not measurable — requires timestamped negative video; "
                                 "see table_v_fieldsim_SYNTHETIC.csv for the replayed "
                                 "per-window FPR (labeled synthetic)",
    }
    json.dump(ops, open(OUT / "operational_metrics.json", "w", encoding="utf-8"),
              indent=2)
    print(json.dumps({"thresholds": len(rows), "false_positives_annotated": len(fps),
                      "n_neg": n_neg, "op_points": op_points,
                      "leakage": leakage["assertion"]}, indent=2))


if __name__ == "__main__":
    main()
