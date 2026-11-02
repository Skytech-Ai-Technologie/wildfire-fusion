"""§10.5 — Reproduces Table IV / V with real numbers + bootstrap CIs.

  Table IV: per-checkpoint P/R/F1/mAP50/latency per dataset (+ image-level eval with
            bootstrap CIs and per-class confusion matrices — both absent from the paper).
  Table V:  field-sim (§9) at each fusion configuration (PTZ-only, +LWIR, +IIoT+fusion,
            IIoT-only) -> detection horizon / alarm rate per config.

Baselines: YOLOv5s / plain YOLOv8n trained by YOU under the SAME splits/augmentation
(paper flags that this parity isn't confirmed in the source — make it true here).

Usage:
  python benchmark/run_benchmark.py --table iv --dataset dfire \
      --weights runs/single_rgb_dfire_0/weights/best.pt --weights runs/best.pt ...
  python benchmark/run_benchmark.py --table v --events 3
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark.metrics import (precision_recall_f1, map_at_iou, measure_latency,  # noqa: E402
                               bootstrap_ci, image_level_pr, confusion_matrix,
                               map_from_detections)
from src.perceptual.dual_stream_yolo import try_load_dual_stream  # noqa: E402


def _cxcywh_to_xyxy_px(bboxes, imgsz: int) -> list:
    import numpy as np
    b = np.asarray(bboxes, dtype=float).reshape(-1, 4) * imgsz
    if not len(b):
        return []
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1).tolist()


def _dual_detections(dual, dataset: str, files: list, conf: float):
    """Run a dual-stream checkpoint over a split. DualStreamYoloDataset supplies the
    6-channel inputs and the letterboxed GT; predict_simple decodes detections.
    Returns (dets [(cls, conf) per image], labels, det box dicts, GT box dicts)."""
    import torch
    import yaml as _yaml
    from src.train_perceptual import DualStreamYoloDataset

    dev = next(dual.parameters()).device
    imgsz = _yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))["imgsz"]
    lwir_dir = REPO / "data" / dataset / "lwir"
    ds = DualStreamYoloDataset(files, lwir_dir if lwir_dir.exists() else None, imgsz=imgsz)
    dets, labels, det_boxes, gt_boxes = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            out = dual.predict_simple(item["img"].unsqueeze(0).to(dev).float().div_(255.0),
                                      conf_thres=conf)[0]
            dets.append(list(zip(out["labels"].tolist(), out["scores"].tolist())))
            det_boxes.append({"labels": out["labels"].tolist(),
                              "scores": out["scores"].tolist(),
                              "boxes": out["boxes"].tolist()})
            gt_boxes.append({"labels": item["cls"].squeeze(-1).tolist(),
                             "boxes": _cxcywh_to_xyxy_px(item["bboxes"], imgsz)})
            labels.append(1 if len(item["cls"]) else 0)
    return dets, labels, det_boxes, gt_boxes


def evaluate_checkpoint(weights: str, dataset: str, split="test", conf=0.25,
                        n_repeats=50) -> dict:
    from ultralytics import YOLO
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    row = {"model": Path(weights).parts[-3] if len(Path(weights).parts) >= 3 else weights,
           "weights": weights, "dataset": dataset}
    img_list = REPO / "data" / dataset / f"{split}.txt"
    files = ([l.strip() for l in img_list.read_text(encoding="utf-8").splitlines() if l.strip()]
             if img_list.exists() else [])

    dual = try_load_dual_stream(weights, dev)
    if dual is not None:
        # raw state_dict + 6-ch input: Ultralytics val()/predict() can't touch it
        row["note"] = "dual-stream checkpoint — custom eval path (predict_simple)"
        if files:
            dets, labels, det_boxes, gt_boxes = _dual_detections(dual, dataset, files, conf)
            row.update(map_from_detections(det_boxes, gt_boxes, n_classes=dual.nc))
            row["map50_95"] = None   # needs an IoU sweep; Table IV's metric is mAP@.5
            _, _, f1 = precision_recall_f1_from_pr(row["precision"], row["recall"])
            row["f1"] = f1
    else:
        data_yaml = REPO / "data" / dataset / "dataset.yaml"
        if data_yaml.exists():  # detection datasets -> ultralytics val
            r = map_at_iou(weights, str(data_yaml), iou_threshold=0.5, split=split)
            p, _, f1 = precision_recall_f1_from_pr(r["precision"], r["recall"])
            row.update({**r, "f1": f1})
        else:
            row.update({"map50": None, "note": f"{data_yaml} missing — run build_splits.py"})
        dets, labels = [], []
        if files:
            model = YOLO(weights)
            model.to(dev)
            for fp in files:
                r = model.predict(fp, conf=conf, verbose=False)[0]
                dets.append(list(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()))
                            if r.boxes is not None and len(r.boxes) else [])
                # the image lists hold resolved native-separator paths (backslashes on
                # Windows) — rebuild via parts so the label lookup works on both
                lbl = Path(*("labels" if part == "images" else part
                             for part in Path(fp).parts)).with_suffix(".txt")
                gt = set()
                if lbl.exists():
                    gt = {int(l.split()[0]) for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()}
                labels.append(1 if gt else 0)

    # image-level eval + bootstrap CI + confusion matrix (both eval paths fill dets/labels)
    if files:
        il = image_level_pr(dets, labels, conf=conf)
        ci = bootstrap_ci(il.pop("per_image_correct"), n_boot=1000)
        y_pred = [1 if any(s >= conf for _, s in d) else 0 for d in dets]
        row.update({"image_level": il, "accuracy_ci": ci,
                    "confusion_matrix_rows0_cols1": confusion_matrix(labels, y_pred)})

    # latency on one image (raw traces saved, per checklist)
    if files:
        if dual is not None:
            import yaml as _yaml
            from src.train_perceptual import DualStreamYoloDataset
            imgsz = _yaml.safe_load(
                (REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))["imgsz"]
            lwir_dir = REPO / "data" / dataset / "lwir"
            item0 = DualStreamYoloDataset([files[0]], lwir_dir if lwir_dir.exists() else None,
                                          imgsz=imgsz)[0]
            img0 = item0["img"].unsqueeze(0).to(dev).float().div_(255.0)
            lat = measure_latency(lambda: dual.predict_simple(img0, conf_thres=conf),
                                  n_repeats=n_repeats)
        else:
            model = YOLO(weights)
            lat = measure_latency(model.predict, files[0], verbose=False, n_repeats=n_repeats)
        row["latency_ms"] = {"mean": lat["mean_ms"], "p95": lat["p95_ms"]}
        (REPO / "results" / f"latency_raw_{row['model']}.json").write_text(
            json.dumps(lat), encoding="utf-8")
    return row


def precision_recall_f1_from_pr(p, r):
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def run_table_iv(args):
    out = REPO / "results"
    out.mkdir(exist_ok=True)
    rows = [evaluate_checkpoint(w, args.dataset, split=args.split, conf=args.conf)
            for w in args.weights]
    with open(out / "table_iv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "weights", "dataset", "map50",
                                          "map50_95", "precision", "recall", "f1",
                                          "image_level", "accuracy_ci",
                                          "confusion_matrix_rows0_cols1", "latency_ms",
                                          "note"])
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in r.items()})
    print(f"Table IV -> {out / 'table_iv.csv'}")
    for r in rows:
        print(json.dumps(r, indent=2, default=str))


def run_table_v(args):
    import importlib
    sim = importlib.import_module("scripts.run_field_sim")
    out = REPO / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for config in ("ptz_only", "plus_lwir", "plus_iiot", "iiot_only"):
        horizons, alarms = [], 0
        for i in range(args.events):
            # no encoder passed -> identity contextual refinement (see run_field_sim)
            ev = sim.simulate_event(i, sim.CONFIG_MODALITIES[config], seed=args.seed)
            if ev["detection_horizon_s"] is not None:
                horizons.append(ev["detection_horizon_s"])
                alarms += 1
        mean_h = float(sum(horizons) / len(horizons)) if horizons else None
        rows.append({"config": config, "n_events": args.events, "alarm_rate": alarms / args.events,
                     "detection_horizon_mean_s": mean_h,
                     "horizon_ci": bootstrap_ci(horizons, n_boot=1000) if horizons else None})
    with open(out / "table_v.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "n_events", "alarm_rate",
                                          "detection_horizon_mean_s", "horizon_ci"])
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, dict) else v for k, v in r.items()})
    print(f"Table V -> {out / 'table_v.csv'}  (SYNTHETIC field replay, not a field trial)")
    for r in rows:
        print(json.dumps(r, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(description="§10.5 benchmark harness")
    ap.add_argument("--table", choices=["iv", "v", "both"], default="both")
    ap.add_argument("--dataset", default="dfire")
    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--events", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.table in ("iv", "both"):
        if not args.weights:
            print("Table IV needs --weights (repeatable); skipping")
        else:
            run_table_iv(args)
    if args.table in ("v", "both"):
        run_table_v(args)


if __name__ == "__main__":
    main()
