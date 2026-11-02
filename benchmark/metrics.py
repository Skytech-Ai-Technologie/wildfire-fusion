"""§10 — Benchmark & metrics harness.

P/R/F1, mAP@.5 (wrapped Ultralytics val), FPR over 24h windows with an EXPLICIT
denominator (the paper flags the window count as an open definition), detection horizon,
latency with raw traces, bootstrap CIs (the paper provides none), and per-class
confusion matrices (also absent from the paper).
"""
import time

import numpy as np


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def map_at_iou(weights_path: str, data_yaml: str, iou_threshold=0.5, split="test"):
    """Use Ultralytics' built-in val() for mAP@.5 — it already matches the paper's
    IoU=0.5 convention. Wrapped here for a uniform interface."""
    from ultralytics import YOLO
    results = YOLO(weights_path).val(data=data_yaml, iou=iou_threshold, split=split,
                                     verbose=False)
    return {"map50": float(results.box.map50), "map50_95": float(results.box.map),
            "precision": float(results.box.mp), "recall": float(results.box.mr)}


def _iou_xyxy(a, b):
    """Pairwise IoU; a: (N,4), b: (M,4), xyxy."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def map_from_detections(dets_per_image, gts_per_image, n_classes=2, iou_threshold=0.5):
    """mAP@IoU from raw boxes — the dual-stream checkpoints are raw state_dicts with a
    6-channel input, which Ultralytics' val() cannot consume (map_at_iou needs a YOLO).

    VOC-style greedy matching (highest-IoU unmatched GT), all-point interpolated AP;
    precision/recall reported at the max-F1 point of the pooled PR curve, mirroring
    Ultralytics' mp/mr convention. Classes with no GT are excluded from the mean.

    dets_per_image: [{"labels": [int], "scores": [float], "boxes": [(x1,y1,x2,y2) px]}]
    gts_per_image:  [{"labels": [int], "boxes": [(x1,y1,x2,y2) px]}]
    """
    aps, precisions, recalls = [], [], []
    for c in range(n_classes):
        gt_n = sum(1 for g in gts_per_image for lc in g["labels"] if lc == c)
        if gt_n == 0:
            continue
        gt_boxes = [[np.asarray(bx, float) for lc, bx in zip(g["labels"], g["boxes"])
                     if lc == c] for g in gts_per_image]
        matched = [np.zeros(len(bxs)) for bxs in gt_boxes]
        rows = sorted(((float(s), i, np.asarray(bx, float))
                       for i, d in enumerate(dets_per_image)
                       for lc, s, bx in zip(d["labels"], d["scores"], d["boxes"])
                       if lc == c), key=lambda r: -r[0])
        if not rows:
            aps.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        tp = np.zeros(len(rows))
        for k, (_, i, box) in enumerate(rows):
            best_j, best_iou = -1, iou_threshold
            for j, gtb in enumerate(gt_boxes[i]):
                if matched[i][j]:
                    continue
                iou = _iou_xyxy(box[None], gtb[None])[0, 0]
                if iou >= best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                matched[i][best_j] = 1
                tp[k] = 1
        tp_c, fp_c = np.cumsum(tp), np.cumsum(1 - tp)
        recall = tp_c / gt_n
        precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):          # envelope from the right
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        k = int(np.argmax(f1))
        precisions.append(float(precision[k]))
        recalls.append(float(recall[k]))
    return {"map50": float(np.mean(aps)) if aps else 0.0,
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0}


def fpr_over_intervals(alarms: list, negative_windows: list) -> dict:
    """negative_windows: confirmed-no-fire 24h windows from field/video logs as
    (start, end) tuples. An alarm is a false positive if its timestamp falls inside a
    negative window. FPR = false_alarms / number_of_24h_windows_monitored — the
    denominator is defined and logged explicitly (the paper's flagged gap)."""
    ts = [a["t"] if isinstance(a, dict) else a for a in alarms]
    false_alarms = sum(1 for t in ts if any(s <= t <= e for s, e in negative_windows))
    n = len(negative_windows)
    return {"false_alarms": false_alarms,
            "windows_monitored": n,
            "fpr_per_window": false_alarms / n if n else 0.0,
            "denominator_definition":
                "denominator: false alarms / number of monitored no-fire windows"}


def detection_horizon(alarm_time: float, legacy_monitor_time: float) -> float:
    return legacy_monitor_time - alarm_time   # positive = fused system alarmed earlier


def measure_latency(fn, *args, n_repeats=100, **kwargs) -> dict:
    """Raw per-call traces are returned (not just mean/median) — save them, per the
    reproducibility checklist."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return {"mean_ms": sum(times) / len(times),
            "p95_ms": sorted(times)[int(0.95 * len(times))],
            "raw_ms": times}


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42) -> dict:
    """Percentile bootstrap CI over resamples of the test set (>=1000 resamples per
    the spec) — the direct fix for the paper's missing confidence intervals."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    stats = vals[idx].mean(axis=1)
    return {"mean": float(vals.mean()),
            "lo": float(np.quantile(stats, alpha / 2)),
            "hi": float(np.quantile(stats, 1 - alpha / 2)),
            "n": int(vals.size)}


def image_level_pr(detections_per_image, labels_per_image, conf=0.25) -> dict:
    """Image-level (any-detection) P/R/F1. detections: list of list-of-(cls, score);
    labels: 1 if any fire/smoke GT present else 0."""
    tp = fp = fn = 0
    per_image = []
    for dets, lab in zip(detections_per_image, labels_per_image):
        pred_pos = any(s >= conf for _, s in dets)
        if pred_pos and lab:
            tp += 1
        elif pred_pos and not lab:
            fp += 1
        elif lab:
            fn += 1
        per_image.append(1.0 if pred_pos == bool(lab) else 0.0)
    p, r, f = precision_recall_f1(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f,
            "per_image_correct": per_image}


def confusion_matrix(y_true, y_pred, labels=(0, 1)) -> list:
    """Per-class confusion matrix; sklearn when available, else a manual 2x2."""
    try:
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()
    except ImportError:
        m = [[0] * len(labels) for _ in labels]
        for t, p in zip(y_true, y_pred):
            m[labels.index(t)][labels.index(p)] += 1
        return m
