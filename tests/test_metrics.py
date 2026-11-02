"""§10 — benchmark metrics tests."""
import pytest

np = pytest.importorskip("numpy")

from benchmark.metrics import (precision_recall_f1, fpr_over_intervals, detection_horizon,
                               measure_latency, bootstrap_ci, image_level_pr, confusion_matrix,
                               map_from_detections)


def test_precision_recall_f1():
    assert precision_recall_f1(80, 20, 20) == pytest.approx((0.8, 0.8, 0.8))
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)
    assert precision_recall_f1(10, 0, 0) == (1.0, 1.0, 1.0)


def test_fpr_over_intervals_denominator_logged():
    windows = [(0, 100), (200, 300)]
    alarms = [{"t": 50}, {"t": 250}, {"t": 150}]     # 150 is outside any window
    out = fpr_over_intervals(alarms, windows)
    assert out["false_alarms"] == 2
    assert out["windows_monitored"] == 2
    assert out["fpr_per_window"] == pytest.approx(1.0)
    assert "denominator" in out["denominator_definition"]


def test_detection_horizon_sign():
    assert detection_horizon(alarm_time=100, legacy_monitor_time=186) == 86.0  # earlier
    assert detection_horizon(200, 100) == -100.0                               # later


def test_measure_latency_raw_traces():
    out = measure_latency(sum, [1, 2, 3], n_repeats=5)
    assert len(out["raw_ms"]) == 5
    assert out["mean_ms"] >= 0
    assert out["p95_ms"] >= out["mean_ms"] - 1e-9


def test_bootstrap_ci_brackets_mean():
    vals = list(np.random.default_rng(0).normal(5, 1, 500))
    ci = bootstrap_ci(vals, n_boot=1000)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert abs(ci["mean"] - 5.0) < 0.2
    assert ci["n"] == 500


def test_image_level_pr():
    dets = [[(1, 0.9)], [(0, 0.1)], [(1, 0.8)], []]
    labels = [1, 0, 0, 0]
    out = image_level_pr(dets, labels, conf=0.5)
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 0)
    assert len(out["per_image_correct"]) == 4


def test_confusion_matrix():
    assert confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0]) == [[2, 0], [1, 1]]


def test_map_from_detections_perfect_and_missed():
    dets = [{"labels": [1], "scores": [0.9], "boxes": [[10, 10, 50, 50]]}]
    gts = [{"labels": [1], "boxes": [[10, 10, 50, 50]]}]
    assert map_from_detections(dets, gts, n_classes=2)["map50"] == pytest.approx(1.0)
    # detection too far for IoU>=0.5 -> no match -> AP 0 for that class
    dets[0]["boxes"] = [[1000, 1000, 1040, 1040]]
    assert map_from_detections(dets, gts, n_classes=2)["map50"] == pytest.approx(0.0)


def test_map_from_detections_fp_order_penalty():
    # FP ranked above the TP halves AP (classic VOC curve); TP-first keeps AP 1.0
    fp_first = [{"labels": [1, 1], "scores": [0.9, 0.8],
                 "boxes": [[500, 500, 540, 540], [10, 10, 50, 50]]}]
    tp_first = [{"labels": [1, 1], "scores": [0.9, 0.8],
                 "boxes": [[10, 10, 50, 50], [500, 500, 540, 540]]}]
    gts = [{"labels": [1], "boxes": [[10, 10, 50, 50]]}]
    assert map_from_detections(fp_first, gts, n_classes=2)["map50"] == pytest.approx(0.5)
    assert map_from_detections(tp_first, gts, n_classes=2)["map50"] == pytest.approx(1.0)


def test_map_from_detections_per_class_meaning():
    # class 0 perfect, class 1 has GT but no detections -> mAP is the mean 0.5, and
    # the per-class P/R average is also dragged to 0.5 by the empty class
    dets = [{"labels": [0], "scores": [0.9], "boxes": [[0, 0, 10, 10]]}]
    gts = [{"labels": [0, 1], "boxes": [[0, 0, 10, 10], [20, 20, 30, 30]]}]
    out = map_from_detections(dets, gts, n_classes=2)
    assert out["map50"] == pytest.approx(0.5)
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)


def test_map_from_detections_greedy_no_double_match():
    # two detections, one GT -> only the higher-scored one can match; second is FP
    dets = [{"labels": [1, 1], "scores": [0.9, 0.85],
             "boxes": [[10, 10, 50, 50], [10, 10, 50, 50]]}]
    gts = [{"labels": [1], "boxes": [[10, 10, 50, 50]]}]
    out = map_from_detections(dets, gts, n_classes=2)
    assert out["map50"] == pytest.approx(1.0)       # FP comes after full recall
    assert out["precision"] == pytest.approx(1.0)   # max-F1 point is at the TP
    assert out["recall"] == pytest.approx(1.0)
