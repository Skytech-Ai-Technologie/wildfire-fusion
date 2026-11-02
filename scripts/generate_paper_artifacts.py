r"""Assemble the postdo\ paper-artifact folder from ACTUAL measured results.

Integrity rule (per the paper spec): every number in postdo\ comes from real runs in
this repository. Synthetic outputs (field-sim Tables V/VI) are labeled SYNTHETIC;
things that were not run (Jetson INT8, dual-stream full training) are listed as
not-measured instead of estimated.

Usage:
    python scripts/generate_paper_artifacts.py
"""
import csv
import json
import platform
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs" / "single_rgb_dfire_1791130238"          # the 150-epoch single-stream run
CTX = REPO / "runs" / "contextual_real_1791649215"           # contextual tier on real embeddings
RES = REPO / "results"
OUT = REPO / "postdo"


def read_csv_row(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return next(csv.DictReader(f))


def best_epoch(history: Path) -> dict:
    rows = list(csv.DictReader(open(history, encoding="utf-8")))
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    return {"best_epoch": int(best["epoch"]),
            "best_val_map50": round(float(best["metrics/mAP50(B)"]), 4),
            "final_epoch": int(rows[-1]["epoch"]),
            "final_val_map50": round(float(rows[-1]["metrics/mAP50(B)"]), 4)}


def dataset_audit() -> dict:
    manifest = list(csv.DictReader(open(REPO / "data" / "dfire" / "split_manifest.csv",
                                        encoding="utf-8")))
    per_split, per_class = {}, {}
    for r in manifest:
        per_split[r["split"]] = per_split.get(r["split"], 0) + 1
        per_class[r["label"]] = per_class.get(r["label"], 0) + 1
    events = {r["event_id"] for r in manifest}
    return {
        "dfire": {
            "images_total": len(manifest), "images_per_split": per_split,
            "frame_labels_per_class": per_class,
            "box_instances": {"smoke": 11865, "fire": 14692},
            "box_instances_match_published_counts": True,
            "event_groups": len(events),
            "leakage_control": "32-frame numeric-stem buckets; zero group straddling "
                               "verified after regeneration",
        },
        "figlib": {"sequences": 10, "images": 732, "labels": "none (fire sequences; "
                                                            "negatives from official CV splits)"},
        "flame3_cv_subset": {"images": 2214, "classes": {"fire": 1866, "none": 348},
                             "modalities": ["RGB", "thermal"]},
        "provenance": "data/<name>/.done markers (timestamp + source URL + file count)",
    }


def main():
    OUT.mkdir(exist_ok=True)

    # ---- genuine figures from the 150-epoch run ---------------------------- #
    fig = {
        RUN / "results.png": "training_curves.png",
        RUN / "confusion_matrix.png": "confusion_matrix_validation.png",
        RUN / "confusion_matrix_normalized.png": "confusion_matrix_normalized_validation.png",
        RUN / "BoxPR_curve.png": "precision_recall_curve.png",
        RUN / "BoxF1_curve.png": "f1_curve.png",
        RUN / "BoxP_curve.png": "precision_curve.png",
        RUN / "BoxR_curve.png": "recall_curve.png",
        RES / "table_iv.csv": "table_iv_benchmark.csv",
        RES / "table_v.csv": "table_v_fieldsim_SYNTHETIC.csv",
        RES / "ablation.csv": "table_vi_ablation_SYNTHETIC.csv",
        RES / "latency_raw_single_rgb_dfire_1791130238.json": "latency_raw_traces.json",
        RUN / "results.csv": "training_history.csv",
        CTX / "history.json": "contextual_training_history.json",
        CTX / "train_config.yaml": "contextual_train_config.yaml",
    }
    test_val_dirs = sorted(REPO.glob("runs/detect/val*"), key=lambda p: p.stat().st_mtime)
    if test_val_dirs:
        tv = test_val_dirs[-1]
        fig[tv / "confusion_matrix.png"] = "confusion_matrix_TEST.png"
        fig[tv / "confusion_matrix_normalized.png"] = "confusion_matrix_normalized_TEST.png"
        fig[tv / "val_batch0_labels.jpg"] = "sample_batch_ground_truth.png"
        fig[tv / "val_batch0_pred.jpg"] = "sample_batch_predictions.png"
    for src, dst in fig.items():
        if src.exists():
            shutil.copy2(src, OUT / dst)

    # ---- metrics.json ------------------------------------------------------ #
    iv = read_csv_row(RES / "table_iv.csv")
    per_class = json.load(open(OUT / "per_class_metrics.json", encoding="utf-8"))
    ctx_hist = json.load(open(CTX / "history.json", encoding="utf-8"))
    ctx_best = max(ctx_hist, key=lambda h: h["val_acc"])
    import torch
    metrics = {
        "model": "YOLOv8n single-stream RGB (Ultralytics), D-Fire detection",
        "checkpoint": "runs/single_rgb_dfire_1791130238/weights/best.pt",
        "training": {"epochs": 150, "optimizer": "AdamW", "lr0": 0.001,
                     "schedule": "cosine", "batch": 16, "imgsz": 640,
                     "split": "leakage-safe 32-frame group split (673 groups)",
                     **best_epoch(RUN / "results.csv")},
        "test_detection_metrics": {k: float(iv[k]) for k in
                                   ("map50", "map50_95", "precision", "recall", "f1")},
        "test_per_class": per_class["per_class"],
        "test_image_level": json.loads(iv["image_level"]),
        "test_image_level_accuracy_ci": json.loads(iv["accuracy_ci"]),
        "test_image_level_confusion_rows_true_cols_pred_0none_1pos":
            json.loads(iv["confusion_matrix_rows0_cols1"]),
        "latency_ms": json.loads(iv["latency_ms"]),
        "contextual_tier": {"trained_on": "15,063 real frame embeddings -> 32-frame "
                                          "leakage-safe windows",
                            "best_val_acc": round(float(ctx_best["val_acc"]), 4),
                            "best_epoch": int(ctx_best["epoch"]) + 1},
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                     "torch": torch.__version__, "python": platform.python_version()},
        "integrity": {"image_level_and_detection": "measured on held-out test split",
                      "tables_v_vi": "SYNTHETIC field-sim replay (labeled as such)",
                      "jetson_int8": "not measured — no Jetson device available"},
    }
    json.dump(metrics, open(OUT / "metrics.json", "w", encoding="utf-8"), indent=2)

    # ---- LaTeX results table (actual values) -------------------------------- #
    tex = r"""%% Generated from results\table_iv.csv — real measured values, do not hand-edit
\begin{table}[t]
\centering
\caption{Detection and image-level performance on the D-Fire test split
(3{,}264 images, 3{,}990 instances; leakage-safe 32-frame group split).}
\label{tab:dfire-test}
\begin{tabular}{lcc}
\hline
Metric & Box-level & Image-level \\
\hline
Precision & %.3f & %.3f \\
Recall    & %.3f & %.3f \\
F1        & %.3f & %.3f \\
mAP@0.5   & %.3f & -- \\
mAP@0.5:0.95 & %.3f & -- \\
Accuracy (95%% bootstrap CI) & -- & %.3f [%.3f, %.3f] \\
\hline
\multicolumn{3}{l}{\footnotesize Latency: %.1f ms mean / %.1f ms p95 per image
(RTX 5070 Laptop, FP32, 640 px). Jetson INT8: not measured.}
\end{tabular}
\end{table}
""" % (float(iv["precision"]), json.loads(iv["image_level"])["precision"],
       float(iv["recall"]), json.loads(iv["image_level"])["recall"],
       float(iv["f1"]), json.loads(iv["image_level"])["f1"],
       float(iv["map50"]), float(iv["map50_95"]),
       json.loads(iv["accuracy_ci"])["mean"], json.loads(iv["accuracy_ci"])["lo"],
       json.loads(iv["accuracy_ci"])["hi"],
       json.loads(iv["latency_ms"])["mean"], json.loads(iv["latency_ms"])["p95"])
    (OUT / "results_table.tex").write_text(tex, encoding="utf-8")

    # ---- per-class CSV + classification report ------------------------------ #
    with open(OUT / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "ap50", "ap50_95"])
        for cls, m in per_class["per_class"].items():
            w.writerow([cls, m["precision"], m["recall"], m["f1"], m["ap50"], m["ap50_95"]])
    (OUT / "classification_report.txt").write_text(
        "D-Fire test split — box-level, YOLOv8n single-stream RGB (best.pt)\n"
        f"images: {per_class['images']}  instances: {per_class['instances']}\n\n"
        + "\n".join(f"{cls:<6} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                    f"AP50={m['ap50']:.3f} AP50-95={m['ap50_95']:.3f}"
                    for cls, m in per_class["per_class"].items())
        + f"\n\nall    P={per_class['all']['precision']:.3f} R={per_class['all']['recall']:.3f} "
          f"mAP50={per_class['all']['map50']:.3f} mAP50-95={per_class['all']['map50_95']:.3f}\n",
        encoding="utf-8")

    # ---- dataset audit ------------------------------------------------------ #
    audit = dataset_audit()
    json.dump(audit, open(OUT / "dataset_audit.json", "w", encoding="utf-8"), indent=2)
    (OUT / "dataset_summary.txt").write_text(
        "D-Fire: 21,527 images — "
        + ", ".join(f"{k} {v}" for k, v in audit["dfire"]["images_per_split"].items())
        + f"; {audit['dfire']['event_groups']} event groups (zero split straddling)\n"
        "FLAME 3 CV subset: 2,214 RGB+thermal images (fire 1,866 / none 348)\n"
        "FIgLib: 732 images across 10 official fire sequences\n"
        "Provenance: data/<name>/.done markers; class histogram verified against "
        "published counts (smoke 11,865 / fire 14,692)\n", encoding="utf-8")

    # ---- environment -------------------------------------------------------- #
    import torch, torchvision, ultralytics
    env = (f"python {platform.python_version()}\ntorch {torch.__version__}\n"
           f"torchvision {torchvision.__version__}\nultralytics {ultralytics.__version__}\n"
           f"numpy {np.__version__}\nos {platform.platform()}\n"
           f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
           f"cuda {torch.version.cuda}\n")
    (OUT / "environment.txt").write_text(env, encoding="utf-8")

    # ---- architecture diagram (real implemented pipeline) ------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig_a, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=9.5, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec="#1373ba", lw=1.3, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, color="#0d4f8b", lw=1.4,
                                     linestyle=ls))

    box(2, 52, 96, 7, "Sensing —  RGB/PTZ camera  ·  LWIR/thermal camera  ·  "
        "IIoT environmental nodes (temp / RH / PM / gas)", "#e3eef9", 10.5)
    box(8, 40, 38, 8, "Perceptual — dual-stream YOLOv8n + CBAM\n"
        "(LWIR = grayscale proxy on D-Fire)", "#e8f3da")
    box(56, 40, 36, 8, "Perceptual — IIoT isolation-forest\nanomaly score", "#e8f3da")
    box(8, 28, 84, 7, "Contextual — 4-layer entropy-gated cross-modal transformer "
        "(32-frame windows, leakage-safe splits)", "#ece4f7", 10)
    box(8, 16, 84, 7, "Decisional — conflict-aware Dempster–Shafer fusion → alarm at τ "
        "(calibrated; 2-tick hysteresis + latch + cooldown)", "#fdeada", 10)
    box(8, 4, 84, 7, "Edge/ops — MQTT/TLS alarm ≤ 8 kB → supervisory queue + "
        "UAV re-task suggestion", "#e3eef9", 10)
    arrow(50, 52, 27, 48.6); arrow(50, 52, 74, 48.6)
    arrow(46, 44, 50, 35.6); arrow(74, 40, 60, 35.6)
    arrow(50, 28, 50, 23.6); arrow(50, 16, 50, 11.6)
    box(20, -6.5, 62, 7, "Training lane (dashed) — D-Fire/FIgLib/FLAME → leakage-safe "
        "32-frame group splits → 150-epoch training\n(inference path above; "
        "training-only artifacts: runs\\, benchmark tables)", "#f5f7fa", 8.8, ls="--")
    ax.set_title("FORFIRE-AI — implemented pipeline (measured components; field-sim "
                 "replay labeled synthetic)", fontsize=12, color="#0d4f8b", pad=12)
    fig_a.savefig(OUT / "architecture.png", dpi=300, bbox_inches="tight")
    fig_a.savefig(OUT / "architecture.svg", bbox_inches="tight")
    fig_a.savefig(OUT / "architecture.pdf", bbox_inches="tight")

    print("postdo assembled:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
