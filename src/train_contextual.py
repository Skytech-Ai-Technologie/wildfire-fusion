"""§5 — Train the contextual tier (entropy-gated transformer).

ASSUMPTION (paper gap): the contextual tier is trained with BCE between the refined
per-modality confidence and the window's ground-truth fire label, using the same
leakage-safe splits as §3.4, optimizer AdamW, lr 5e-4, 30 epochs, early stopping on
val loss.

Modes:
  --synthetic             toy windows so the whole tier trains without datasets
                          (clearly labeled synthetic — for pipeline smoke tests only)
  --embeddings file.npz   arrays produced by extract_embeddings() / build_windows():
                          emb (N, T, D), conf (N, T), labels (N,), splits (N,) —
                          splits carries the manifest's leakage-safe train/val/test
                          assignment per window and is honored when present

Usage:
  python src/train_contextual.py --synthetic
  python src/train_contextual.py --embeddings runs/emb/windows.npz
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.contextual.gated_transformer import ContextualEncoder, ConcatFusionEncoder  # noqa: E402


def load_cfg():
    return yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Embedding extraction from a trained perceptual model                         #
# --------------------------------------------------------------------------- #
def extract_embeddings(weights: str, img_list: list, out_npz: str, imgsz=640):
    """Run the perceptual model over images; save GAP'd P5 embedding + max det score.

    Output npz: emb (N, D), conf (N,), im_files (list of str). Call build_windows() to
    group frames into T-length windows by event id (from split_manifest.csv).
    """
    from ultralytics import YOLO
    import cv2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLO(weights).model.to(dev).eval()
    feats = {}

    def hook(_m, _i, output):
        feats["p5"] = output

    det.model[9].register_forward_hook(hook)   # SPPF output = P5

    embs, confs, files = [], [], []
    with torch.no_grad():
        for p in img_list:
            im = cv2.imread(p)
            if im is None:
                continue
            im = cv2.resize(im, (imgsz, imgsz))
            x = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            out = det(x.unsqueeze(0).to(dev))
            emb = feats["p5"].mean(dim=(2, 3)).squeeze(0).cpu().numpy()   # (D,)
            embs.append(emb)
            # max detection confidence from eval-mode head output (y: (B, 4+nc, N))
            y = out[0] if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]) else None
            conf = float(y[0, 4:].max()) if y is not None and y.numel() else 0.0
            confs.append(conf)
            files.append(p)
    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=np.stack(embs), conf=np.asarray(confs), im_files=np.array(files))
    print(f"saved {len(embs)} frame embeddings -> {out}")
    return out


def build_windows(manifest_csv: str, emb_npz: str, out_npz: str, T=32):
    """Group frame-level embeddings into (N, T, D) windows by event_id from
    split_manifest.csv. Window label = max frame label (fire=2, smoke=1 > none=0 -> 1).
    Windows are truncated (not padded) to the last T frames of each event.

    The event's manifest split is carried through to the npz ('splits' array) so
    train() can honor the leakage-safe split instead of a random one. Frames are
    ordered by their stem number, because a group's frames may live in different
    mirror split dirs and the manifest rows are dir-major, not frame-ordered."""
    import csv as _csv
    embz = np.load(emb_npz, allow_pickle=True)
    by_file = {str(f): (e, c) for f, e, c in
               zip(embz["im_files"], embz["emb"], embz["conf"])}
    by_event = {}
    with open(manifest_csv, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            e, c = by_file.get(row["frame_path"], (None, None))
            if e is None:
                continue
            by_event.setdefault(row["event_id"], []).append(
                (row["split"], row["frame_path"], e, c, row["label"]))

    def _stem_num(fp):
        m = re.search(r"(\d+)$", Path(fp).stem)
        return int(m.group(1)) if m else 0

    embs, confs, labels, splits = [], [], [], []
    for ev_id, ev_rows in by_event.items():
        ev_rows.sort(key=lambda r: (_stem_num(r[1]), r[1]))
        ev_rows = ev_rows[-T:]
        if len(ev_rows) < T // 2:
            continue
        if len(ev_rows) < T:
            # left-pad by repeating the earliest frame so every window has exactly T
            # steps (np.stack needs uniform shapes; documented, not silent)
            ev_rows = [ev_rows[0]] * (T - len(ev_rows)) + ev_rows
        embs.append(np.stack([r[2] for r in ev_rows]))
        confs.append(np.asarray([r[3] for r in ev_rows], dtype=np.float32))
        labels.append(1 if any(r[4] in ("fire", "smoke") for r in ev_rows) else 0)
        ev_splits = {r[0] for r in ev_rows}
        if len(ev_splits) > 1:
            print(f"[windows] WARNING: event {ev_id} spans splits {sorted(ev_splits)} "
                  "— using its first frame's split")
        splits.append(ev_rows[0][0])
    out = Path(out_npz)
    np.savez(out, emb=np.stack(embs), conf=np.stack(confs),
             labels=np.asarray(labels, dtype=np.float32),
             splits=np.asarray(splits))
    from collections import Counter
    print(f"saved {len(labels)} windows (T={T}) -> {out} "
          f"(splits: {dict(Counter(splits))})")
    return out


# --------------------------------------------------------------------------- #
# Synthetic windows (smoke test; clearly labeled synthetic)                     #
# --------------------------------------------------------------------------- #
def make_synthetic_windows(n_windows=256, T=32, d_model=128, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n_windows).astype(np.float32)
    emb = rng.normal(0, 0.5, (n_windows, T + 2, d_model)).astype(np.float32)  # T frames + IIoT + LWIR
    conf = rng.uniform(0.05, 0.35, (n_windows, T + 2)).astype(np.float32)
    fire = labels.astype(bool)
    emb[fire] += 0.8                                   # separable signal along dim 0
    conf[fire] += rng.uniform(0.3, 0.5, (int(fire.sum()), T + 2))
    conf = np.clip(conf, 0.01, 0.99)
    return emb, conf, labels, d_model


def train(args, cfg):
    c = cfg["contextual"]
    T = c["temporal_window"]
    if args.synthetic:
        emb, conf, labels, d_emb = make_synthetic_windows(T=T, seed=args.seed)
        splits = None                      # synthetic windows are i.i.d. — random split
    else:
        z = np.load(args.embeddings, allow_pickle=True)
        emb, conf, labels = z["emb"], z["conf"], z["labels"]
        d_emb = emb.shape[-1]
        T = emb.shape[1]
        splits = z["splits"] if "splits" in z.files else None
        if splits is None:
            print("[contextual] WARNING: embeddings npz has no 'splits' array — it was "
                  "built by an older build_windows(); re-run it against "
                  "split_manifest.csv to train leakage-safe")

    enc_cls = ConcatFusionEncoder if args.concat_ablation else ContextualEncoder
    model = enc_cls(d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                    dropout=c["dropout"], in_dim=d_emb)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"])
    bce = nn.BCELoss()
    emb_t = torch.from_numpy(emb).float()
    conf_t = torch.from_numpy(conf).float()
    lab_t = torch.from_numpy(labels).float()

    n = len(labels)
    if splits is not None:
        tr_idx = np.where(splits == "train")[0]
        val_idx = np.where(splits == "val")[0]
        if not len(tr_idx) or not len(val_idx):
            print(f"[contextual] WARNING: manifest splits have no train or val windows "
                  f"(train={len(tr_idx)}, val={len(val_idx)}) — falling back to a "
                  "random 80/20 window split")
            splits = None
        else:
            held_out = int((splits == "test").sum())
            print(f"[contextual] leakage-safe split from manifest: {len(tr_idx)} train "
                  f"/ {len(val_idx)} val windows (test held out: {held_out})")
    if splits is None:
        idx = np.random.default_rng(args.seed).permutation(n)
        n_val = max(int(0.2 * n), 1)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

    name = f"contextual_{'synthetic' if args.synthetic else 'real'}_{int(time.time())}"
    out = REPO / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best_val, patience = float("inf"), c["early_stopping_patience"]

    for epoch in range(c["epochs"]):
        model.train()
        opt.zero_grad()
        refined, _ = model(emb_t[tr_idx], conf_t[tr_idx])
        prob = refined.mean(dim=1)
        loss = bce(prob, lab_t[tr_idx])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            rv, _ = model(emb_t[val_idx], conf_t[val_idx])
            val_loss = float(bce(rv.mean(dim=1), lab_t[val_idx]))
            acc = float(((rv.mean(dim=1) > 0.5) == (lab_t[val_idx] > 0.5)).float().mean())
        history.append({"epoch": epoch, "loss": float(loss.detach()), "val_loss": val_loss,
                        "val_acc": acc})
        marker = ""
        if val_loss < best_val:
            best_val, patience = val_loss, c["early_stopping_patience"]
            torch.save(model.state_dict(), out / "best.pt")
            marker = "  <- best"
        else:
            patience -= 1
        print(f"epoch {epoch + 1}/{c['epochs']}  loss {float(loss):.4f}  "
              f"val {val_loss:.4f}  acc {acc:.3f}{marker}")
        if patience <= 0:
            print("early stopping")
            break

    (out / "train_config.yaml").write_text(yaml.safe_dump({
        "mode": "synthetic" if args.synthetic else str(args.embeddings),
        "arch": type(model).__name__, "T": T, "in_dim": d_emb,
        **{k: c[k] for k in ("d_model", "n_heads", "n_layers", "dropout", "lr", "epochs")},
        "SYNTHETIC": bool(args.synthetic)}), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="§5 contextual training")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on toy windows (pipeline smoke test, labeled synthetic)")
    ap.add_argument("--embeddings", default=None, help="npz from build_windows()")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concat-ablation", action="store_true",
                    help="§11: swap the transformer for the concat+MLP baseline")
    args = ap.parse_args()
    if not args.synthetic and not args.embeddings:
        ap.error("pass --synthetic or --embeddings")
    train(args, load_cfg())


if __name__ == "__main__":
    main()
