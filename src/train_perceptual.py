"""§4 — Train the perceptual tier.

Paths:
  --arch single  --mode rgb|lwir   Ultralytics YOLOv8n (workhorse; runs today)
  --arch dual                      Dual-stream RGB+LWIR YOLOv8n+CBAM (custom loop)
  --arch baseline --model yolov5su|yolov8n   paper baselines under identical splits

Datasets: dfire (detection, YOLO boxes) is the smoke test; flame/figlib train as
classification (class-dir trees built by build_splits.py --copy or manual arrange).

Dual-stream LWIR source: data/<ds>/lwir/ paired frames when present, else a grayscale
proxy of the RGB frame (documented placeholder — swap in FLAME2 RGB/IR pairs for a real
two-modal input).

Examples:
  python src/train_perceptual.py --dataset dfire --arch single --mode rgb
  python src/train_perceptual.py --dataset dfire --arch dual --epochs 3     # smoke test
  python src/train_perceptual.py --dataset dfire --arch baseline --model yolov5su
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.perceptual.dual_stream_yolo import DualStreamYOLOv8n, _HAS_ULTRALYTICS  # noqa: E402
from src.perceptual.losses import WeightedV8DetectionLoss  # noqa: E402

CLASSES = ["smoke", "fire"]
EXTS = (".jpg", ".jpeg", ".png")


def load_cfg():
    return yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))


def dataset_img_list(ds: str, split: str) -> list:
    p = REPO / "data" / ds / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run scripts/build_splits.py --dataset {ds}")
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Ultralytics single-stream / baseline path                                    #
# --------------------------------------------------------------------------- #
def train_single(args, cfg) -> Path:
    from ultralytics import YOLO
    data_yaml = REPO / "data" / args.dataset / "dataset.yaml"
    model_name = {"yolov8n": "yolov8n.pt", "yolov5su": "yolov5su.pt"}.get(
        args.model or "yolov8n", args.model)
    if args.model == "yolov7-tiny":
        print("NOTE: YOLOv7-tiny is not shipped with Ultralytics. Train it from "
              "https://github.com/WongKinYiu/yolov7 under the SAME split lists in "
              "data/<ds>/{train,val,test}.txt to keep the paper's parity requirement, "
              "or rerun with --model yolov5su / yolov8n.")
        sys.exit(1)
    if args.mode == "lwir":
        print("WARNING: single-stream LWIR on an RGB-only dataset trains on grayscale "
              "images; pair FLAME2 for a real LWIR run.")

    p = cfg["perceptual_training"]
    aug = p["augmentation"]
    model = YOLO(model_name)
    name = f"single_{args.mode}_{args.dataset}_{int(time.time())}"
    model.train(
        data=str(data_yaml), epochs=args.epochs or p["epochs"], imgsz=cfg["imgsz"],
        batch=p["batch_size"], optimizer=p["optimizer"], lr0=p["lr0"],
        cos_lr=(p["lr_schedule"] == "cosine"), seed=p["seed"],
        hsv_h=aug["hsv_h"], hsv_s=aug["hsv_s"], hsv_v=aug["hsv_v"], mosaic=aug["mosaic"],
        flipud=aug["flipud"], fliplr=aug["fliplr"], degrees=aug["degrees"],
        translate=aug["translate"], scale=aug["scale"],
        project=str(REPO / "runs"), name=name, exist_ok=True,
    )
    out = REPO / "runs" / name
    (out / "train_config.yaml").write_text(yaml.safe_dump(
        {"overrides": {"epochs": args.epochs or p["epochs"], "model": model_name,
                       "arch": "single", "mode": args.mode, "dataset": args.dataset},
         "defaults": p}), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Dual-stream custom loop                                                      #
# --------------------------------------------------------------------------- #
def _letterbox(im, new):
    import cv2
    import numpy as np
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = round(h * r), round(w * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, im.shape[2]), 114, dtype=im.dtype)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = im
    return canvas, r, (left, top)


def _load_yolo_labels(lbl_path: Path):
    import numpy as np
    if not lbl_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            parts = line.split()
            rows.append([float(x) for x in parts[:5]])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


class DualStreamYoloDataset:
    """Yields 6-channel images + ultralytics-style labels. LWIR = paired frame from
    data/<ds>/lwir/ when present, else grayscale proxy of RGB (documented)."""

    def __init__(self, img_list: list, lwir_dir: Path | None, imgsz=640):
        import numpy as np
        self.items = [Path(p) for p in img_list]
        self.lwir_dir = lwir_dir
        self.imgsz = imgsz

    def __len__(self):
        return len(self.items)

    def _find_lwir(self, p: Path):
        import cv2
        if self.lwir_dir is not None:
            for ext in (".jpg", ".jpeg", ".png"):
                q = self.lwir_dir / (p.stem + ext)
                if q.exists():
                    im = cv2.imread(str(q), cv2.IMREAD_COLOR)
                    if im is not None:
                        return im, False
        rgb = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if rgb is None:
            raise IOError(f"unreadable image: {p}")
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), True

    def __getitem__(self, i: int):
        import cv2
        import numpy as np
        cv2.setNumThreads(0)   # spawned workers: no per-thread buffer pools ballooning RSS
        p = self.items[i]
        rgb = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if rgb is None:
            raise IOError(f"unreadable image: {p}")
        lwir, used_proxy = self._find_lwir(p)
        rgb_lb, r, (dx, dy) = _letterbox(rgb, self.imgsz)
        lwir_lb, _, _ = _letterbox(lwir, self.imgsz)
        # uint8 end-to-end: the float32 copy (9.4 MB per item) made worker RSS balloon
        # to ~9 GB each on Windows; normalization happens on the GPU instead
        img = np.concatenate([rgb_lb, lwir_lb], axis=2)
        img = torch.from_numpy(img).permute(2, 0, 1)          # (6, H, W) uint8

        h, w = rgb.shape[:2]
        # ultralytics convention: labels/<...> mirrors images/<...> (same stem);
        # fall back to a .txt sitting next to the image for ad-hoc layouts.
        lbl = p.with_suffix(".txt").resolve()
        if "images" in lbl.parts:
            mirror = Path(*("labels" if part == "images" else part for part in lbl.parts))
            if mirror.exists():
                lbl = mirror
        labels = _load_yolo_labels(lbl)
        if len(labels):
            if labels.shape[1] < 5:
                labels = np.pad(labels, ((0, 0), (0, 5 - labels.shape[1])))
            cls = labels[:, 0:1]
            cx = (labels[:, 1] * w * r + dx) / self.imgsz
            cy = (labels[:, 2] * h * r + dy) / self.imgsz
            bw = labels[:, 3] * w * r / self.imgsz
            bh = labels[:, 4] * h * r / self.imgsz
            bboxes = np.stack([cx, cy, bw, bh], axis=1)
        else:
            cls = np.zeros((0, 1), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)
        return {"img": img, "cls": torch.from_numpy(cls).float(),
                "bboxes": torch.from_numpy(bboxes).float(),
                "im_file": str(p), "lwir_proxy": used_proxy}


def collate(batch):
    imgs = torch.stack([b["img"] for b in batch])
    cls, bboxes, batch_idx = [], [], []
    for i, b in enumerate(batch):
        cls.append(b["cls"])
        bboxes.append(b["bboxes"])
        batch_idx.append(torch.full((b["cls"].shape[0],), i, dtype=torch.float32))
    return {"img": imgs,
            "cls": torch.cat(cls) if cls else torch.zeros(0, 1),
            "bboxes": torch.cat(bboxes) if bboxes else torch.zeros(0, 4),
            "batch_idx": torch.cat(batch_idx) if batch_idx else torch.zeros(0),
            "im_file": [b["im_file"] for b in batch]}


def train_dual(args, cfg, resume_dir: Path | None = None) -> Path:
    if not _HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics required for the dual-stream model")
    try:
        from torch.utils.data import DataLoader
    except ImportError:
        raise

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = cfg["perceptual_training"]
    train_list = dataset_img_list(args.dataset, "train")
    val_list = dataset_img_list(args.dataset, "val")
    lwir_dir = REPO / "data" / args.dataset / "lwir"
    n_proxy = 0

    train_ds = DualStreamYoloDataset(train_list, lwir_dir if lwir_dir.exists() else None,
                                     imgsz=cfg["imgsz"])
    n_wkr = p.get("num_workers", 0)
    dl = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True,
                    collate_fn=collate,
                    num_workers=n_wkr,                # 0: single-threaded
                    pin_memory=(dev == "cuda"),
                    persistent_workers=False)   # respawn each epoch: caps worker RSS growth
    val_ds = (DualStreamYoloDataset(val_list, lwir_dir if lwir_dir.exists() else None,
                                    imgsz=cfg["imgsz"]) if val_list else None)
    # val workers are transient (persistent_workers=False): the train loader's workers
    # stay alive between epochs, and both sets alive at once OOM'd 32 GB on Windows
    val_dl = (DataLoader(val_ds, batch_size=p["batch_size"], shuffle=False,
                         collate_fn=collate,
                         num_workers=n_wkr,
                         pin_memory=(dev == "cuda"),
                         persistent_workers=False) if val_ds else None)

    model = DualStreamYOLOv8n(nc=len(CLASSES), pretrained="yolov8n.pt",
                              use_cbam=not args.no_cbam).to(dev)
    criterion = WeightedV8DetectionLoss(model)   # caches model device — create AFTER .to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr0"])
    epochs = args.epochs or p["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if resume_dir is not None:
        out = Path(resume_dir)
    else:
        name = f"dual_{args.dataset}_{int(time.time())}"
        out = REPO / "runs" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_config.yaml").write_text(yaml.safe_dump({
            "arch": "dual", "dataset": args.dataset, "nc": len(CLASSES),
            "use_cbam": not args.no_cbam, "optimizer": "AdamW", "lr0": p["lr0"],
            "lr_schedule": "cosine", "epochs": epochs, "batch_size": p["batch_size"],
            "seed": p["seed"], "imgsz": cfg["imgsz"],
            "lwir_source": "paired data/<ds>/lwir/ dir if present, else grayscale proxy",
            "loss": "Ldet = 1.0*CIoU(+DFL) + 0.5*focal/BCE(TAL) + 0.7*obj BCE (eq. 2, see losses.py)",
        }), encoding="utf-8")

    start_epoch, best = 0, float("inf")
    if resume_dir is not None:
        bundle = torch.load(out / "resume.pt", map_location=dev, weights_only=True)
        model.load_state_dict(bundle["model"])
        opt.load_state_dict(bundle["optimizer"])
        sched.load_state_dict(bundle["scheduler"])
        start_epoch, best = int(bundle["epoch"]), float(bundle["best"])
        print(f"[dual] resuming {out.name} from epoch {start_epoch}/{epochs} "
              f"(best val so far {best:.4f})")
        if start_epoch >= epochs:
            print(f"[dual] WARNING: this run already reached its {epochs}-epoch target — "
                  "pass --epochs N to train longer")

    for epoch in range(start_epoch, epochs):
        model.train()
        running, nb = 0.0, 0
        for batch in dl:
            imgs = batch["img"].to(dev, non_blocking=True).float().div_(255.0)
            tgt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                loss, items = criterion(model(imgs), tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            running += float(loss)
            nb += 1
        sched.step()
        avg = running / max(nb, 1)

        # per-epoch validation loss drives best.pt selection (train loss alone
        # always decreases and says nothing about generalization)
        val_avg = None
        if val_dl is not None:
            model.eval()
            # Detect must stay in train mode: its eval-mode forward returns the
            # postprocessed (y, x) tuple, which the loss cannot consume
            model.head.train()
            v_running, v_nb = 0.0, 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(dev == "cuda")):
                for batch in val_dl:
                    imgs = batch["img"].to(dev, non_blocking=True).float().div_(255.0)
                    tgt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    vloss, _ = criterion(model(imgs), tgt)
                    v_running += float(vloss)
                    v_nb += 1
            val_avg = v_running / max(v_nb, 1)

        marker = ""
        if (val_avg if val_avg is not None else avg) < best:
            best = val_avg if val_avg is not None else avg
            torch.save(model.state_dict(), out / "best.pt")
            marker = "  <- best"
        torch.save(model.state_dict(), out / "last.pt")
        # full resume bundle: weights + optimizer + scheduler + progress + best val
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch + 1, "best": best},
                   out / "resume.pt")
        msg = f"epoch {epoch + 1}/{epochs}  loss {avg:.4f}"
        msg += f"  val {val_avg:.4f}" if val_avg is not None else \
               "  (no val list — best on train loss)"
        print(msg + marker)

    (out / "val_list.txt").write_text("\n".join(val_list), encoding="utf-8")
    print(f"done -> {out}  (best.pt = lowest {'val' if val_dl is not None else 'train'} "
          "loss; evaluate with benchmark/run_benchmark.py)")
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="§4 perceptual training")
    ap.add_argument("--dataset", default=None, choices=["flame", "dfire", "figlib"])
    ap.add_argument("--arch", choices=["single", "dual", "baseline"], default="single")
    ap.add_argument("--mode", choices=["rgb", "lwir"], default="rgb")
    ap.add_argument("--model", default=None, help="baseline: yolov5su|yolov8n")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-cbam", action="store_true", help="§11 ablation: train without CBAM")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR|auto",
                    help="resume an interrupted dual-stream run: pass its runs/<name> "
                         "folder, or 'auto' for the newest one with resume.pt")
    args = ap.parse_args()

    cfg = load_cfg()
    if args.resume:
        run = Path(args.resume)
        if args.resume == "auto":
            cands = sorted(REPO.glob("runs/dual_*/resume.pt"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not cands:
                sys.exit("no dual-stream run with resume.pt found under runs\\")
            run = cands[0].parent
        if not (run / "resume.pt").exists():
            sys.exit(f"{run} has no resume.pt — only the dual-stream custom loop "
                     "supports resume (single/baseline runs resume via their "
                     "weights\\last.pt with YOLO(...).train(resume=True))")
        rcfg = yaml.safe_load((run / "train_config.yaml").read_text(encoding="utf-8"))
        args.dataset = rcfg.get("dataset", args.dataset)
        args.no_cbam = not rcfg.get("use_cbam", True)
        print(f"[resume] dual-stream run {run.name} (dataset={args.dataset})")
        out = train_dual(args, cfg, resume_dir=run)
    else:
        if not args.dataset:
            ap.error("--dataset is required unless --resume is used")
        if args.arch in ("single", "baseline"):
            out = train_single(args, cfg)
        else:
            out = train_dual(args, cfg)
    print(f"artifacts -> {out}")


if __name__ == "__main__":
    main()
