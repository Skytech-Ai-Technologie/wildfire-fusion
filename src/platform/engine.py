"""Perceptual tier for the live platform.

Runs a trained checkpoint on frames coming off the PC camera and returns
detections + the P5 embedding the contextual tier consumes.

Two checkpoint formats are supported and probed automatically:

  * **dual**  — a raw `DualStreamYOLOv8n` state_dict (runs\\dual_*/best.pt). Input is
    a 6-channel tensor: [0:3] RGB and [3:6] LWIR. A PC webcam only provides one
    modality, so the LWIR half is filled with the **grayscale proxy of the RGB
    frame** — exactly the documented placeholder the training path uses on
    RGB-only datasets (see src/perceptual/dual_stream_yolo.py and the assumption
    register in the README). It is NOT thermal data.
  * **single** — an Ultralytics checkpoint (runs\\single_*/weights/best.pt). RGB only.

Two preprocessing details are load-bearing and easy to get wrong; both are pinned
by tests/test_platform.py:

  1. **Channel order is BGR.** The custom training loop fed `cv2.imread()` output
     straight into the network (no BGR->RGB swap), so the weights expect BGR in
     channels 0:3. Ultralytics' own pipeline does the opposite.
  2. **Letterbox padding is 114 gray**, centred, same aspect-preserving scale rule
     as `src/train_perceptual.py:_letterbox`.
"""
import time

import numpy as np
import torch

from src.platform.camera import _cv2

CLASSES = ["smoke", "fire"]


def letterbox(im, new):
    """Aspect-preserving resize + centered 114-gray pad.

    Mirrors `src/train_perceptual.py:_letterbox` exactly — keep the two in sync
    (tests/test_platform.py asserts they agree).
    """
    cv2 = _cv2()

    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = round(h * r), round(w * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, im.shape[2]), 114, dtype=im.dtype)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = im
    return canvas, r, (left, top)


def _decode(y, conf_thres, iou_thres, nc, classes=CLASSES):
    """Head output (B, 4+nc, N) -> list of per-image detection dicts."""
    from torchvision.ops import nms

    out = []
    for b in range(y.shape[0]):
        xywh = y[b, :4].T                       # (N, 4) cxcywh in letterboxed px
        cls_p = y[b, 4:].T                      # (N, nc) sigmoid probs
        scores, labels = cls_p.max(dim=1)
        x1y1 = xywh[:, :2] - xywh[:, 2:] / 2
        x2y2 = xywh[:, :2] + xywh[:, 2:] / 2
        boxes = torch.cat([x1y1, x2y2], dim=1)
        keep = scores > conf_thres
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if boxes.numel():
            k = nms(boxes, scores, iou_thres)
            boxes, scores, labels = boxes[k], scores[k], labels[k]
        dets = []
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            name = classes[int(label)] if int(label) < len(classes) else str(int(label))
            dets.append({"box": [float(v) for v in box], "score": float(score),
                         "label": int(label), "name": name})
        out.append(dets)
    return out


class PerceptionEngine:
    """Wraps a trained perceptual checkpoint for single-frame, live use."""

    def __init__(self, weights, device="auto", imgsz=640, conf=0.25, iou=0.45):
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.arch = None
        self.model = None
        self.nc = 2
        self._device = self._pick_device(device)
        self._hook = None
        self._p5 = {}
        self.lwir_source = "n/a"

    @staticmethod
    def _pick_device(device):
        if device != "auto":
            return device
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    # -- loading ------------------------------------------------------------ #
    def load(self):
        from src.perceptual.dual_stream_yolo import try_load_dual_stream

        model = try_load_dual_stream(self.weights, self._device)
        if model is not None:
            self.arch = "dual"
            self.model = model
            self.nc = model.nc
            self.lwir_source = "grayscale proxy of the RGB frame (NOT thermal)"
            # P5 = SPPF output = absolute layer 9 = shared[6] (shared starts at layer 3)
            self._hook = self.model.shared[6].register_forward_hook(
                lambda _m, _i, o: self._p5.__setitem__("p5", o))
            return self

        # fall back to an Ultralytics checkpoint (single-stream / baseline)
        from ultralytics import YOLO

        det = YOLO(self.weights)
        self.model = det.model.to(self._device).eval()
        self.arch = "single"
        self.lwir_source = "n/a (single-stream RGB model)"
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            self.nc = len(names)
            self.classes = [names.get(i, str(i)) for i in range(len(names))]
        else:
            self.nc = 2
            self.classes = list(CLASSES)
        self._hook = self.model.model[9].register_forward_hook(
            lambda _m, _i, o: self._p5.__setitem__("p5", o))
        return self

    def set_thresholds(self, conf=None, iou=None):
        if conf is not None:
            self.conf = float(max(0.01, min(0.99, conf)))
        if iou is not None:
            self.iou = float(max(0.05, min(0.95, iou)))

    # -- inference ---------------------------------------------------------- #
    @torch.inference_mode()
    def infer(self, bgr_frame):
        """bgr_frame: HxWx3 uint8 BGR. Returns a dict with detections, the P5
        embedding and the raw confidence signals."""
        cv2 = _cv2()

        t0 = time.perf_counter()
        h, w = bgr_frame.shape[:2]

        if self.arch == "dual":
            rgb_lb, r, (dx, dy) = letterbox(bgr_frame, self.imgsz)
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            lwir_lb, _, _ = letterbox(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), self.imgsz)
            # BGR order in channels 0:3 — matches the training loop (see module docstring)
            img = np.concatenate([rgb_lb, lwir_lb], axis=2)
        else:
            rgb_lb, r, (dx, dy) = letterbox(bgr_frame, self.imgsz)
            img = rgb_lb

        x = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        x = x.unsqueeze(0).to(self._device)

        self._p5.clear()
        out = self.model(x)
        y = out[0] if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]) else out
        dets = _decode(y, self.conf, self.iou, self.nc,
                       getattr(self, "classes", CLASSES))[0]

        # map boxes back from letterboxed space to the original frame
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            d["box"] = [max(0.0, min(float(w), (x1 - dx) / r)),
                        max(0.0, min(float(h), (y1 - dy) / r)),
                        max(0.0, min(float(w), (x2 - dx) / r)),
                        max(0.0, min(float(h), (y2 - dy) / r))]

        pi_head = float(y[0, 4:].max()) if y.numel() else 0.0
        pi_detect = max([d["score"] for d in dets], default=0.0)

        emb = None
        p5 = self._p5.get("p5")
        if p5 is not None:
            emb = p5.mean(dim=(2, 3)).squeeze(0).detach().cpu().numpy().astype(np.float32)

        return {
            "detections": dets,
            "embedding": emb,
            "pi_head": pi_head,
            "pi_detect": pi_detect,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "frame_size": [int(w), int(h)],
        }

    def close(self):
        if self._hook is not None:
            try:
                self._hook.remove()
            except Exception:  # noqa: BLE001
                pass
            self._hook = None
        self.model = None
