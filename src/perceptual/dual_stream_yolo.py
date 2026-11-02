"""§4.2 — Dual-stream YOLOv8n + CBAM.

Design (documented ASSUMPTION — the source paper does not specify fusion depth):

  * RGB and LWIR each run their own copy of the YOLOv8n stem -> Conv -> stage-2 C2f,
    each followed by CBAM ("insert a CBAM module right after every C2f block output,
    both streams").
  * Fusion = channel-concat + 1x1 conv right after the two stage-2 C2f blocks, then
    CBAM once on the merged tensor.
  * Everything from the P3 stage onward (backbone, SPPF, neck, Detect head) is shared
    between streams, with CBAM after every remaining C2f block.
  * Built from Ultralytics' own DetectionModel modules (deep-copied, not reimplemented)
    so weights start from `yolov8n.pt` and the Detect head / TAL loss / export tooling
    stay compatible.

Input convention: a 6-channel tensor, channels [0:3] = RGB, [3:6] = LWIR. Single-channel
LWIR must be replicated to 3 channels upstream (see DualStreamYoloDataset in
train_perceptual.py). When no paired LWIR frame exists the dataset substitutes a
grayscale proxy and flags it, so the architecture is exercisable on RGB-only datasets
like D-Fire — swap in FLAME2 RGB/IR pairs for a real two-modal input.
"""
import copy
import re
from pathlib import Path

import torch
import torch.nn as nn
import yaml

try:
    from ultralytics import YOLO
    from ultralytics.nn.modules import Conv, Detect
    from ultralytics.nn.tasks import DetectionModel
    _HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    _HAS_ULTRALYTICS = False

from .cbam import CBAM

# v8n layer indices duplicated per stream: stem (0), first conv (1), stage-2 C2f (2)
STREAM_LAYERS = (0, 1, 2)
# v8n graph (absolute layer index -> `from` spec), verbatim from the yolov8.yaml head.
SHARED_FROM = {
    3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1,
    11: [-1, 6], 12: -1, 13: -1, 14: [-1, 4], 15: -1, 16: -1,
    17: [-1, 12], 18: -1, 19: -1, 20: [-1, 9], 21: -1, 22: [15, 18, 21],
}
# absolute indices whose outputs are consumed later -> must be cached during forward
CACHE_INDICES = (4, 6, 9, 12, 15, 18, 21)
# C2f outputs that get CBAM (backbone P3/P4/P5 + every head C2f, per spec §4.2)
CBAM_INDICES = (4, 6, 8, 12, 15, 18, 21)


def _require_ultralytics():
    if not _HAS_ULTRALYTICS:
        raise RuntimeError("Ultralytics is required for DualStreamYOLOv8n: pip install ultralytics")


def _copy_conv_weights(dst_conv, src_conv):
    if dst_conv.weight.shape != src_conv.weight.shape:
        return False
    with torch.no_grad():
        dst_conv.weight.copy_(src_conv.weight)
        if dst_conv.bias is not None and src_conv.bias is not None:
            dst_conv.bias.copy_(src_conv.bias)
    return True


def _conv2ds(module):
    """All nn.Conv2d in deterministic registration order (structure differs across
    Ultralytics versions — 8.4 nests cv3 one level deeper than 8.2)."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            yield m


def _rehead(base_model, nc):
    """Swap the 80-class Detect head for an nc-class one, transferring the pretrained
    box branch (class-independent) and every cls-trunk conv whose shape matches."""
    old = base_model.model[-1]
    ch = tuple(m[0].conv.in_channels for m in old.cv2)   # per-level in-channels
    head = Detect(nc=nc, ch=ch)
    head.cv2.load_state_dict(old.cv2.state_dict())       # box branch: nc-independent
    for new_m, old_m in zip(_conv2ds(head.cv3), _conv2ds(old.cv3)):
        _copy_conv_weights(new_m, old_m)                 # skips the nc-sized final convs
    head.stride = old.stride
    base_model.model[-1] = head
    return base_model


def _load_base(nc, pretrained):
    _require_ultralytics()
    if pretrained:
        base = YOLO(pretrained).model                    # DetectionModel, nc=80
        if base.model[-1].nc != nc:
            base = _rehead(base, nc)
        return base
    return DetectionModel("yolov8n.yaml", ch=3, nc=nc, verbose=False)


class DualStreamYOLOv8n(nn.Module):
    def __init__(self, nc=2, pretrained="yolov8n.pt", use_cbam=True):
        super().__init__()
        base = _load_base(nc, pretrained)
        L = list(base.model)                             # DetectionModel.model: 23 modules, abs index == position
        self.nc = nc
        self.use_cbam = use_cbam

        c_stage2 = L[3].conv.in_channels                 # stage-2 C2f output channels (32 for v8n)

        def make_stream():
            mods = [copy.deepcopy(L[i]) for i in STREAM_LAYERS]
            if use_cbam:
                mods.append(CBAM(c_stage2))
            return nn.Sequential(*mods)

        # LWIR stream is initialised from the RGB stem weights (ASSUMPTION: LWIR is a
        # low-res grayscale modality; the copy converges faster than random init).
        self.rgb_stream = make_stream()
        self.lwir_stream = make_stream()
        self.fuse_conv = Conv(2 * c_stage2, c_stage2, 1, 1)   # channel-concat + 1x1 conv
        self.cbam_fuse = CBAM(c_stage2) if use_cbam else nn.Identity()

        self.shared = nn.ModuleList([copy.deepcopy(L[i]) for i in range(3, 23)])
        self.head = self.shared[-1]                      # Detect
        self.stride = base.stride
        self.names = {0: "smoke", 1: "fire"} if nc == 2 else {i: str(i) for i in range(nc)}
        # Placeholder gains — the live loss (WeightedV8DetectionLoss) carries the paper's λs.
        self.args = {"box": 7.5, "cls": 0.5, "dfl": 1.5}

        self.cbam = nn.ModuleDict()
        if use_cbam:
            for i in CBAM_INDICES:
                self.cbam[str(i)] = CBAM(L[i].cv2.conv.out_channels)

        # Ultralytics 8.3+ loads checkpoint weights with requires_grad=False (EMA
        # convention); this model is meant to be trained, so unfreeze everything.
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        """x: (B, 6, H, W) — channels [0:3] RGB, [3:6] LWIR (replicate 1-ch LWIR upstream).

        Train mode returns the raw per-level maps (compatible with v8DetectionLoss);
        eval mode returns Ultralytics' postprocessed (y, x) tuple.
        """
        r = self.rgb_stream(x[:, 0:3])
        l = self.lwir_stream(x[:, 3:6])
        cur = self.cbam_fuse(self.fuse_conv(torch.cat([r, l], dim=1)))
        cache = {}
        out = cur
        for j, m in enumerate(self.shared):
            i = j + 3
            f = SHARED_FROM[i]
            if isinstance(f, list):
                inp = [cur if a == -1 else cache[a] for a in f]
            elif f == -1:
                inp = cur
            else:
                inp = cache[f]
            out = m(inp)
            if i in self.cbam:
                out = self.cbam[str(i)](out)
            if i in CACHE_INDICES:
                cache[i] = out
            cur = out
        return out

    @torch.no_grad()
    def predict_simple(self, x6, conf_thres=0.25, iou_thres=0.45):
        """Self-contained NMS decode for image-level metrics — bypasses Ultralytics'
        validator (which expects an unmodified DetectionModel). Returns one dict per
        image: {"boxes": (M,4) xyxy px, "scores": (M,), "labels": (M,)}."""
        from torchvision.ops import nms

        was_training = self.training
        self.eval()
        try:
            out = self.forward(x6)
            y = out[0] if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]) else None
            if y is None:  # raw train-mode maps -> run head in eval to postprocess
                self.head.eval()
                y, _ = self.head([m for m in out])
            boxes_all, scores_all, labels_all = [], [], []
            for b in range(y.shape[0]):
                xywh = y[b, :4].T                      # (N, 4) px
                cls_p = y[b, 4:].T                     # (N, nc) sigmoid probs
                scores, labels = cls_p.max(dim=1)
                x1y1 = xywh[:, :2] - xywh[:, 2:] / 2
                x2y2 = xywh[:, :2] + xywh[:, 2:] / 2
                boxes = torch.cat([x1y1, x2y2], dim=1)
                keep = scores > conf_thres
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if boxes.numel():
                    k = nms(boxes, scores, iou_thres)
                    boxes, scores, labels = boxes[k], scores[k], labels[k]
                boxes_all.append(boxes.cpu())
                scores_all.append(scores.cpu())
                labels_all.append(labels.cpu())
            return [{"boxes": b, "scores": s, "labels": l}
                    for b, s, l in zip(boxes_all, scores_all, labels_all)]
        finally:
            self.train(was_training)


def build_dual_stream(nc=2, pretrained="yolov8n.pt", use_cbam=True) -> DualStreamYOLOv8n:
    return DualStreamYOLOv8n(nc=nc, pretrained=pretrained, use_cbam=use_cbam)


def load_dual_stream(weights_path: str, device: str = "cpu") -> DualStreamYOLOv8n:
    """Rebuild a DualStreamYOLOv8n from a raw train_dual() state_dict (best.pt /
    last.pt). Ultralytics' YOLO() cannot consume these checkpoints. nc and CBAM
    presence are probed from the tensors; a train_config.yaml next to the checkpoint
    (train_dual writes one) wins when present."""
    _require_ultralytics()
    try:
        # weights_only=True accepts exactly what train_dual() saves (pure tensors) and
        # rejects pickled objects like Ultralytics checkpoints — on torch>=2.6 it is
        # also the only mode that loads at all without an allowlist
        sd = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception as e:  # noqa: BLE001 — any load failure = not a dual state_dict
        raise ValueError(f"{weights_path} is not a loadable raw state_dict ({e})") from e
    if not (isinstance(sd, dict) and sd and all(hasattr(v, "shape") for v in sd.values())):
        raise ValueError(f"{weights_path} is not a raw DualStreamYOLOv8n state_dict")
    nc, use_cbam = 2, any(k.startswith("cbam.") for k in sd)
    # cls-branch final convs are 1x1 with nc output channels (box finals are 64-wide)
    finals = [v.shape[0] for k, v in sd.items()
              if "cv3" in k and k.endswith(".conv.weight")
              and v.ndim == 4 and v.shape[2] == 1 and v.shape[3] == 1]
    if finals:
        nc = int(min(finals))
    cfg_p = Path(weights_path).parent / "train_config.yaml"
    if cfg_p.exists():
        d = yaml.safe_load(cfg_p.read_text(encoding="utf-8")) or {}
        nc = int(d.get("nc", nc))
        use_cbam = bool(d.get("use_cbam", use_cbam))
    # rebuild exactly the way train_dual() did (YOLO base + _rehead): the Detect head's
    # module tree from a reheaded pretrained model differs from a fresh yaml build, and
    # the strict load below needs an identical tree
    model = DualStreamYOLOv8n(nc=nc, pretrained="yolov8n.pt", use_cbam=use_cbam).to(device)
    model.load_state_dict(sd)   # strict: a probe mismatch must fail loudly
    return model.eval()


def try_load_dual_stream(weights_path: str, device: str = "cpu"):
    """load_dual_stream, or None when the file is not a dual-stream state_dict (e.g.
    an Ultralytics checkpoint from the single/baseline paths). Real load errors
    (corrupt file, shape mismatch) propagate loudly."""
    try:
        return load_dual_stream(weights_path, device)
    except ValueError:
        return None
