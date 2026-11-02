"""§4.3 — Detection loss, implements eq. (2):

    Ldet = λ1 * L_CIoU + λ2 * L_focal + λ3 * L_obj

Two layers live here:

1. Pure reference functions (`complete_iou_loss`, `focal_loss`, `bce_objectness_loss`,
   `detection_loss`) — the exact eq.-(2) components, unit-testable with plain tensors
   and no Ultralytics dependency.

2. `WeightedV8DetectionLoss` — the training-time loss actually plugged into the
   dual-stream YOLOv8n. Ultralytics 8.x already provides CIoU+DFL box loss and a BCE
   classification loss assigned with Task-Aligned Assignment (TAL). We reuse that
   machinery, rescale the components with the paper's λ weights (box+dfl ← λ1,
   classification ← λ2), swap BCE for focal weighting when focal_gamma > 0, and add the
   explicit objectness BCE term that eq. (2) expects. YOLOv8 has no separate objectness
   head, so objectness is supervised as "the max class logit in a cell must be high iff
   a GT box center falls in that cell" — documented ASSUMPTION filling the paper gap.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

LAMBDA_CIOU = 1.0
LAMBDA_FOCAL = 0.5
LAMBDA_OBJ = 0.7
FOCAL_GAMMA = 1.5


# --------------------------------------------------------------------------- #
# 1. Pure eq.-(2) reference components                                        #
# --------------------------------------------------------------------------- #
def complete_iou_loss(pred_boxes, target_boxes, eps=1e-7):
    """CIoU loss on axis-aligned xyxy boxes, shape (N, 4). Returns a scalar."""
    pred_boxes = torch.as_tensor(pred_boxes, dtype=torch.float32)
    target_boxes = torch.as_tensor(target_boxes, dtype=torch.float32)
    if pred_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0

    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)
    pw, ph = (px2 - px1).clamp(min=0), (py2 - py1).clamp(min=0)
    tw, th = (tx2 - tx1).clamp(min=0), (ty2 - ty1).clamp(min=0)

    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter
    iou = inter / union.clamp(min=eps)

    # smallest enclosing box
    cw = (torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(min=eps)
    ch = (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(min=eps)
    c2 = cw ** 2 + ch ** 2
    rho2 = ((px1 + px2 - tx1 - tx2) ** 2 + (py1 + py2 - ty1 - ty2) ** 2) / 4

    # aspect-ratio consistency term
    v = (4 / torch.pi ** 2) * (torch.atan(tw / th.clamp(min=eps)) - torch.atan(pw / ph.clamp(min=eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(min=eps)
    return (1 - iou + rho2 / c2 + alpha * v).mean()


def focal_loss(logits, targets, gamma=FOCAL_GAMMA, alpha=0.25, eps=1e-8):
    """Binary focal loss on raw logits; targets in [0, 1] (soft targets allowed)."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.float32)
    if logits.numel() == 0:
        return logits.sum() * 0.0
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets.clamp(eps, 1 - eps), reduction="none")
    return (alpha * (1 - pt) ** gamma * bce).mean()


def bce_objectness_loss(obj_logits, obj_targets):
    """Plain BCE-with-logits objectness loss. obj_logits raw; obj_targets in {0,1}."""
    obj_logits = torch.as_tensor(obj_logits, dtype=torch.float32)
    obj_targets = torch.as_tensor(obj_targets, dtype=torch.float32)
    if obj_logits.numel() == 0:
        return obj_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(obj_logits, obj_targets)


def detection_loss(pred, target, lambda1=LAMBDA_CIOU, lambda2=LAMBDA_FOCAL,
                   lambda3=LAMBDA_OBJ, gamma=FOCAL_GAMMA):
    """Reference eq.-(2) combination over dict inputs:

        pred:   {"boxes": (N,4) xyxy, "cls_logits": (N,C), "obj_logits": (N,)}
        target: {"boxes": (N,4) xyxy, "cls": (N,C) or (N,), "obj": (N,)}

    NOTE: this is the paper-faithful reference implementation for tests/sanity checks.
    The live training path uses WeightedV8DetectionLoss, which reuses Ultralytics'
    TAL assignment (their box loss is already CIoU-based).
    """
    l_ciou = complete_iou_loss(pred["boxes"], target["boxes"])
    cls_t = target["cls"]
    if cls_t.dim() == 1:
        cls_t = F.one_hot(cls_t.long(), num_classes=pred["cls_logits"].shape[-1]).float()
    l_focal = focal_loss(pred["cls_logits"], cls_t, gamma=gamma)
    l_obj = bce_objectness_loss(pred["obj_logits"], target["obj"])
    total = lambda1 * l_ciou + lambda2 * l_focal + lambda3 * l_obj
    return total, {"ciou": float(l_ciou), "focal": float(l_focal), "obj": float(l_obj)}


# --------------------------------------------------------------------------- #
# 2. Ultralytics-integrated training loss                                     #
# --------------------------------------------------------------------------- #
def _import_v8_detection_loss():
    try:
        from ultralytics.utils.loss import v8DetectionLoss  # Ultralytics >= 8.1
        return v8DetectionLoss
    except ImportError:
        try:
            from ultralytics.yolo.utils.loss import v8DetectionLoss  # older layouts
            return v8DetectionLoss
        except ImportError:
            return None


class _FocalBCE(nn.Module):
    """BCE-with-logits with focal modulation, reduction='none' (Ultralytics divides by
    target_scores_sum after calling self.bce, so we must keep per-element output)."""

    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        return (1 - pt).pow(self.gamma) * bce


def objectness_bce(feats, batch, nc, reg_max, strides, pred_scores=None):
    """λ3 term of eq. (2): explicit objectness BCE.

    Target = 1 in any grid cell whose center coincides with a GT box center cell,
    0 elsewhere; prediction = max class logit per cell. ASSUMPTION documented in the
    module docstring (YOLOv8 has no obj head).

    Ultralytics 8.3+ layout: `feats` are the head's input maps (for per-level H, W) and
    `pred_scores` is the concatenated (B, nc, N) class-logit tensor (row-major per
    level, P3 first — matching forward_head). 8.2 layout: `feats` are the raw
    (B, 4*reg_max+nc, H, W) maps and `pred_scores` is None.
    """
    device = feats[0].device
    gt_bboxes = batch["bboxes"].to(device)      # (Ntot, 4) normalized xywh
    gt_batch = batch["batch_idx"].to(device)    # (Ntot,)
    total = feats[0].new_zeros(())

    if pred_scores is not None:                 # 8.3+/8.4
        B = pred_scores.shape[0]
        start = 0
        for f in feats:
            _, _, H, W = f.shape
            n = H * W
            logit_map = pred_scores[:, :, start:start + n].reshape(B, nc, H, W).amax(dim=1)
            obj_target = torch.zeros(B, H, W, device=device)
            if gt_bboxes.numel():
                gx = (gt_bboxes[:, 0] * W).floor().clamp(0, W - 1).long()
                gy = (gt_bboxes[:, 1] * H).floor().clamp(0, H - 1).long()
                obj_target[gt_batch.long(), gy, gx] = 1.0
            total = total + F.binary_cross_entropy_with_logits(logit_map, obj_target,
                                                               reduction="mean")
            start += n
    else:                                       # 8.2 raw per-level maps
        B = feats[0].shape[0]
        for f, s in zip(feats, strides):
            _, _, H, W = f.shape
            logit_map = f[:, reg_max * 4:, :, :].amax(dim=1)
            obj_target = torch.zeros(B, H, W, device=device)
            if gt_bboxes.numel():
                gx = (gt_bboxes[:, 0] * W).floor().clamp(0, W - 1).long()
                gy = (gt_bboxes[:, 1] * H).floor().clamp(0, H - 1).long()
                obj_target[gt_batch.long(), gy, gx] = 1.0
            total = total + F.binary_cross_entropy_with_logits(logit_map, obj_target,
                                                               reduction="mean")
    return total


class _LossModelShim(nn.Module):
    """Gives v8DetectionLoss what it reads off a DetectionModel:
    `model.model[-1]` (the Detect head), `model.args` (the λ loss gains) and
    `model.parameters()` (device probing). An nn.Module so .parameters() exists."""

    def __init__(self, head, lambda_ciou, lambda_focal):
        super().__init__()
        self.model = nn.ModuleList([head])
        self.args = SimpleNamespace(box=lambda_ciou, cls=lambda_focal, dfl=lambda_ciou)


def _feats_from_preds(preds):
    """Extract the per-level raw feature maps from any Ultralytics 8.x train/eval output.

    8.2 train: list of (B, no, H, W) maps.
    8.3+/8.4 train: dict {"boxes", "scores", "feats": [maps]} (or nested under "one2many"
                    for end-to-end heads).
    eval: (y, <dict>) tuple -> the dict part.
    """
    out = preds[1] if isinstance(preds, tuple) else preds
    if isinstance(out, dict):
        if "feats" not in out and "one2many" in out:
            out = out["one2many"]
        return out.get("feats")
    return out


_v8DetectionLoss = _import_v8_detection_loss()

if _v8DetectionLoss is not None:

    class WeightedV8DetectionLoss(_v8DetectionLoss):
        """v8DetectionLoss with the paper's λ weights (eq. 2) + explicit obj BCE.

        Usage:
            model = DualStreamYOLOv8n(nc=2).to(device)     # move to device FIRST
            criterion = WeightedV8DetectionLoss(model)     # caches model device at init
            loss, loss_items = criterion(model(imgs), batch)
        """

        def __init__(self, model, lambda_ciou=LAMBDA_CIOU, lambda_focal=LAMBDA_FOCAL,
                     lambda_obj=LAMBDA_OBJ, focal_gamma=FOCAL_GAMMA):
            head = model.head if hasattr(model, "head") else model.model[-1]
            shim = _LossModelShim(head, lambda_ciou, lambda_focal)
            super().__init__(shim)
            self.lambda_obj = lambda_obj
            if focal_gamma and focal_gamma > 0:
                self.bce = _FocalBCE(focal_gamma)

        def __call__(self, preds, batch):
            feats = _feats_from_preds(preds)
            loss, items = super().__call__(preds, batch)
            bs = feats[0].shape[0]
            pred_scores = None
            if isinstance(preds, dict) or (isinstance(preds, tuple) and isinstance(preds[1], dict)):
                parsed = preds[1] if isinstance(preds, tuple) else preds
                if "feats" not in parsed and "one2many" in parsed:
                    parsed = parsed["one2many"]
                pred_scores = parsed["scores"]        # 8.3+/8.4: (B, nc, N) class logits
            l_obj = objectness_bce(feats, batch, self.nc, self.reg_max, self.stride.tolist(),
                                   pred_scores=pred_scores)
            obj_term = self.lambda_obj * l_obj * bs   # parent scales by batch_size too
            if isinstance(items, dict):               # Ultralytics >= 8.3: loss is an
                # unsummed (3,) vector (box, cls, dfl) x batch_size, items a dict
                items = dict(items)
                items["obj_loss"] = float(obj_term.detach())
                return loss.sum() + obj_term, items
            return loss + obj_term, torch.cat([items, obj_term.detach()[None]])  # 8.2 scalar

else:  # pragma: no cover - Ultralytics not installed

    class WeightedV8DetectionLoss:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Ultralytics is not installed; WeightedV8DetectionLoss unavailable.")
