"""Quick smoke test for the dual-stream model + weighted loss (not part of pytest —
downloads yolov8n.pt when run with --pretrained)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.perceptual.dual_stream_yolo import DualStreamYOLOv8n
from src.perceptual.losses import WeightedV8DetectionLoss

pretrained = None
if "--pretrained" in sys.argv:
    pretrained = "yolov8n.pt"

m = DualStreamYOLOv8n(nc=2, pretrained=pretrained, use_cbam=True)
m.train()
x = torch.rand(2, 6, 64, 64)
out = m(x)
assert isinstance(out, dict) and "feats" in out, type(out)   # Ultralytics 8.3+ head dict
print("feats:", [tuple(o.shape) for o in out["feats"]])
print("boxes:", tuple(out["boxes"].shape), "scores:", tuple(out["scores"].shape))
assert out["scores"].shape == (2, 2, 8 * 8 + 4 * 4 + 2 * 2), "expected (B, nc, N) class logits"

crit = WeightedV8DetectionLoss(m)
batch = {
    "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
    "cls": torch.tensor([[0.0], [1.0], [0.0]]),
    "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3],
                            [0.3, 0.7, 0.1, 0.1],
                            [0.6, 0.2, 0.2, 0.2]]),
}
loss, items = crit(out, batch)
print("loss:", float(loss.detach()), "items:", items)
loss.backward()
gn = m.rgb_stream[0].conv.weight.grad
print("backward OK, rgb stem grad finite:", bool(torch.isfinite(gn).all()))

m.eval()
with torch.no_grad():
    preds = m.predict_simple(torch.rand(1, 6, 64, 64))
print("predict_simple ->", len(preds), "imgs; keys:", list(preds[0].keys()))
print("DUAL_STREAM_SMOKE_OK")
