"""§4.2 — dual-stream YOLOv8n tests (offline build, no weight download)."""
import pytest

torch = pytest.importorskip("torch")
ultralytics = pytest.importorskip("ultralytics")

from src.perceptual.dual_stream_yolo import DualStreamYOLOv8n  # noqa: E402
from src.perceptual.losses import WeightedV8DetectionLoss  # noqa: E402


@pytest.fixture(scope="module")
def model():
    # offline build from the bundled yolov8n.yaml — no checkpoint download
    m = DualStreamYOLOv8n(nc=2, pretrained=None, use_cbam=True)
    return m


def test_all_parameters_trainable(model):
    assert all(p.requires_grad for p in model.parameters()), \
        "checkpoint-loaded weights arrive frozen in Ultralytics 8.3+; model must train"


def test_train_mode_output_shapes(model):
    model.train()
    out = model(torch.rand(2, 6, 64, 64))
    assert isinstance(out, dict) and "feats" in out           # Ultralytics 8.3+ head dict
    assert out["scores"].shape == (2, 2, 84)                  # (B, nc, P3+P4+P5 anchors)
    assert out["boxes"].shape[0] == 2 and out["boxes"].shape[2] == 84


def test_cbam_present_after_every_c2f(model):
    # 7 shared CBAMs (P3/P4/P5 backbone + 4 head C2f) in model.cbam, plus one per
    # stream and one post-fusion
    assert len(model.cbam) == 7
    assert "4" in model.cbam and "21" in model.cbam
    assert isinstance(model.rgb_stream[3], object) and model.rgb_stream[-1].__class__.__name__ == "CBAM"
    assert model.lwir_stream[-1].__class__.__name__ == "CBAM"
    assert model.cbam_fuse.__class__.__name__ == "CBAM"


def test_weighted_loss_components_and_backward(model):
    model.train()
    crit = WeightedV8DetectionLoss(model)
    batch = {
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.6, 0.2, 0.2, 0.2]]),
    }
    loss, items = crit(model(torch.rand(2, 6, 64, 64)), batch)
    assert float(loss.detach()) >= 0
    assert set(items) >= {"box_loss", "cls_loss", "obj_loss"}
    loss.backward()
    assert torch.isfinite(model.rgb_stream[0].conv.weight.grad).all()


def test_predict_simple_decode(model):
    model.eval()
    with torch.no_grad():
        preds = model.predict_simple(torch.rand(1, 6, 64, 64), conf_thres=0.99)
    assert len(preds) == 1
    assert preds[0]["boxes"].shape[1] == 4                    # empty at a 0.99 cutoff, but well-formed
