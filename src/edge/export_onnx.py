"""§7.1 — ONNX export for the perceptual and contextual tiers."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def export_perceptual_onnx(weights_path: str, out_path: str | None = None, imgsz=640):
    """Export a (single- or dual-stream) perceptual checkpoint to ONNX via Ultralytics.

    Ultralytics writes <weights>.onnx next to the weights file; we move/rename to
    out_path when given.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    exported = model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True)
    exported = Path(exported if isinstance(exported, str) else str(weights_path).replace(".pt", ".onnx"))
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != out.resolve():
            exported.replace(out)
        return out
    return exported


def export_contextual_onnx(checkpoint_path: str, out_path: str, d_model=256,
                           n_heads=8, n_layers=4, dropout=0.1, in_dim=None,
                           max_tokens=3):
    """Export the contextual transformer with dynamic axes on batch and token dims.

    checkpoint_path: state_dict saved by train_contextual.py (may be None to export a
    freshly initialized model for shape verification only).
    """
    import torch
    from src.contextual.gated_transformer import ContextualEncoder

    model = ContextualEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                              dropout=dropout, in_dim=in_dim)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
    model.eval()

    x = torch.randn(1, max_tokens, in_dim or d_model)
    conf = torch.rand(1, max_tokens)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (x, conf), str(out),
        input_names=["tokens", "modality_confidence"],
        output_names=["refined_pi", "context"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "tokens"},
            "modality_confidence": {0: "batch", 1: "tokens"},
            "refined_pi": {0: "batch", 1: "tokens"},
            "context": {0: "batch", 1: "tokens"},
        },
        opset_version=17,
    )
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="§7.1 ONNX export")
    ap.add_argument("--perceptual-weights")
    ap.add_argument("--contextual-checkpoint")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--in-dim", type=int, default=None)
    ap.add_argument("--out", default="runs/export")
    args = ap.parse_args()
    if args.perceptual_weights:
        p = export_perceptual_onnx(args.perceptual_weights,
                                   str(Path(args.out) / "perceptual.onnx"))
        print(f"perceptual ONNX -> {p}")
    if args.contextual_checkpoint or args.perceptual_weights is None:
        p = export_contextual_onnx(args.contextual_checkpoint or "",
                                   str(Path(args.out) / "contextual.onnx"),
                                   d_model=args.d_model, in_dim=args.in_dim)
        print(f"contextual ONNX -> {p}")
