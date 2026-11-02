"""Launch the live platform: trained model + a live video feed + a browser UI.

    python scripts/run_platform.py                       # local camera (device 0)
    python scripts/run_platform.py --camera 1            # another device
    python scripts/run_platform.py --list-cameras        # probe available devices
    python scripts/run_platform.py --source synthetic    # no camera needed
    python scripts/run_platform.py --source file --path clip.mp4
    python scripts/run_platform.py --source file --path "rtsp://host:554/stream"
                                         # IP / RTSP / PTZ camera streams work too —
                                         # any URL OpenCV/FFmpeg can open

Model selection is automatic: the newest runs/dual_*/best.pt (dual-stream
YOLOv8n+CBAM) is preferred, falling back to the newest runs/single_*/weights/best.pt.
Pass --weights to override. The contextual checkpoint defaults to the newest
runs/contextual_*/best.pt; --no-contextual falls back to identity refinement
(same convention as scripts/run_field_sim.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.platform.camera import build_source, make_source, probe_cameras  # noqa: E402
from src.platform.engine import PerceptionEngine  # noqa: E402
from src.platform.pipeline import LivePipeline, WindowedContextualRefiner  # noqa: E402
from src.platform.server import serve  # noqa: E402


def _newest(paths):
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def default_weights():
    dual = _newest(REPO.glob("runs/dual_*/best.pt"))
    if dual is not None:
        return dual
    return _newest(REPO.glob("runs/single_*/weights/best.pt"))


def default_contextual():
    real = _newest(REPO.glob("runs/contextual_real_*/best.pt"))
    if real is not None:
        return real
    return _newest(REPO.glob("runs/contextual_*/best.pt"))


class IdentityEncoder(torch.nn.Module):
    """No contextual checkpoint -> refined pi == raw pi. An untrained
    ContextualEncoder would emit arbitrary confidences and never alarm
    (same guard as scripts/run_field_sim.py)."""

    def forward(self, x, conf):
        return conf, x


def load_contextual(path, device):
    import yaml
    from src.contextual.gated_transformer import ContextualEncoder

    cfg = {"d_model": 256, "n_heads": 8, "n_layers": 4, "dropout": 0.1, "in_dim": 256}
    cfg_path = path.parent / "train_config.yaml"
    if cfg_path.exists():
        try:
            d = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            for k in ("d_model", "n_heads", "n_layers", "dropout", "in_dim"):
                if k in d:
                    cfg[k] = d[k]
        except Exception:  # noqa: BLE001 — fall back to the defaults
            pass
    enc = ContextualEncoder(d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                             n_layers=cfg["n_layers"], dropout=cfg["dropout"],
                             in_dim=cfg["in_dim"])
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else state
    enc.load_state_dict(sd)
    enc.eval()
    enc.to(device)
    return enc, cfg


def build_mqtt_client(host, port, client_id="wildfire-platform"):
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("[platform] paho-mqtt not installed -> alarms are not published")
        return None
    try:
        client = mqtt.Client(client_id=client_id)
        client.connect(host, port, keepalive=30)
        client.loop_start()
        print(f"[platform] MQTT connected: {host}:{port}")
        return client
    except Exception as e:  # noqa: BLE001 — a broker outage must not stop the UI
        print(f"[platform] MQTT connect failed ({e}) -> alarms are not published")
        return None


def main():
    ap = argparse.ArgumentParser(description="Live wildfire detection platform")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--source", choices=["camera", "file", "youtube", "synthetic"],
                    default="camera")
    ap.add_argument("--camera", type=int, default=0, help="camera device index")
    ap.add_argument("--camera2", type=int, default=None,
                    help="optional second camera -> fused as a second RGB modality")
    ap.add_argument("--path", default=None, help="video file (with --source file)")
    ap.add_argument("--youtube", default=None,
                    help="YouTube URL to play instead of the camera (with --source youtube)")
    ap.add_argument("--media-dir", default="media",
                    help="folder for video files (and for videos downloaded from YouTube)")
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf"])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=8, help="capture/inference rate")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--contextual", default=None)
    ap.add_argument("--no-contextual", action="store_true")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--tau", type=float, default=None,
                    help="alarm threshold (default: configs/default.yaml -> 0.78)")
    ap.add_argument("--policy", default="auto", choices=["auto", "weighted", "dempster"])
    ap.add_argument("--iiot", action="store_true",
                    help="add a SIMULATED IIoT modality (independent synthetic walk)")
    ap.add_argument("--mqtt-host", default=None)
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--cell-id", default="live-cell")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_cameras:
        found = probe_cameras()
        print(json.dumps(found, indent=2))
        if not found:
            print("No camera could be opened. Try --source synthetic.")
        return

    weights = Path(args.weights) if args.weights else default_weights()
    if weights is None or not Path(weights).exists():
        raise SystemExit(
            "No trained perceptual checkpoint found.\n"
            "Train one first:\n"
            "  python src/train_perceptual.py --dataset dfire --arch dual\n"
            "or pass --weights <path/to/best.pt>")
    print(f"[platform] perceptual weights : {weights}")

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[platform] device             : {device}")

    engine = PerceptionEngine(str(weights), device=device, imgsz=args.imgsz,
                               conf=args.conf, iou=args.iou)
    t0 = time.time()
    engine.load()
    print(f"[platform] loaded {engine.arch} model in {time.time() - t0:.1f}s")

    ctx_path = None
    if not args.no_contextual:
        ctx_path = Path(args.contextual) if args.contextual else default_contextual()
        if ctx_path is not None and not ctx_path.exists():
            ctx_path = None
    if ctx_path is not None:
        encoder, cfg = load_contextual(ctx_path, device)
        print(f"[platform] contextual encoder: {ctx_path}  {cfg}")
    else:
        encoder = IdentityEncoder()
        print("[platform] contextual encoder: NONE -> identity refinement "
              "(pass --contextual <best.pt> for the trained transformer)")

    refiner = WindowedContextualRefiner(encoder, T=32)
    refiner.contextual_loaded = ctx_path is not None

    from src.pipeline.fusion_loop import FusionLoop
    loop = FusionLoop(cell_id=args.cell_id, contextual_encoder=refiner,
                      decision_policy=args.policy)
    if args.tau is not None:
        loop.fire_alarm_threshold = args.tau
    print(f"[platform] alarm threshold   : tau={loop.fire_alarm_threshold}  "
          f"policy={loop.decision_policy}")

    mqtt_client = None
    if args.mqtt_host:
        mqtt_client = build_mqtt_client(args.mqtt_host, args.mqtt_port, args.cell_id)

    media_dir = REPO / args.media_dir
    media_dir.mkdir(parents=True, exist_ok=True)

    # Probe the cameras BEFORE the platform opens one: a second handle on the same
    # DirectShow device can stall the live capture, so the platform's own device
    # must not be probed while it is in use.
    detected = probe_cameras(max_index=4)
    if detected:
        print("[platform] cameras detected: "
              + ", ".join(f"{c['index']} ({c['resolution'][0]}x{c['resolution'][1]})"
                          for c in detected))

    if args.source == "youtube":
        if not args.youtube:
            raise SystemExit("--source youtube needs --youtube <url>")
        source = make_source("youtube", args.youtube, width=args.width,
                             height=args.height, fps=args.fps,
                             media_dir=media_dir)
    else:
        source = build_source(args.source, index=args.camera, path=args.path,
                              width=args.width, height=args.height, fps=args.fps,
                              backend=args.backend)
    secondary = None
    if args.camera2 is not None:
        secondary = build_source("camera", index=args.camera2, width=args.width,
                                 height=args.height, fps=args.fps,
                                 backend=args.backend)

    source_config = {"width": args.width, "height": args.height, "fps": args.fps,
                     "backend": args.backend, "media_dir": str(media_dir)}

    pipeline = LivePipeline(engine, refiner, loop, source, cell_id=args.cell_id,
                            secondary_source=secondary, use_iiot=args.iiot,
                            source_config=source_config)
    pipeline.start()
    time.sleep(1.5)   # let the first frame land so the UI opens on a live view
    err = pipeline.status().get("error")
    if err and args.source == "camera":
        print(f"[platform] WARNING: {err}")
        print("[platform] falling back to the synthetic source so the UI still runs")
        pipeline.stop()
        source = build_source("synthetic", width=args.width, height=args.height,
                              fps=args.fps)
        pipeline = LivePipeline(engine, refiner, loop, source, cell_id=args.cell_id,
                                secondary_source=None, use_iiot=args.iiot,
                                source_config=source_config)
        pipeline.start()

    media_config = {"media_dir": str(media_dir), "max_cameras": 4,
                    "extra_dirs": [str(REPO / "data")],
                    "cameras": detected}
    serve(pipeline, host=args.host, port=args.port, verbose=args.verbose,
          media_config=media_config)


if __name__ == "__main__":
    main()
