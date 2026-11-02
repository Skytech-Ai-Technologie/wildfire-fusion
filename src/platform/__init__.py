"""Live platform — runs the trained pipeline against a PC camera.

Layers (mirrors the repo's four-layer architecture, README §"Layout"):

    camera.py    source of frames: PC webcam | video file | synthetic test pattern
    engine.py   perceptual tier  — dual-stream YOLOv8n+CBAM (or single-stream), P5 embeddings
    pipeline.py contextual+decisional tiers — windowed entropy-gated transformer ->
                 Dempster-Shafer fusion -> alarm, wired through the repo's FusionLoop
    server.py   stdlib HTTP server: MJPEG video + JSON telemetry + control endpoints
    static/     browser UI (live view, fusion telemetry, controls)

The perceptual weights and the contextual checkpoint are the ones produced by
src/train_perceptual.py and src/train_contextual.py — the platform never trains.
"""
# Import cv2 here, before anything else pulls it in (ultralytics imports it at
# module level and prints a backend notice on the way). The platform's camera
# capture uses DirectShow, so that notice is a red herring.
from src.platform.camera import _cv2 as _quiet_cv2  # noqa: E402

_quiet_cv2()
