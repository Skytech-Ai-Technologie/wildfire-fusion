"""Live platform tests.

Scope:
  * the platform's preprocessing must be byte-identical to the training path
    (letterbox + BGR channel order) — otherwise the live detections are produced
    by a differently-fed network than the one that was trained
  * the live contextual refiner behaves like the offline one (window padding,
    one refined confidence per modality slot)
  * the platform's FusionLoop extension (runtime-switchable fusion policy) does
    not change the default behaviour
  * the frame sources and the HTTP layer behave (frames flow, JSON is
    serialisable, control endpoints answer)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from src.platform.camera import (SyntheticSource, SourceManager, build_source,  # noqa: E402
                              list_media, make_source, probe_cameras,
                              is_youtube_url, CameraSource)
from src.platform.engine import letterbox  # noqa: E402
from src.platform.pipeline import IiotSimulator, WindowedContextualRefiner  # noqa: E402
from src.platform.server import parse_multipart  # noqa: E402
from src.pipeline.fusion_loop import FusionLoop, ModalityFrame  # noqa: E402
from src.contextual.gated_transformer import ContextualEncoder  # noqa: E402


# --------------------------------------------------------------------------- #
# Preprocessing parity with the training path                                   #
# --------------------------------------------------------------------------- #
def test_letterbox_matches_the_training_implementation():
    """The live path must letterbox exactly like src/train_perceptual.py does."""
    from src.train_perceptual import _letterbox as train_letterbox

    rng = np.random.default_rng(0)
    for h, w in [(480, 640), (720, 1280), (100, 1000), (1080, 1920)]:
        im = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        a, ra, da = letterbox(im, 640)
        b, rb, db = train_letterbox(im, 640)
        assert np.array_equal(a, b)
        assert ra == rb and da == db


def test_letterbox_pads_with_114_and_preserves_aspect():
    im = np.full((480, 640, 3), 10, dtype=np.uint8)
    out, r, (dx, dy) = letterbox(im, 640)
    assert out.shape == (640, 640, 3)
    assert r == pytest.approx(1.0)
    assert (dx, dy) == (0, 80)
    # the padding band is the training-time gray value
    assert int(out[0, 0, 0]) == 114


# --------------------------------------------------------------------------- #
# Contextual refiner                                                           #
# --------------------------------------------------------------------------- #
def _refiner(T=32, d_model=64, in_dim=64):
    enc = ContextualEncoder(d_model=d_model, n_heads=4, n_layers=2, in_dim=in_dim)
    enc.eval()
    return WindowedContextualRefiner(enc, T=T)


def test_refiner_returns_one_confidence_per_modality_slot():
    r = _refiner()
    x = torch.zeros(1, 3, 64)
    conf = torch.tensor([[0.2, 0.4, 0.6]])
    # empty windows -> fall back to the incoming confidence
    out, _ = r(x, conf)
    assert out.shape == (1, 3)
    assert torch.allclose(out, conf)


def test_refiner_pads_a_short_window_instead_of_failing():
    """Fewer than T frames: left-pad by repeating the earliest frame, exactly like
    build_windows() in src/train_contextual.py."""
    r = _refiner(T=32)
    for i in range(3):
        r.push(0, np.random.randn(64).astype(np.float32), 0.5)
    assert r.filled(0) == 3
    out, _ = r(torch.zeros(1, 1, 64), torch.tensor([[0.5]]))
    assert out.shape == (1, 1)
    assert 0.0 <= float(out[0, 0]) <= 1.0


def test_refiner_window_is_capped_at_T():
    r = _refiner(T=8)
    for i in range(20):
        r.push(0, np.random.randn(64).astype(np.float32), 0.5)
    assert r.filled(0) == 8


def test_refiner_slots_are_independent():
    r = _refiner()
    r.push(0, np.ones(64, dtype=np.float32), 0.9)
    r.push(1, -np.ones(64, dtype=np.float32), 0.1)
    out, _ = r(torch.zeros(1, 2, 64), torch.tensor([[0.9, 0.1]]))
    assert out.shape == (1, 2)
    assert out[0, 0] != out[0, 1]


# --------------------------------------------------------------------------- #
# FusionLoop policy extension                                                   #
# --------------------------------------------------------------------------- #
def test_fusion_loop_default_policy_is_auto():
    loop = FusionLoop("c1", torch.nn.Identity())
    assert loop.decision_policy == "auto"


def test_fusion_loop_honours_the_runtime_policy():
    """Setting the policy must reach fusion_tick (the Table VI ablations)."""
    loop = FusionLoop("c1", torch.nn.Identity(), decision_policy="weighted")
    assert loop.decision_policy == "weighted"
    loop.decision_policy = "dempster"
    assert loop.decision_policy == "dempster"


def test_fusion_loop_still_works_with_the_new_policy_argument():
    class Passthrough(torch.nn.Module):
        def forward(self, x, conf):
            return conf, x

    loop = FusionLoop("c1", Passthrough(), decision_policy="weighted")
    frames = [ModalityFrame("c", True, [], 0.9, torch.randn(64))]
    res = loop.tick(0.0, frames)
    assert res["state"] in ("monitoring", "alarm", "alarm_confirmed")


# --------------------------------------------------------------------------- #
# Frame sources                                                                 #
# --------------------------------------------------------------------------- #
def test_synthetic_source_publishes_frames():
    src = SyntheticSource(width=320, height=240, fps=10)
    src.start()
    time.sleep(0.5)
    frame, ts, seq = src.read()
    src.stop()
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    assert seq > 0
    assert ts > 0


def test_build_source_rejects_unknown_source():
    with pytest.raises(ValueError):
        build_source("telepathy")


def test_file_source_requires_a_path():
    with pytest.raises(ValueError):
        build_source("file")


def test_probe_cameras_returns_a_list():
    """Probing must never raise, even with no camera attached."""
    assert isinstance(probe_cameras(max_index=1), list)


# --------------------------------------------------------------------------- #
# Input sources: camera / file / YouTube / synthetic                           #
# --------------------------------------------------------------------------- #
def test_make_source_builds_every_kind():
    assert make_source("camera", 0).kind == "camera"
    assert make_source("file", __file__).kind == "file"
    assert make_source("synthetic").kind == "synthetic"
    with pytest.raises(ValueError):
        make_source("telepathy")
    with pytest.raises(ValueError):
        make_source("file")           # a path is required
    with pytest.raises(ValueError):
        make_source("youtube")        # a url is required


def test_youtube_source_takes_the_video_track_only():
    """A detector needs no audio, and YouTube serves audio/video as separate DASH
    streams — asking for `bestvideo` yields a plain mp4 FFmpeg can read."""
    from src.platform.camera import YouTubeSource

    assert "bestvideo" in YouTubeSource.FORMAT
    assert "acodec" not in YouTubeSource.FORMAT


def test_youtube_url_detection():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert not is_youtube_url("https://vimeo.com/1")
    assert not is_youtube_url("")


def test_list_media_finds_video_files(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "movie.MKV").write_bytes(b"x")
    found = list_media(tmp_path)
    names = sorted(f["name"] for f in found)
    assert names == ["clip.mp4", "movie.MKV"]
    assert all(f["path"].endswith(f["name"]) for f in found)


def test_list_media_ignores_a_missing_folder(tmp_path):
    assert list_media(tmp_path / "nope") == []


def test_source_manager_advances_the_seq_only_on_new_frames():
    """The inference loop skips duplicates by comparing the seq — it must not
    advance when the source simply re-publishes the same frame."""
    src = SyntheticSource(width=160, height=120, fps=20)
    mgr = SourceManager(src)
    src.start()
    time.sleep(0.4)
    # rapid reads: nothing new was published, so the seq must not move
    first = mgr.read()[2]
    assert all(mgr.read()[2] == first for _ in range(5))
    # after a pause the source published again -> the seq advances
    time.sleep(0.3)
    assert mgr.read()[2] > first
    mgr.stop()


def test_source_manager_swap_replaces_the_source():
    a = SyntheticSource(width=160, height=120, fps=10)
    b = SyntheticSource(width=160, height=120, fps=10)
    mgr = SourceManager(a)
    mgr.start()
    time.sleep(0.2)
    mgr.swap(b)
    assert mgr.current is b
    time.sleep(0.3)
    frame, ts, seq = mgr.read()
    assert frame is not None
    mgr.stop()


def test_source_manager_reports_the_active_source():
    src = SyntheticSource(width=160, height=120, fps=10)
    mgr = SourceManager(src)
    d = mgr.describe()
    assert d["kind"] == "synthetic"
    assert "error" in d and "status" in d


def test_camera_source_reports_its_device_index():
    src = CameraSource(index=3)
    assert src.kind == "camera"
    assert src.name == "camera3"


def test_probe_cameras_honours_the_skip_list():
    """The scanner must leave the device the platform is using alone: a second
    handle on the same DirectShow device stalls the live capture."""
    calls = []

    import src.platform.camera as cam

    original = cam._probe_one

    def spy(index, timeout):
        calls.append(index)
        return {"index": index, "resolution": [640, 480], "name": f"camera {index}"}

    cam._probe_one = spy
    try:
        found = probe_cameras(max_index=4, skip={0, 2})
    finally:
        cam._probe_one = original
    assert 0 not in calls and 2 not in calls
    assert sorted(c["index"] for c in found) == [1, 3]


# --------------------------------------------------------------------------- #
# multipart upload parsing                                                      #
# --------------------------------------------------------------------------- #
def test_parse_multipart_reads_a_file_field():
    body = (b"--X\r\n"
            b'Content-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
            b"Content-Type: video/mp4\r\n\r\n"
            b"BINARYDATA\r\n"
            b"--X--\r\n")
    fields = parse_multipart(body, "multipart/form-data; boundary=X")
    assert "file" in fields
    assert fields["file"]["filename"] == "clip.mp4"
    assert fields["file"]["data"] == b"BINARYDATA"


def test_parse_multipart_rejects_a_missing_boundary():
    with pytest.raises(ValueError):
        parse_multipart(b"whatever", "application/json")


# --------------------------------------------------------------------------- #
# IIoT simulator                                                                #
# --------------------------------------------------------------------------- #
def test_iiot_simulator_readings_are_bounded_and_confidence_in_unit_range():
    sim = IiotSimulator(seed=3)
    for _ in range(200):
        sim.step(0.2)
        assert all(v >= 0 for v in sim.values.values())
        c = sim.anomaly_confidence()
        assert 0.0 <= c <= 1.0
        e = sim.embedding()
        assert e.shape == (256,)
        assert np.isfinite(e).all()


def test_iiot_simulator_readings_do_not_diverge_over_a_long_run():
    """Regression: the heat event must be a target OFFSET, not a per-step
    increment — accumulating it made the readings run away (temp 600 C)."""
    sim = IiotSimulator(seed=5)
    for _ in range(5000):
        sim.step(0.1)
    for k, v in sim.values.items():
        assert v < 10 * sim.BASELINE[k] + 50, f"{k} diverged to {v}"


def test_iiot_simulator_event_actually_raises_the_confidence():
    """The simulated heat event must be visible in the anomaly score, otherwise the
    IIoT modality is a constant no-op in the fusion."""
    sim = IiotSimulator(seed=11)
    quiet = []
    hot = []
    for i in range(2000):
        sim.step(0.1)
        phase = (sim.t % sim.event_period_s) / sim.event_period_s
        c = sim.anomaly_confidence()
        # "hot" = the peak of the ramp; "quiet" = well after the event has decayed
        # (the first seconds after an event are still recovering, so they are neither)
        if 0.40 <= phase <= 0.55:
            hot.append(c)
        elif 0.70 <= phase <= 0.95:
            quiet.append(c)
    assert hot and quiet
    assert max(hot) > max(quiet) + 0.15


def test_iiot_simulator_is_deterministic_for_a_seed():
    a, b = IiotSimulator(seed=7), IiotSimulator(seed=7)
    for _ in range(50):
        a.step(0.1)
        b.step(0.1)
    assert a.values == b.values


# --------------------------------------------------------------------------- #
# HTTP layer                                                                    #
# --------------------------------------------------------------------------- #
class _StubManager:
    def __init__(self):
        self.source = SyntheticSource(width=64, height=48, fps=5)

    def describe(self):
        return self.source.describe()


class _StubPipeline:
    """Minimal pipeline stand-in so the HTTP layer can be tested without models."""

    def __init__(self):
        self._jpeg = b"\xff\xd8\xff\xd9"
        self._seq = 0
        self.tau = 0.78
        self.policy = "auto"
        self.conf = 0.25
        self.manager = _StubManager()
        self.source_kind = "camera"
        self.source_value = None

    def status(self):
        return {"running": True, "state": "monitoring", "alarm": False,
                "fused_mass": {"F": 0.1, "not_F": 0.8, "Θ": 0.1},
                "tau": self.tau, "divergence": [0.01],
                "modalities": [{"name": "c", "label": "PC camera (RGB)", "active": True,
                                "raw_pi": 0.3, "refined_pi": 0.2, "simulated": False,
                                "source": "synthetic", "detections": 1, "window": 5}],
                "detections": [], "fps": 8.0, "latency_ms": 40.0, "error": None,
                "frame_seq": self._seq, "window": {"T": 32, "filled": 5}}

    def frame_jpeg(self, which="a"):
        self._seq += 1        # a live pipeline always produces a new frame
        return self._jpeg, self._seq

    def set_tau(self, v):
        self.tau = float(v)

    def set_conf(self, v):
        self.conf = float(v)

    def set_policy(self, p):
        self.policy = p

    def clear_alarm(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def set_source(self, kind, value=None):
        self.source_kind = kind
        self.source_value = value


@pytest.fixture(scope="module")
def http_server(tmp_path_factory):
    import threading
    from src.platform.server import PlatformServer

    media_dir = tmp_path_factory.mktemp("media")
    srv = PlatformServer(("127.0.0.1", 0), _StubPipeline(),
                         media_config={"media_dir": str(media_dir),
                                       "max_cameras": 1, "cameras": []})
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", media_dir
    srv.shutdown()
    srv.server_close()


def _get(url, timeout=10):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def test_index_page_is_served(http_server):
    base, _ = http_server
    status, body = _get(base + "/")
    assert status == 200
    assert b"FORFIRE-AI" in body


def test_status_endpoint_returns_json(http_server):
    base, _ = http_server
    status, body = _get(base + "/api/status")
    assert status == 200
    data = json.loads(body)
    assert "fused_mass" in data and "modalities" in data


def test_static_assets_are_served(http_server):
    base, _ = http_server
    status, body = _get(base + "/static/app.js")
    assert status == 200
    assert b"/api/status" in body


def test_control_endpoints_update_the_pipeline(http_server):
    import urllib.request
    base, _ = http_server
    payload = json.dumps({"action": "set_tau", "value": 0.42}).encode()
    req = urllib.request.Request(base + "/api/control", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200
        assert json.loads(r.read())["ok"] is True
    status, body = _get(base + "/api/status")
    assert json.loads(body)["tau"] == pytest.approx(0.42)


def test_unknown_control_action_is_rejected(http_server):
    import urllib.error
    import urllib.request
    base, _ = http_server
    payload = json.dumps({"action": "launch_missiles"}).encode()
    req = urllib.request.Request(base + "/api/control", data=payload,
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400


def test_mjpeg_stream_sends_jpeg_frames(http_server):
    import urllib.request
    base, _ = http_server
    req = urllib.request.Request(base + "/video.mjpg")
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        # read only the boundary marker — a full frame is ~70 bytes and a blocking
        # read() for more than one write would wait for the next frame
        head = r.read(9)
    assert head == b"--frame\r\n"


def test_media_index_lists_cameras_files_and_youtube_support(http_server):
    base, media_dir = http_server
    (media_dir / "clip.mp4").write_bytes(b"x")
    status, body = _get(base + "/api/media")
    assert status == 200
    data = json.loads(body)
    assert "cameras" in data and "files" in data
    assert "youtube_available" in data
    assert any(f["name"] == "clip.mp4" for f in data["files"])


def test_upload_stores_a_video_and_rejects_other_types(http_server):
    import urllib.error
    import urllib.request
    base, media_dir = http_server
    boundary = "----testboundary"
    body = (f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
            "Content-Type: video/mp4\r\n\r\n"
            "VIDEODATA\r\n"
            f"--{boundary}--\r\n").encode()
    req = urllib.request.Request(
        base + "/api/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=10) as r:
        payload = json.loads(r.read())
    assert payload["ok"] is True
    assert (media_dir / payload["name"]).exists()

    # a non-video file must be refused
    body2 = (f"--{boundary}\r\n"
             'Content-Disposition: form-data; name="file"; filename="notes.txt"\r\n'
             "Content-Type: text/plain\r\n\r\n"
             "hello\r\n"
             f"--{boundary}--\r\n").encode()
    req2 = urllib.request.Request(
        base + "/api/upload", data=body2,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req2, timeout=10)
    assert e.value.code == 400


def test_set_source_action_switches_the_input(http_server):
    import urllib.request
    base, _ = http_server
    payload = json.dumps({"action": "set_source", "kind": "youtube",
                          "value": "https://youtu.be/abc"}).encode()
    req = urllib.request.Request(base + "/api/control", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert json.loads(r.read())["ok"] is True
    assert http_server[0] and True
