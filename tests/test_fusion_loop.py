"""§8 — fusion loop / Algorithm 1 + operational policy tests.

The alarm-path tests use a deterministic passthrough encoder: an UNTRAINED
ContextualEncoder emits arbitrary refined confidences, so asserting alarm behavior
against it would test random initialization, not the fusion logic. The entropy-gate
behavior of the real encoder is covered in test_contextual.py.
"""
import queue

import pytest

torch = pytest.importorskip("torch")

from src.pipeline.fusion_loop import FusionLoop, ModalityFrame, fusion_tick  # noqa: E402
from src.contextual.gated_transformer import ContextualEncoder  # noqa: E402


class PassthroughEncoder(torch.nn.Module):
    """Algorithm 1 stub: refined pi == raw pi (deterministic fusion-logic tests)."""

    def forward(self, x, modality_confidence):
        return modality_confidence, x


def _enc():
    enc = ContextualEncoder(d_model=64, n_heads=4, n_layers=2, in_dim=64)
    enc.eval()
    return enc


def _frame(name, pi, t=None):
    return ModalityFrame(name=name, active=True, delta=[], pi=pi,
                         embedding=torch.randn(64) * 0.1, ts=t)


def test_fusion_tick_no_active_modalities():
    res = fusion_tick("c1", 0.0,
                      [ModalityFrame("c", False, [], 0.9, torch.randn(64))],
                      PassthroughEncoder(), {"c": 0.8})
    assert res["state"] == "monitoring"
    assert res["reason"] == "no_active_modalities"


def test_fusion_tick_alarm_reaches_supervisory_queue():
    q = queue.Queue()
    res = fusion_tick("c1", 0.0, [_frame("c", 0.95), _frame("d", 0.95)],
                      PassthroughEncoder(), {"c": 0.9, "d": 0.9},
                      mqtt_client=None, supervisory_queue=q, alarm_threshold=0.5)
    assert res["state"] == "alarm"
    assert not q.empty()
    msg = q.get(timeout=1)
    assert msg["suggest_uav_dispatch"] is False    # drone ("d") present


def test_fusion_tick_without_mqtt_client_does_not_crash_on_alarm():
    res = fusion_tick("c1", 0.0, [_frame("c", 0.95)], PassthroughEncoder(),
                      {"c": 0.9}, mqtt_client=None, alarm_threshold=0.5)
    assert res["state"] == "alarm"


def test_uav_suggested_when_drone_absent():
    q = queue.Queue()
    fusion_tick("c1", 0.0, [_frame("c", 0.95)], PassthroughEncoder(), {"c": 0.9},
                mqtt_client=None, supervisory_queue=q, alarm_threshold=0.5)
    assert q.get(timeout=1)["suggest_uav_dispatch"] is True


def test_monitoring_below_threshold():
    res = fusion_tick("c1", 0.0, [_frame("c", 0.05), _frame("d", 0.05)],
                      PassthroughEncoder(), {"c": 0.9, "d": 0.9}, alarm_threshold=0.78)
    assert res["state"] == "monitoring"


def _loop_config(tmp_path):
    """Explicit decisional config so tests never depend on whatever calibration file
    happens to sit in configs/ (calibrate_decisional.py writes one)."""
    cfg = tmp_path / "calibration.yaml"
    cfg.write_text("reliability: {c: 0.9, d: 0.9, s: 0.9}\n"
                   "fire_alarm_threshold: 0.5\ndivergence_threshold: 0.3\n",
                   encoding="utf-8")
    return str(cfg)


def test_hysteresis_requires_consecutive_ticks(tmp_path):
    loop = FusionLoop("c1", PassthroughEncoder(), config_path=_loop_config(tmp_path))
    high = 0.97
    r1 = loop.tick(0.0, [_frame("c", high), _frame("d", high)])
    r2 = loop.tick(1.0, [_frame("c", high), _frame("d", high)])
    if loop.op["alarm_hysteresis"]["consecutive_ticks_required"] == 2:
        assert r1["state"] == "alarm"          # raw alarm on tick 1, not yet confirmed
        assert r2["state"] == "alarm_confirmed"
    else:
        assert r2["state"] == "alarm_confirmed"


def test_packet_loss_marks_modality_inactive(tmp_path):
    loop = FusionLoop("c1", PassthroughEncoder(), config_path=_loop_config(tmp_path))
    loop.on_heartbeat("s", 0.0)
    res = loop.tick(1000.0, [_frame("c", 0.5), _frame("s", 0.5)])
    # "s" missed >2 beats of 60s -> marked inactive; must not appear in per-modality pi
    assert "s" not in res.get("per_modality_pi", {})


def test_timestamp_sync_drops_stale_frames(tmp_path):
    loop = FusionLoop("c1", PassthroughEncoder(), config_path=_loop_config(tmp_path))
    res = loop.tick(10.0, [_frame("c", 0.9, t=100.0)])   # 90s skew > 2s tolerance
    assert res.get("reason") == "no_active_modalities" or \
        "c" not in res.get("per_modality_pi", {})


def test_alarm_latch_and_cooldown(tmp_path):
    loop = FusionLoop("c1", PassthroughEncoder(), config_path=_loop_config(tmp_path))
    for i in range(2):
        loop.tick(i, [_frame("c", 0.97), _frame("d", 0.97)])
    loop.clear_alarm(t=100.0)
    # during cooldown, even confirmed alarms are suppressed
    loop._consecutive = 5
    res = loop.tick(200.0, [_frame("c", 0.97), _frame("d", 0.97)])
    assert res["state"] == "suppressed_cooldown"


def test_cooldown_burst_does_not_re_latch(tmp_path):
    """Regression: a raw-alarm burst DURING cooldown must not set the latch — after
    the cooldown expires, re-arming again requires `req` consecutive ticks, not one."""
    loop = FusionLoop("c1", PassthroughEncoder(), config_path=_loop_config(tmp_path))
    req = loop.op["alarm_hysteresis"]["consecutive_ticks_required"]
    for i in range(req):
        loop.tick(float(i), [_frame("c", 0.97), _frame("d", 0.97)])   # confirm + latch
    loop.clear_alarm(t=100.0)                                         # cooldown 100..400
    for i in range(req):                                              # burst while cooling
        res = loop.tick(150.0 + i, [_frame("c", 0.97), _frame("d", 0.97)])
        assert res["state"] != "alarm_confirmed"   # raw until hysteresis, else suppressed
    loop.tick(200.0, [_frame("c", 0.05), _frame("d", 0.05)])          # fire gone meanwhile
    res = loop.tick(500.0, [_frame("c", 0.97), _frame("d", 0.97)])    # single tick post-cooldown
    assert res["state"] == "alarm"                                    # NOT confirmed
    res = loop.tick(510.0, [_frame("c", 0.97), _frame("d", 0.97)])    # 2nd consecutive tick
    assert res["state"] == "alarm_confirmed"


def test_untrained_encoder_still_runs_end_to_end():
    """Smoke: the real (untrained) ContextualEncoder must run through Algorithm 1 and
    return a well-formed result — value assertions live elsewhere."""
    res = fusion_tick("c1", 0.0, [_frame("c", 0.5), _frame("s", 0.5)], _enc(),
                      {"c": 0.9, "s": 0.6}, mqtt_client=None)
    assert res["state"] in ("monitoring", "alarm")
    assert set(res["per_modality_pi"]) == {"c", "s"}
    assert all(0.0 <= v <= 1.0 for v in res["per_modality_pi"].values())
