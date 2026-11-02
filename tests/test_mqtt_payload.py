"""§7.3 — MQTT payload budget tests (8 kB cap, drop-bulky-fields policy)."""
import json

from src.edge.mqtt_client import build_event_payload, MAX_PAYLOAD_BYTES, NullMqttClient


def _decision(div_len=10):
    return {"alarm": True,
            "fused_mass": {"F": 0.9, "not_F": 0.05, "Θ": 0.05},
            "pairwise_divergence": [0.01] * div_len}


def test_payload_within_budget():
    data = build_event_payload("cell-1", 123.0, _decision())
    assert len(data) <= MAX_PAYLOAD_BYTES
    assert json.loads(data)["cell_id"] == "cell-1"


def test_oversized_payload_drops_divergence_not_truncate():
    data = build_event_payload("cell-1", 123.0, _decision(div_len=20000))
    assert len(data) <= MAX_PAYLOAD_BYTES
    payload = json.loads(data)
    assert "divergence" not in payload          # bulky field dropped whole
    assert "fused_mass" in payload              # decision-critical fields kept


def test_null_client_smoke():
    NullMqttClient().publish("wildfire/alarm/cell-1", b"x", qos=1)
