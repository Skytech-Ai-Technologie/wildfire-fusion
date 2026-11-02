"""§7.3 — MQTT/TLS transport for alarms (8 kB payload budget per the paper).

Cert provisioning / firmware versions are [INFORMATION REQUIRED] in the source —
document whatever mTLS setup you actually use. A self-signed CA is fine for a prototype;
note it explicitly in the repo README as not-production-hardened.
"""
import json

try:
    import paho.mqtt.client as mqtt
    _HAS_PAHO = True
except ImportError:  # pragma: no cover
    mqtt = None
    _HAS_PAHO = False

MAX_PAYLOAD_BYTES = 8 * 1024   # 8 kB budget per paper


class NullMqttClient:
    """Drop-in stand-in for paho's client so the fusion loop runs without a broker.
    Silent by design — a per-alarm print floods long simulations."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def publish(self, topic, payload=None, qos=0, retain=False):
        if self.verbose:
            print(f"[mqtt:null] {topic} ({len(payload or b'')} bytes, qos={qos})")

    def connect(self, *args, **kwargs):
        pass

    def loop_start(self):
        pass

    def disconnect(self):
        pass


def build_event_payload(cell_id: str, t: float, decision: dict) -> bytes:
    payload = {
        "cell_id": cell_id, "t": t,
        "alarm": decision["alarm"],
        "fused_mass": decision["fused_mass"],
        "divergence": decision["pairwise_divergence"],
    }
    data = json.dumps(payload).encode("utf-8")
    if len(data) > MAX_PAYLOAD_BYTES:
        # drop bulky fields (e.g. embeddings) rather than truncate JSON mid-string
        payload.pop("divergence", None)
        data = json.dumps(payload).encode("utf-8")
    return data


def publish_event(client, topic: str, payload: bytes, qos: int = 1):
    client.publish(topic, payload, qos=qos, retain=False)


def make_tls_client(client_id, ca_certs, certfile, keyfile):
    if not _HAS_PAHO:
        raise RuntimeError("paho-mqtt is not installed: pip install paho-mqtt")
    c = mqtt.Client(client_id=client_id)
    c.tls_set(ca_certs=ca_certs, certfile=certfile, keyfile=keyfile)
    return c
