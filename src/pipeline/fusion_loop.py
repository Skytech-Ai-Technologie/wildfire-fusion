"""§8 — Supervisory fusion loop, Algorithm 1.

fusion_tick() is the paper's Algorithm 1 shape: contextual refinement (eq. 3) ->
per-modality mass functions (eq. 4) -> conflict-aware DS combine (eq. 5) -> τ threshold
-> MQTT publish + supervisory queue (with UAV re-tasking suggestion).

FusionLoop() wraps it with the explicitly-unspecified operational policies the spec
lists as missing from the paper (all configurable via configs/default.yaml -> operational):

  * timestamp_sync_tolerance_s — drop frames whose capture timestamp drifts too far
    from tick time (ASSUMPTION: max clock skew tolerated across sensors = 2 s)
  * packet_loss_policy — mark modality inactive after N missed heartbeats
  * uav_unavailable_policy — suggest UAV dispatch only when the drone stream is absent
  * sensor_dropout_policy — documented no-op hook (reliability stats here are static,
    so "exclude from reliability normalization until reconnect" = don't update them
    during dropout; note_dropout() is where you'd wire a running-statistics scheme)
  * alarm_hysteresis — require 2 consecutive alarm ticks, latch until a human clears,
    then cool down before re-arming
"""
import queue
from dataclasses import dataclass, field

import torch

from src.decisional.evidential_fusion import mass_function, decide, load_decisional_config
from src.edge.mqtt_client import build_event_payload, publish_event, NullMqttClient


@dataclass
class ModalityFrame:
    name: str                 # "c" | "d" | "s"
    active: bool
    delta: list               # detected events
    pi: float                 # raw confidence
    embedding: torch.Tensor
    ts: float | None = None   # capture timestamp for sync checks (ASSUMPTION field)


def fusion_tick(cell_id, t, frames: list, contextual_encoder, reliability: dict,
                mqtt_client=None, supervisory_queue=None,
                alarm_threshold: float | None = None,
                divergence_threshold: float | None = None,
                decision_policy: str = "auto"):
    active = [f for f in frames if f.active]
    if not active:
        return {"state": "monitoring", "reason": "no_active_modalities"}

    # Algorithm 1 lines 3-4: contextual encoding -> refined confidences
    stacked = torch.stack([f.embedding for f in active]).unsqueeze(0)
    conf_in = torch.tensor([[f.pi for f in active]], dtype=torch.float32)
    refined_pi, _ = contextual_encoder(stacked, conf_in)
    refined = refined_pi.squeeze(0).tolist()

    # line 5: mass functions (eq. 4)
    masses = [mass_function(pi, reliability[f.name]) for f, pi in zip(active, refined)]

    # lines 6-7: conflict-aware DS combine + threshold
    alarm, info = decide(masses, alarm_threshold=alarm_threshold,
                         divergence_threshold=divergence_threshold, policy=decision_policy)
    info["per_modality_pi"] = {f.name: float(p) for f, p in zip(active, refined)}

    if alarm:
        # line 8: notify supervisory layer, optionally re-task UAV
        payload = build_event_payload(cell_id, t, {"alarm": True, **info})
        if mqtt_client is not None:
            publish_event(mqtt_client, f"wildfire/alarm/{cell_id}", payload)
        if supervisory_queue is not None:
            supervisory_queue.put({"cell_id": cell_id, "t": t, "info": info,
                                   "suggest_uav_dispatch":
                                       "d" not in [f.name for f in active]})
        return {"state": "alarm", **info}
    return {"state": "monitoring", **info}


class FusionLoop:
    """Stateful supervisory loop around fusion_tick with the §8 operational policies."""

    def __init__(self, cell_id, contextual_encoder, config_path=None, mqtt_client=None,
                 supervisory_queue: queue.Queue | None = None, decision_policy: str = "auto"):
        cfg = load_decisional_config(config_path)
        self.cell_id = cell_id
        self.encoder = contextual_encoder
        # fusion policy: "auto" (conflict-aware, the paper's rule) | "weighted" |
        # "dempster" — the two ablations of Table VI. Runtime-switchable so the
        # live platform can show the ablation without a restart.
        self.decision_policy = decision_policy
        self.reliability = cfg["reliability"]
        self.fire_alarm_threshold = cfg["fire_alarm_threshold"]
        self.divergence_threshold = cfg["divergence_threshold"]
        self.op = cfg["operational"]
        self.mqtt_client = mqtt_client if mqtt_client is not None else NullMqttClient()
        self.supervisory_queue = supervisory_queue if supervisory_queue is not None else queue.Queue()

        self.last_beat: dict[str, float] = {}
        self._consecutive = 0
        self._latched = False
        self._cooldown_until = -float("inf")
        self.dropped_modalities: list[str] = []

    # -- sensor lifecycle --------------------------------------------------- #
    def on_heartbeat(self, name: str, t: float):
        self.last_beat[name] = t

    def note_dropout(self, name: str, t: float):
        """sensor_dropout_policy hook: reliability stats are static here, so dropout
        handling = stop updating them until reconnect (documented in README)."""
        if name not in self.dropped_modalities:
            self.dropped_modalities.append(name)

    def clear_alarm(self, t: float):
        """Human clears the alarm -> cooldown before re-arming."""
        self._latched = False
        self._consecutive = 0
        self._cooldown_until = t + self.op["alarm_hysteresis"]["cooldown_s"]

    # -- per-tick policies --------------------------------------------------- #
    def _apply_packet_loss(self, f: ModalityFrame, t: float) -> ModalityFrame:
        beat = self.op["beat_period_s"]
        last = self.last_beat.get(f.name)
        missed = 2 if "2_missed_beats" in self.op["packet_loss_policy"] else 1
        if last is not None and (t - last) > missed * beat:
            if f.active:
                self.note_dropout(f.name, t)
            return ModalityFrame(f.name, False, f.delta, f.pi, f.embedding, f.ts)
        return f

    def tick(self, t: float, frames: list) -> dict:
        # 1. packet-loss policy
        frames = [self._apply_packet_loss(f, t) for f in frames]
        # 2. timestamp-sync policy
        tol = self.op["timestamp_sync_tolerance_s"]
        frames = [f for f in frames
                  if (not f.active) or f.ts is None or abs(t - f.ts) <= tol]
        # 3. Algorithm 1
        res = fusion_tick(self.cell_id, t, frames, self.encoder, self.reliability,
                          self.mqtt_client, self.supervisory_queue,
                          alarm_threshold=self.fire_alarm_threshold,
                          divergence_threshold=self.divergence_threshold,
                          decision_policy=self.decision_policy)
        # 4. alarm hysteresis
        req = self.op["alarm_hysteresis"]["consecutive_ticks_required"]
        raw_alarm = res.get("state") == "alarm"
        if raw_alarm:
            self._consecutive += 1
        else:
            self._consecutive = 0
        if raw_alarm and (self._consecutive >= req or self._latched):
            if t < self._cooldown_until:
                # suppressed during cooldown: do NOT re-latch, or a burst inside the
                # cooldown would let a single tick re-confirm right after it expires
                return {**res, "state": "suppressed_cooldown",
                        "consecutive_ticks": self._consecutive}
            self._latched = True
            return {**res, "state": "alarm_confirmed",
                    "consecutive_ticks": self._consecutive}
        return res
