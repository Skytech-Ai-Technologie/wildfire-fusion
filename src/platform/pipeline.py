"""Live pipeline: camera -> perceptual -> contextual -> decisional -> UI.

This is the repo's four-layer architecture running in real time on one PC:

    Perceptual   PerceptionEngine (src/platform/engine.py) — trained dual-stream
                 YOLOv8n+CBAM, or a single-stream Ultralytics checkpoint
    Contextual   WindowedContextualRefiner — the trained entropy-gated transformer,
                 fed a rolling window of the last T frames (the semantics it was
                 trained with: T = temporal frames of one event)
    Decisional   src/pipeline/fusion_loop.py — conflict-aware Dempster-Shafer fusion,
                 threshold, hysteresis, MQTT publish, supervisory queue
    Edge/ops     the annotated MJPEG frame + JSON telemetry

Two documented deviations from the offline replay path (scripts/run_field_sim.py):

  1. `fusion_tick()` stacks the modalities along the encoder's time axis, so the
     trained contextual transformer sees M modalities where it was trained on 32
     temporal frames. Here each modality owns its own temporal window and the
     encoder is run once per modality, which is the semantics the checkpoint was
     trained with. The refined confidence vector still has shape (1, M), so
     fusion_tick() and every policy in it are used unmodified.
  2. The LWIR half of the dual-stream input is the grayscale proxy of the RGB
     frame (a webcam has no thermal sensor). It is labelled as such everywhere.
"""
import math
import threading
import time

import numpy as np
import torch

from src.pipeline.fusion_loop import FusionLoop, ModalityFrame
from src.platform.camera import SourceManager, make_source, _cv2

CLASSES = ["smoke", "fire"]

# repo palette (README / docs/index.html) so the UI and the video agree
C_FIRE = (77, 51, 204)        # BGR  #cc334d
C_SMOKE = (0, 140, 215)       # BGR  #d78c00
C_OK = (85, 128, 0)           # BGR  #008055
C_ALARM = (77, 51, 204)
C_PANEL = (32, 32, 32)
C_TEXT = (240, 240, 240)


class WindowedContextualRefiner(torch.nn.Module):
    """Runs the trained entropy-gated transformer over a rolling temporal window
    per modality slot.

    `push(slot, embedding, pi)` appends one frame to that slot's window;
    `forward(x, conf)` (the interface fusion_tick() expects) returns the refined
    confidence for every slot, taking the last timestep of each window.

    With fewer than T frames the window is left-padded by repeating the earliest
    frame — the same padding rule as `build_windows()` in src/train_contextual.py.
    """

    def __init__(self, encoder, T=32):
        super().__init__()
        self.encoder = encoder
        self.T = T
        self._windows = {}

    def push(self, slot, embedding, pi):
        if embedding is None:
            return
        w = self._windows.setdefault(slot, [])
        w.append((torch.as_tensor(embedding, dtype=torch.float32).flatten(),
                  float(pi)))
        if len(w) > self.T:
            del w[0]

    def filled(self, slot):
        return len(self._windows.get(slot, []))

    def _device(self):
        for p in self.encoder.parameters():
            return p.device
        return torch.device("cpu")

    def clear(self):
        self._windows.clear()

    @torch.no_grad()
    def forward(self, x, conf):
        M = x.shape[1]
        dev = self._device()
        outs = []
        for i in range(M):
            w = self._windows.get(i)
            if not w:
                outs.append(float(conf[0, i]))
                continue
            E = torch.stack([e for e, _ in w])                      # (n, d)
            C = torch.tensor([[p for _, p in w]], dtype=torch.float32)
            if E.shape[0] < self.T:
                pad = self.T - E.shape[0]
                E = torch.cat([E[:1].repeat(pad, 1), E], dim=0)
                C = torch.cat([C[:, :1].repeat(1, pad), C], dim=1)
            refined, _ = self.encoder(E.unsqueeze(0).to(dev), C.to(dev))
            outs.append(float(refined[0, -1]))
        return torch.tensor([outs], dtype=torch.float32), x


class IiotSimulator:
    """Standalone SIMULATED IIoT node (temp/RH/PM/gas).

    Independent of the camera on purpose: the simulated environmental signal
    ramps on its own schedule, the way scripts/run_field_sim.py injects one, so
    the multi-modal conflict path in the DS fusion is exercised without the two
    modalities simply echoing each other.

    The paper's IIoT branch is an isolation forest fitted on >=12 months of
    history (src/perceptual/iiot_branch.py). No such history ships with this
    repo, so this simulator uses a transparent z-score composite instead. Every
    value it produces is labelled SIMULATED in the UI.
    """

    BASELINE = {"temp_c": 22.0, "rh_pct": 45.0, "pm25": 8.0, "pm10": 14.0, "gas_mq2": 0.4}
    SCALE = {"temp_c": 6.0, "rh_pct": 8.0, "pm25": 25.0, "pm10": 30.0, "gas_mq2": 0.8}

    def __init__(self, seed=0, event_period_s=150.0, event_ramp_s=90.0):
        self.rng = np.random.default_rng(seed)
        self.event_period_s = event_period_s
        self.event_ramp_s = event_ramp_s
        self.t = 0.0
        self.values = dict(self.BASELINE)

    def step(self, dt):
        """Mean-reverting walk toward a baseline, offset by the heat event.

        The event contributes a TARGET OFFSET, not a per-step increment — adding it
        every step would make the readings diverge without bound. The mean reversion
        pulls each channel back toward its target, so the walk stays bounded.
        """
        self.t += dt
        phase = (self.t % self.event_period_s) / self.event_period_s
        ramp = (phase - 0.25) / 0.30 if 0.25 < phase < 0.55 else 0.0
        offsets = {"temp_c": 2.6 * ramp, "pm25": 9.0 * ramp, "pm10": 11.0 * ramp,
                   "gas_mq2": 0.28 * ramp, "rh_pct": -4.0 * ramp}
        for k, base in self.BASELINE.items():
            target = base + offsets.get(k, 0.0)
            drift = (target - self.values[k]) * 0.08
            noise = float(self.rng.normal(0, 0.25)) * self.SCALE[k] * 0.05
            self.values[k] = max(0.0, self.values[k] + drift + noise)

    def anomaly_confidence(self):
        """Transparent z-score composite -> [0, 1]. Stands in for the isolation
        forest's anomaly score (not fitted here — no history data in the repo)."""
        z = 0.0
        for k, base in self.BASELINE.items():
            z += abs(self.values[k] - base) / self.SCALE[k]
        z /= len(self.BASELINE)
        return float(np.clip(z / 0.6, 0.0, 1.0))

    def embedding(self, dim=256):
        """Deterministic placeholder embedding so the slot fits the encoder's
        256-d input. A fixed seeded projection of the five readings scaled by the
        anomaly score — a stand-in for the paper's per-node embedding."""
        feats = np.array([(self.values[k] - self.BASELINE[k]) / self.SCALE[k]
                          for k in self.BASELINE], dtype=np.float32)
        R = np.random.default_rng(1234).standard_normal((5, dim)).astype(np.float32)
        vec = feats @ R
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec = vec / n
        return (vec * self.anomaly_confidence()).astype(np.float32)


class LivePipeline:
    """Owns the capture source, the engine, the fusion loop and the shared state."""

    def __init__(self, engine, refiner, loop, source, cell_id="live-cell",
                 secondary_source=None, use_iiot=False, config_path=None,
                 source_config=None):
        self.engine = engine
        self.refiner = refiner
        self.loop = loop
        self.source = source
        self.manager = SourceManager(source, secondary_source)
        self.secondary_source = secondary_source
        self.use_iiot = use_iiot
        self.cell_id = cell_id
        self.config_path = config_path
        self.iiot = IiotSimulator() if use_iiot else None
        # how to rebuild a source when the user switches input at runtime
        self.source_config = source_config or {}

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._frame_jpeg = None
        self._frame_jpeg_b = None
        self._frame_seq = 0
        self.state = {
            "running": False, "state": "idle", "alarm": False,
            "fused_mass": {"F": 0.0, "not_F": 0.0, "Θ": 1.0},
            "tau": loop.fire_alarm_threshold,
            "divergence": [], "modalities": [], "detections": [],
            "fps": 0.0, "latency_ms": 0.0, "error": None,
            "frame_seq": 0, "window": {"T": getattr(refiner, "T", 32), "filled": 0},
        }
        self._fps_hist = []
        self._last_tick = None

    # -- lifecycle ---------------------------------------------------------- #
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self.manager.start()
        self._thread = threading.Thread(target=self._run, name="live-pipeline", daemon=True)
        self._thread.start()
        with self._lock:
            self.state["running"] = True
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.manager.stop()
        with self._lock:
            self.state["running"] = False
            self.state["state"] = "stopped"

    def set_source(self, kind, value=None):
        """Swap the active input at runtime: "camera" | "file" | "youtube" |
        "synthetic". The model, the fusion loop and the server keep running."""
        cfg = dict(self.source_config)
        new_source = make_source(kind, value, **cfg)
        self.manager.swap(new_source)
        return new_source

    def active_camera_indices(self):
        """Device indices the platform currently holds open.

        The UI's device scanner must not probe these: a second handle on the same
        DirectShow device can stall the live capture."""
        from src.platform.camera import CameraSource

        out = set()
        for src in (self.manager.current, self.manager.secondary):
            if isinstance(src, CameraSource):
                out.add(src.index)
        return out

    # -- main loop ---------------------------------------------------------- #
    def _run(self):
        last_seq = -1
        while not self._stop.is_set():
            t0 = time.time()
            frame, ts, seq = self.manager.read()
            if frame is None:
                err = self.manager.describe().get("error")
                with self._lock:
                    if err:
                        self.state["error"] = err
                time.sleep(0.05)
                continue
            if seq == last_seq:
                # the source has not published anything new — don't spend GPU on a
                # duplicate frame
                time.sleep(0.005)
                continue
            last_seq = seq
            try:
                self._process(frame, ts)
            except Exception as e:  # noqa: BLE001 — a bad frame must not kill the loop
                with self._lock:
                    self.state["error"] = f"{type(e).__name__}: {e}"
                time.sleep(0.05)
                continue
            # pace to the requested fps
            src = self.manager.current
            budget = 1.0 / max(src.fps if src else 8, 1)
            elapsed = time.time() - t0
            if elapsed < budget:
                time.sleep(budget - elapsed)

    def _process(self, frame, ts):
        t = time.time()
        res = self.engine.infer(frame)
        dets = res["detections"]
        emb = res["embedding"]
        raw_pi = res["pi_head"]

        frames = []
        modal = []
        frame_age_ms = max(0.0, (t - ts) * 1000.0)

        # --- slot 0: the PC camera (modality "c") -------------------------- #
        self.refiner.push(0, emb, raw_pi)
        frames.append(ModalityFrame(name="c", active=True,
                                    delta=[bool(d) for d in dets],
                                    pi=raw_pi, embedding=self._as_tensor(emb),
                                    ts=ts))
        modal.append({"name": "c", "label": "PC camera (RGB)", "active": True,
                      "raw_pi": raw_pi, "refined_pi": None, "simulated": False,
                      "source": self.manager.current.name if self.manager.current else "-",
                      "detections": len(dets), "frame_age_ms": round(frame_age_ms, 1),
                      "window": self.refiner.filled(0)})

        # --- slot 1: optional second camera (modality "d") ----------------- #
        if self.secondary_source is not None:
            sframe, sts, sseq = self.secondary_source.read()
            if sframe is not None:
                sres = self.engine.infer(sframe)
                self.refiner.push(1, sres["embedding"], sres["pi_head"])
                frames.append(ModalityFrame(name="d", active=True,
                                            delta=[bool(d) for d in sres["detections"]],
                                            pi=sres["pi_head"],
                                            embedding=self._as_tensor(sres["embedding"]),
                                            ts=sts))
                modal.append({"name": "d", "label": "Camera B (RGB, not thermal)",
                              "active": True, "raw_pi": sres["pi_head"],
                              "refined_pi": None, "simulated": False,
                              "source": self.secondary_source.name,
                              "detections": len(sres["detections"]),
                              "window": self.refiner.filled(1)})
                sframe_out = self._render(sframe, sres["detections"], None)
                cv2 = _cv2()
                ok_b, buf_b = cv2.imencode(".jpg", sframe_out,
                                          [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok_b:
                    with self._lock:
                        self._frame_jpeg_b = buf_b.tobytes()

        # --- slot 2: optional simulated IIoT (modality "s") ----------------- #
        if self.iiot is not None:
            self.iiot.step(max(0.0, t - (self._last_tick or t)))
            self._last_tick = t
            pi_s = self.iiot.anomaly_confidence()
            emb_s = self.iiot.embedding()
            self.refiner.push(2, emb_s, pi_s)
            frames.append(ModalityFrame(name="s", active=True, delta=[pi_s > 0.5],
                                        pi=pi_s,
                                        embedding=self._as_tensor(emb_s),
                                        ts=t))
            modal.append({"name": "s", "label": "IIoT node (SIMULATED)",
                          "active": True, "raw_pi": pi_s, "refined_pi": None,
                          "simulated": True, "source": "simulated", "detections": 0,
                          "window": self.refiner.filled(2),
                          "readings": {k: round(v, 2) for k, v in self.iiot.values.items()}})

        # --- heartbeats + fusion (the repo's Algorithm 1) ------------------- #
        for f in frames:
            self.loop.on_heartbeat(f.name, t)
        out = self.loop.tick(t, frames)

        refined = out.get("per_modality_pi", {})
        for m in modal:
            m["refined_pi"] = refined.get(m["name"])

        mass = out.get("fused_mass", {"F": 0.0, "not_F": 0.0, "Θ": 1.0})
        state_name = out.get("state", "monitoring")
        alarm = state_name in ("alarm", "alarm_confirmed")

        annotated = self._render(frame, dets, {
            "state": state_name, "mass": mass, "tau": self.loop.fire_alarm_threshold,
            "latency_ms": res["latency_ms"],
        })
        cv2 = _cv2()
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            with self._lock:
                self._frame_jpeg = buf.tobytes()
                self._frame_seq += 1

        now = time.time()
        self._fps_hist.append(now)
        if len(self._fps_hist) > 30:
            self._fps_hist.pop(0)
        fps = 0.0
        if len(self._fps_hist) > 1:
            span = self._fps_hist[-1] - self._fps_hist[0]
            fps = (len(self._fps_hist) - 1) / span if span > 0 else 0.0

        with self._lock:
            self.state.update({
                "state": state_name, "alarm": alarm,
                "fused_mass": {k: float(v) for k, v in mass.items()},
                "tau": float(self.loop.fire_alarm_threshold),
                "divergence": [float(d) for d in out.get("pairwise_divergence", [])],
                "modalities": modal, "detections": dets,
                "fps": round(fps, 2),
                "latency_ms": round(res["latency_ms"], 1),
                "error": None, "frame_seq": self._frame_seq,
                "window": {"T": getattr(self.refiner, "T", 32),
                           "filled": self.refiner.filled(0)},
            })

    @staticmethod
    def _as_tensor(v):
        if v is None:
            return torch.zeros(256, dtype=torch.float32)
        return torch.as_tensor(v, dtype=torch.float32).flatten()

    # -- rendering ---------------------------------------------------------- #
    def _render(self, frame, dets, info):
        cv2 = _cv2()

        img = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            color = C_FIRE if d["name"] == "fire" else C_SMOKE
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            txt = f"{d['name']} {d['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, max(0, y1 - 1)),
                          color, -1)
            cv2.putText(img, txt, (x1 + 4, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if info:
            alarm = info["state"] in ("alarm", "alarm_confirmed")
            banner_c = C_ALARM if alarm else C_OK
            cv2.rectangle(img, (0, 0), (img.shape[1], 34), C_PANEL, -1)
            cv2.rectangle(img, (0, 0), (8, 34), banner_c, -1)
            mf = info["mass"].get("F", 0.0)
            txt = (f"m(F)={mf:.3f}   tau={info['tau']:.2f}   "
                   f"latency={info['latency_ms']:.0f}ms")
            cv2.putText(img, txt, (18, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        C_TEXT, 1, cv2.LINE_AA)
            # state word right-aligned so it never collides with the metrics
            label = "ALARM" if alarm else info["state"].upper()
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(img, label, (img.shape[1] - tw - 14, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_c, 2, cv2.LINE_AA)
        return img

    # -- accessors ---------------------------------------------------------- #
    def frame_jpeg(self, which="a"):
        with self._lock:
            if which == "b":
                return self._frame_jpeg_b, self._frame_seq
            return self._frame_jpeg, self._frame_seq

    def status(self):
        with self._lock:
            s = dict(self.state)
            s["modalities"] = list(s.get("modalities", []))
            # the threshold can be changed at runtime -> report the live value, not
            # the one cached with the last processed frame
            s["tau"] = float(self.loop.fire_alarm_threshold)
            s["device"] = self.engine._device
            s["arch"] = self.engine.arch
            s["lwir_source"] = self.engine.lwir_source
            s["weights"] = str(self.engine.weights)
            s["cell_id"] = self.cell_id
            src = self.manager.describe()
            s["source"] = src.get("name", "-")
            s["source_kind"] = src.get("kind", "none")
            s["source_info"] = src.get("info", "")
            s["source_status"] = src.get("status", "")
            s["source_error"] = src.get("error")
            s["contextual_loaded"] = bool(getattr(self.refiner, "contextual_loaded", False))
            s["iiot_simulated"] = self.iiot is not None
            return s

    # -- controls ----------------------------------------------------------- #
    def set_tau(self, tau):
        self.loop.fire_alarm_threshold = float(max(0.0, min(0.999, tau)))

    def set_conf(self, conf):
        self.engine.set_thresholds(conf=conf)

    def set_policy(self, policy):
        if policy in ("auto", "weighted", "dempster"):
            self.loop.decision_policy = policy

    def clear_alarm(self):
        self.loop.clear_alarm(time.time())
