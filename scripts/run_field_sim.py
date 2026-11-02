"""§9 — Synthetic field-trial replay: exercises the whole pipeline end to end.

You don't have Tunisia field hardware, but you can still exercise the fusion logic:

  * Camera/LWIR streams: labeled FIgLib sequences when downloaded + a trained
    perceptual checkpoint exist (--real), else a documented synthetic plume-growth
    confidence curve (default).
  * IIoT stream: baseline noise + an injected ramp in temp/PM/gas starting some offset
    (positive or negative) before/after the labeled ignition frame, mimicking the
    paper's finding that IIoT sometimes leads and sometimes lags visual detection.
  * Run FusionLoop over the synthetic timeline and record alarm time vs ground-truth
    ignition time -> the detection-horizon metric (§10.3) on YOUR setup, clearly
    labeled synthetic, not a claim about the real field trial.

Fusion configurations (for Table V):
    ptz_only | plus_lwir | plus_iiot (PTZ+LWIR+IIoT, the full system) | iiot_only

Usage:
    python scripts/run_field_sim.py --events 3
    python scripts/run_field_sim.py --config plus_iiot --events 5 --seed 7
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.pipeline.fusion_loop import FusionLoop, ModalityFrame  # noqa: E402
from src.contextual.gated_transformer import ContextualEncoder  # noqa: E402

CONFIG_MODALITIES = {
    "ptz_only": ["c"],
    "plus_lwir": ["c", "d"],
    "plus_iiot": ["c", "d", "s"],
    "iiot_only": ["s"],
}


class IdentityEncoder(torch.nn.Module):
    """Refined pi == raw pi. Used when no trained contextual checkpoint is supplied —
    an UNTRAINED ContextualEncoder would emit arbitrary confidences and no alarms would
    ever fire. Pass --checkpoint to exercise the real transformer."""

    def forward(self, x, conf):
        return conf, x


def confidence_curves(t: np.ndarray, ignition_s: float, rng: np.random.Generator) -> dict:
    """Documented synthetic confidence curves (no real-field claim):
    camera ramps ~45s after ignition, LWIR ~30s after (thermal contrast leads RGB in
    daylight smoke), IIoT ramp offset is a free parameter of the sim (lead or lag)."""
    def ramp(start, rise, ceiling):
        return ceiling / (1 + np.exp(-(t - start) / rise))
    cam = np.clip(0.03 + ramp(ignition_s + 45, 20, 0.9) + rng.normal(0, 0.02, t.shape), 0, 0.97)
    lwir = np.clip(0.04 + ramp(ignition_s + 30, 18, 0.92) + rng.normal(0, 0.02, t.shape), 0, 0.97)
    return {"c": cam, "d": lwir}


def iiot_curve(t: np.ndarray, ignition_s: float, offset_s: float, rng: np.random.Generator) -> np.ndarray:
    """IIoT confidence: baseline + linear ramp in temp/PM/gas beginning at
    ignition + offset (negative offset = sensor leads the visual detection)."""
    ramp = np.clip((t - (ignition_s + offset_s)) / 240.0, 0.0, 1.0)
    return np.clip(0.05 + 0.8 * ramp + rng.normal(0, 0.03, t.shape), 0.0, 0.97)


def simulate_event(event_id: int, modalities: list, encoder=None, duration_s=3600,
                   ignition_s=1800, iiot_offset_s=-120.0, seed=0, config_path=None) -> dict:
    if encoder is None:
        encoder = IdentityEncoder()
    rng = np.random.default_rng(seed + event_id)
    ts = np.arange(0, duration_s, dtype=float)
    curves = confidence_curves(ts, ignition_s, rng)
    curves["s"] = iiot_curve(ts, ignition_s, iiot_offset_s, rng)

    loop = FusionLoop(cell_id=f"sim-cell-{event_id}", contextual_encoder=encoder,
                      config_path=config_path)
    first_alarm = None
    rows = []
    d_model = 256
    base_vec = torch.randn(d_model)

    for t in ts:
        for name in modalities:
            loop.on_heartbeat(name, t)
        frames = []
        for name in modalities:
            pi = float(curves[name][int(t)])
            emb = base_vec * pi + torch.randn(d_model) * 0.1
            frames.append(ModalityFrame(name=name, active=True, delta=[pi > 0.5],
                                        pi=pi, embedding=emb, ts=t))
        res = loop.tick(float(t), frames)
        if res["state"] in ("alarm", "alarm_confirmed") and first_alarm is None:
            first_alarm = float(t)
            res_first = res
        rows.append({"t": t, **{f"pi_{m}": float(curves[m][int(t)]) for m in modalities},
                     "state": res["state"],
                     "fused_F": res.get("fused_mass", {}).get("F")})

    horizon = None if first_alarm is None else ignition_s - first_alarm  # >0 = earlier
    return {"event_id": event_id, "ignition_s": ignition_s, "first_alarm_s": first_alarm,
            "detection_horizon_s": horizon, "config": "+".join(modalities),
            "iiot_offset_s": iiot_offset_s, "seed": seed,
            "fused_mass_at_alarm": res_first.get("fused_mass") if first_alarm is not None else None,
            "_rows": rows}


def main():
    ap = argparse.ArgumentParser(description="§9 synthetic field-trial replay")
    ap.add_argument("--config", choices=list(CONFIG_MODALITIES), default="plus_iiot")
    ap.add_argument("--events", type=int, default=3)
    ap.add_argument("--duration", type=float, default=3600)
    ap.add_argument("--ignition", type=float, default=1800)
    ap.add_argument("--iiot-offset", type=float, default=-120.0,
                    help="IIoT ramp start relative to ignition (negative = leads)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/field_sim")
    ap.add_argument("--calibration", default=None,
                    help="path to configs/decisional_calibration.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="trained contextual encoder state_dict; default = identity refinement")
    args = ap.parse_args()

    cfg = __import__("src.decisional.evidential_fusion", fromlist=["load_decisional_config"]) \
        .load_decisional_config(args.calibration)
    if args.checkpoint:
        enc = ContextualEncoder(d_model=256, n_heads=8, n_layers=4, dropout=0.1)
        state = torch.load(args.checkpoint, map_location="cpu")
        enc.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
        enc.eval()
    else:
        print("[field_sim] no --checkpoint given -> identity contextual refinement "
              "(untrained encoders produce arbitrary confidences)")

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    modalities = CONFIG_MODALITIES[args.config]

    events = []
    encoder = enc if args.checkpoint else None
    with open(out / f"timeline_{args.config}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for i in range(args.events):
            ev = simulate_event(i, modalities, encoder, duration_s=args.duration,
                                ignition_s=args.ignition, iiot_offset_s=args.iiot_offset,
                                seed=args.seed, config_path=args.calibration)
            events.append({k: v for k, v in ev.items() if k != "_rows"})
            if i == 0:
                w.writerow(["event_id"] + list(ev["_rows"][0].keys()))
            for r in ev["_rows"]:
                w.writerow([i] + list(r.values()))

    horizons = [e["detection_horizon_s"] for e in events if e["detection_horizon_s"] is not None]
    summary = {
        "config": args.config, "modalities": modalities, "n_events": args.events,
        "detection_horizon_mean_s": float(np.mean(horizons)) if horizons else None,
        "detection_horizon_std_s": float(np.std(horizons)) if horizons else None,
        "alarm_rate": len(horizons) / args.events,
        "SYNTHETIC_LABEL": "synthetic replay per spec §9 — NOT a real field trial; "
                           "paper reports 86s ± 14s across only 3 events (do not oversell N=3)",
    }
    (out / f"summary_{args.config}.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
