"""§11 — Ablations reproducing Table VI, with the paper's missing statistics added:
deltas vs the full system + bootstrap CIs on each delta.

Variants (everything else held constant):
  no_cbam             — needs a retrained no-CBAM perceptual checkpoint (--no-cbam-weights);
                        fusion-level delta is not measurable for this variant, so it is
                        reported separately / skipped with a message.
  concat_transformer  — ContextualEncoder replaced by ConcatFusionEncoder (cat + MLP).
  weighted_avg_only   — force conflict_aware_fuse to always weighted-average.
  dempster_only       — force conflict_aware_fuse to always use Dempster's rule.

Usage:
  python benchmark/ablation.py --variants full concat_transformer weighted_avg_only dempster_only
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark.metrics import bootstrap_ci  # noqa: E402
from scripts.run_field_sim import (simulate_event, CONFIG_MODALITIES,  # noqa: E402
                                   IdentityEncoder)
from src.contextual.gated_transformer import ContextualEncoder, ConcatFusionEncoder  # noqa: E402


def run_variant(variant: str, events: int, seed: int) -> dict:
    """Returns per-event detection horizons for one ablation variant (SYNTHETIC replay).

    All variants run under identity contextual refinement, so the measured deltas come
    from the FUSION POLICY alone. The encoder-architecture ablation (transformer vs
    concat) is only meaningful with trained checkpoints — pass them via --encoder-a /
    --encoder-b when you have them; without weights this variant is skipped.
    """
    policy = {"weighted_avg_only": "weighted", "dempster_only": "dempster"}.get(variant, "auto")
    if variant == "concat_transformer":
        print("[concat_transformer] needs trained checkpoints for a fair comparison; "
              "skipped in the checkpoint-less run (see run_variant docstring).")
        return {"variant": variant, "horizons": None}
    if variant == "no_cbam":
        print("[no_cbam] requires a retrained no-CBAM perceptual checkpoint "
              "(src/train_perceptual.py --arch dual --no-cbam) wired into the sim; "
              "skipping rather than silently re-running the full system.")
        return {"variant": variant, "horizons": None}

    # simulate_event uses FusionLoop with policy=auto; for policy ablations we monkey-
    # patch the decide() threshold path via the loop's decision_policy argument.
    horizons = []
    for i in range(events):
        ev = _simulate_with_policy(i, None, policy, seed)
        if ev is not None:
            horizons.append(ev)
    return {"variant": variant, "horizons": horizons}


def _simulate_with_policy(event_id, encoder, policy, seed):
    """Copy of simulate_event with a decision_policy override (keeps run_field_sim
    untouched for normal use)."""
    from src.pipeline.fusion_loop import FusionLoop, ModalityFrame
    import src.pipeline.fusion_loop as fl

    encoder = encoder or IdentityEncoder()
    rng = np.random.default_rng(seed + event_id)
    duration_s, ignition_s, offset_s = 3600, 1800, -120.0
    ts = np.arange(0, duration_s, dtype=float)
    from scripts.run_field_sim import confidence_curves, iiot_curve
    curves = confidence_curves(ts, ignition_s, rng)
    curves["s"] = iiot_curve(ts, ignition_s, offset_s, rng)

    orig_decide = fl.decide
    override = policy if policy in ("weighted", "dempster") else "auto"

    def patched_decide(masses, alarm_threshold=None, divergence_threshold=None, policy="auto"):
        # fusion_tick always passes its own (auto) policy; the ablation overrides it.
        return orig_decide(masses, alarm_threshold=alarm_threshold,
                           divergence_threshold=divergence_threshold,
                           policy=override if override != "auto" else policy)

    fl.decide = patched_decide
    try:
        loop = FusionLoop(cell_id=f"abl-{event_id}", contextual_encoder=encoder)
        first_alarm = None
        base_vec = torch.randn(256)
        for t in ts:
            for name in CONFIG_MODALITIES["plus_iiot"]:
                loop.on_heartbeat(name, t)
            frames = [ModalityFrame(name=n, active=True, delta=[],
                                    pi=float(curves[n][int(t)]),
                                    embedding=base_vec * float(curves[n][int(t)])
                                    + torch.randn(256) * 0.1, ts=t)
                      for n in CONFIG_MODALITIES["plus_iiot"]]
            res = loop.tick(float(t), frames)
            if res["state"] in ("alarm", "alarm_confirmed") and first_alarm is None:
                first_alarm = float(t)
        return None if first_alarm is None else ignition_s - first_alarm
    finally:
        fl.decide = orig_decide


def main():
    ap = argparse.ArgumentParser(description="§11 ablations")
    ap.add_argument("--variants", nargs="+",
                    default=["full", "concat_transformer", "weighted_avg_only", "dempster_only"])
    ap.add_argument("--events", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ablation.csv")
    args = ap.parse_args()

    results = {v: run_variant(v, args.events, args.seed) for v in args.variants}
    full = results.get("full", {}).get("horizons") or []

    rows = []
    for v, r in results.items():
        if r["horizons"] is None:
            rows.append({"variant": v, "mean_horizon_s": None, "delta_vs_full_s": None,
                         "delta_ci": None})
            continue
        mean_h = float(np.mean(r["horizons"])) if r["horizons"] else None
        if v == "full":
            delta, ci = 0.0, bootstrap_ci(r["horizons"])
        else:
            n = min(len(full), len(r["horizons"]))
            deltas = [f - g for f, g in zip(full[:n], r["horizons"][:n])] if n else []
            delta = float(np.mean(deltas)) if deltas else None
            ci = bootstrap_ci(deltas) if deltas else None
        rows.append({"variant": v, "mean_horizon_s": mean_h,
                     "delta_vs_full_s": delta, "delta_ci": ci})

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "mean_horizon_s", "delta_vs_full_s",
                                          "delta_ci"])
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, float, int)) else v
                        for k, v in r.items()})
    print(f"Table VI ablation -> {out}  (bootstrap CIs fix the paper's missing "
          "significance analysis; SYNTHETIC replay)")
    for r in rows:
        print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
