"""§6 — Conflict-aware Dempster–Shafer evidential fusion.

Implements eq. (4)-(5) plus the belief-divergence fallback described in the source text.
Pure functions, no ML dependencies — unit-testable first, per the spec's build order.

Reliability coefficients r_m and the alarm threshold τ are NEVER hardcoded at call
sites: load them with load_decisional_config(), which layers
configs/default.yaml <- configs/decisional_calibration.yaml (written by
calibrate_decisional.py). The paper itself says r_m "must be recalibrated if deployment
region, sensor layout, or class prior changes".
"""
import copy
from itertools import product
from pathlib import Path

THETA = frozenset({"F", "not_F"})   # frame of discernment: {fire, not-fire}

# Module-level fallbacks (spec values); production code should use load_decisional_config().
DIVERGENCE_THRESHOLD = 0.3   # ASSUMPTION — [INFORMATION REQUIRED] in source; tune on val set
FIRE_ALARM_THRESHOLD = 0.78  # τ, per paper — recalibrated for FPR < 2% on THEIR val set.
                             # RECALIBRATE on your own val set before trusting it.

_DEFAULT_OPERATIONAL = {
    "timestamp_sync_tolerance_s": 2.0,
    "iiot_alignment_window_s": 30.0,
    "beat_period_s": 60.0,
    "packet_loss_policy": "mark_modality_inactive_after_2_missed_beats",
    "uav_unavailable_policy": "fuse_on_camera_and_iiot_only",
    "sensor_dropout_policy": "exclude_from_reliability_normalization_until_reconnect",
    "alarm_hysteresis": {"consecutive_ticks_required": 2, "cooldown_s": 300},
}


def mass_function(pi_hat: float, r_m: float) -> dict:
    """Eq. (4): m(F) = r*pi, m(¬F) = r*(1-pi), m(Θ) = 1 - r"""
    pi_hat = float(min(max(pi_hat, 0.0), 1.0))
    r_m = float(min(max(r_m, 0.0), 1.0))
    return {"F": r_m * pi_hat, "not_F": r_m * (1 - pi_hat), "Θ": 1 - r_m}


def _intersect(a: str, b: str):
    sets = {"F": {"F"}, "not_F": {"not_F"}, "Θ": {"F", "not_F"}}
    inter = sets[a] & sets[b]
    if not inter:
        return None
    if inter == {"F"}:
        return "F"
    if inter == {"not_F"}:
        return "not_F"
    return "Θ"


def dempster_combine(m_a: dict, m_b: dict):
    """Classical orthogonal sum, eq. (5). Returns (combined_mass, conflict_K)."""
    conflict = 0.0
    combined = {"F": 0.0, "not_F": 0.0, "Θ": 0.0}
    for (ka, va), (kb, vb) in product(m_a.items(), m_b.items()):
        target = _intersect(ka, kb)
        if target is None:
            conflict += va * vb
        else:
            combined[target] += va * vb
    norm = 1.0 - conflict
    if norm <= 1e-9:
        # total conflict — fall back to weighted average instead of dividing by ~0
        return weighted_average_combine(m_a, m_b), conflict
    return {k: v / norm for k, v in combined.items()}, conflict


def weighted_average_combine(m_a: dict, m_b: dict, w_a: float = 0.5, w_b: float = 0.5) -> dict:
    return {k: w_a * m_a.get(k, 0) + w_b * m_b.get(k, 0) for k in list(THETA) + ["Θ"]}


def belief_divergence(m_a: dict, m_b: dict) -> float:
    """A Jousselme-style pignistic distance between two mass functions.

    ASSUMPTION: the exact divergence measure is [INFORMATION REQUIRED] in the source
    (it only says "a belief-divergence measure" is computed); Jousselme distance is the
    standard, well-cited choice for conflict-aware DS fusion — swap in whatever the
    original authors actually used once you have that detail.
    """
    import math
    keys = ["F", "not_F", "Θ"]
    jaccard = {
        ("F", "F"): 1.0, ("not_F", "not_F"): 1.0, ("Θ", "Θ"): 1.0,
        ("F", "not_F"): 0.0, ("not_F", "F"): 0.0,
        ("F", "Θ"): 0.5, ("Θ", "F"): 0.5,
        ("not_F", "Θ"): 0.5, ("Θ", "not_F"): 0.5,
    }
    va = [m_a.get(k, 0) for k in keys]
    vb = [m_b.get(k, 0) for k in keys]
    num = sum(va[i] * vb[j] * jaccard[(keys[i], keys[j])] for i in range(3) for j in range(3))
    da = sum(va[i] * va[j] * jaccard[(keys[i], keys[j])] for i in range(3) for j in range(3))
    db = sum(vb[i] * vb[j] * jaccard[(keys[i], keys[j])] for i in range(3) for j in range(3))
    cos_sim = num / (math.sqrt(da * db) + 1e-9)
    return 1.0 - cos_sim   # 0 = identical, 1 = maximally divergent


def conflict_aware_fuse(masses: list, policy: str = "auto",
                        divergence_threshold: float | None = None):
    """Fold a list of per-modality masses pairwise, switching to weighted-average when
    pairwise belief-divergence exceeds the threshold.

    policy: "auto"      — divergence-gated switching (the paper's conflict-aware rule)
            "weighted"  — always weighted-average (ablation §11: no evidential aggregator)
            "dempster"  — always Dempster's rule (ablation §11: no conflict-aware fallback)
    """
    if policy not in ("auto", "weighted", "dempster"):
        raise ValueError(f"unknown fusion policy: {policy}")
    thr = DIVERGENCE_THRESHOLD if divergence_threshold is None else divergence_threshold
    fused = masses[0]
    divergences = []
    for m in masses[1:]:
        d = belief_divergence(fused, m)
        divergences.append(d)
        if policy == "weighted" or (policy == "auto" and d > thr):
            fused = weighted_average_combine(fused, m)
        else:
            fused, _ = dempster_combine(fused, m)
    return fused, divergences


def decide(masses: list, alarm_threshold: float | None = None,
           divergence_threshold: float | None = None, policy: str = "auto"):
    fused, divergences = conflict_aware_fuse(masses, policy=policy,
                                             divergence_threshold=divergence_threshold)
    tau = FIRE_ALARM_THRESHOLD if alarm_threshold is None else alarm_threshold
    alarm = fused["F"] > tau
    return alarm, {"fused_mass": fused, "pairwise_divergence": divergences}


# --------------------------------------------------------------------------- #
#  Config layering: defaults <- default.yaml <- decisional_calibration.yaml    #
# --------------------------------------------------------------------------- #
def load_decisional_config(config_path=None) -> dict:
    """Returns {"reliability": {c,d,s}, "fire_alarm_threshold": float,
    "divergence_threshold": float, "operational": {...}}.

    Layering: module defaults <- configs/default.yaml (decisional + operational
    sections) <- configs/decisional_calibration.yaml (written by calibrate_decisional.py).
    Works with no pyyaml installed (falls back to pure defaults with a warning).
    """
    merged = {
        "reliability": {"c": 0.80, "d": 0.80, "s": 0.60},   # ASSUMPTION fallbacks
        "fire_alarm_threshold": FIRE_ALARM_THRESHOLD,
        "divergence_threshold": DIVERGENCE_THRESHOLD,
        "operational": copy.deepcopy(_DEFAULT_OPERATIONAL),
    }
    root = Path(__file__).resolve().parents[2]
    candidates = [root / "configs" / "default.yaml"]
    if config_path is not None:
        candidates.append(Path(config_path))
    else:
        candidates.append(root / "configs" / "decisional_calibration.yaml")

    for path in candidates:
        if not path.exists():
            continue
        try:
            import yaml
        except ImportError:
            print(f"[evidential_fusion] pyyaml not installed; skipping {path.name}")
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        d = data.get("decisional", {}) if "decisional" in data else data
        if isinstance(d, dict):
            merged["reliability"].update(d.get("reliability", {}) or {})
            if "fire_alarm_threshold" in d:
                merged["fire_alarm_threshold"] = float(d["fire_alarm_threshold"])
            if "divergence_threshold" in d:
                merged["divergence_threshold"] = float(d["divergence_threshold"])
        if isinstance(data.get("operational"), dict):
            merged["operational"].update(data["operational"])
    return merged
