"""§4.4 — IIoT anomaly branch: isolation-forest anomaly score over env. node readings.

Feature vector per reading: temp_c, rh_pct, pm25, pm10, gas_mq2 (the paper's IIoT node
sensors). The model is fit on >=12 months of normal-history readings; at fusion time a
reading is squashed to a [0,1] confidence-like value.

ASSUMPTION (paper gap): sampling every 60s to align with FIgLib's ~60s cadence; resample/
forward-fill gaps < 5 min, drop gaps >= 5 min as missing (documented policy — the paper
does not specify one, this is ours).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["temp_c", "rh_pct", "pm25", "pm10", "gas_mq2"]


def resample_history(history_df: pd.DataFrame, sample_interval_s: int = 60,
                     max_gap_s: int = 300) -> pd.DataFrame:
    """Resample to a fixed cadence, forward-filling gaps < max_gap_s. Gaps >= max_gap_s
    are treated as sensor outages: they are NOT forward-filled across — the timeline is
    split into segments at each outage and each segment resampled independently
    (documented policy — the paper does not specify one, this is ours).

    history_df must have a 't' column (epoch seconds, UTC) and the FEATURES columns.
    """
    df = history_df.sort_values("t").drop_duplicates(subset="t", keep="last")
    ts = df["t"].to_numpy()
    # segment boundaries: first row, every row that starts a >=max_gap_s outage, end
    starts = [0] + [i for i in range(1, len(ts)) if ts[i] - ts[i - 1] >= max_gap_s]
    bounds = starts + [len(ts)]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = df.iloc[a:b]
        if seg.empty:
            continue
        grid = pd.DataFrame({"t": np.arange(seg["t"].min(), seg["t"].max() + sample_interval_s,
                                            sample_interval_s)})
        segs.append(grid.merge(seg, on="t", how="left").ffill())
    out = pd.concat(segs, ignore_index=True) if segs else df.iloc[0:0]
    return out.dropna(subset=FEATURES)


def fit_iiot_anomaly_model(history_df: pd.DataFrame, contamination: float = 0.02,
                           n_estimators: int = 200, random_state: int = 42):
    """history_df: >=12 months of readings, one row per node per sample interval."""
    X = history_df[FEATURES].copy()
    mu, sigma = X.mean(), X.std().replace(0, 1)
    X_norm = (X - mu) / sigma                      # z-score normalization — ASSUMPTION
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                            random_state=random_state)
    model.fit(X_norm.to_numpy())                   # fit on ndarray: live vector is an ndarray too
    return model, {"mu": mu, "sigma": sigma}


def iiot_confidence(model, norm_stats, live_reading: dict) -> float:
    x = np.array([[(live_reading[f] - norm_stats["mu"][f]) / norm_stats["sigma"][f]
                   for f in FEATURES]])
    raw = -model.score_samples(x)[0]               # higher = more anomalous
    return float(np.clip((raw + 0.5), 0.0, 1.0))   # squash to [0,1] confidence-like value


def align_reading(frame_ts: float, readings_df: pd.DataFrame,
                  tolerance_s: float = 30.0) -> dict | None:
    """§4.4 frame/reading alignment.

    Tag every camera frame and every IIoT reading with a UTC timestamp at capture time;
    at fusion time, pull the IIoT reading whose timestamp is closest to the frame
    timestamp within a ±tolerance_s window (ASSUMPTION — half the FIgLib sampling
    interval). If no reading falls inside the window, return None so the caller sets the
    's' modality inactive for that tick rather than interpolating across a gap.
    """
    if readings_df is None or len(readings_df) == 0:
        return None
    diffs = (readings_df["t"] - frame_ts).abs()
    idx = diffs.idxmin()
    if float(diffs.loc[idx]) > tolerance_s:
        return None
    return readings_df.loc[idx].to_dict()
