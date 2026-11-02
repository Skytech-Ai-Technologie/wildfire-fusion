"""§4.4 — IIoT isolation-forest branch tests."""
import pytest

np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")
pd = pytest.importorskip("pandas")

from src.perceptual.iiot_branch import (fit_iiot_anomaly_model, iiot_confidence,  # noqa: E402
                                        align_reading, resample_history, FEATURES)


def _history(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "t": np.arange(n) * 60.0,
        "temp_c": rng.normal(25, 3, n),
        "rh_pct": rng.normal(50, 10, n),
        "pm25": rng.normal(10, 3, n),
        "pm10": rng.normal(20, 5, n),
        "gas_mq2": rng.normal(100, 15, n),
    })


def test_anomalous_reading_scores_higher():
    model, stats = fit_iiot_anomaly_model(_history())
    normal = {f: float(_history(10, seed=1)[f].mean()) for f in FEATURES}
    wildfire = {"temp_c": 60.0, "rh_pct": 5.0, "pm25": 300.0, "pm10": 500.0,
                "gas_mq2": 900.0}
    c_normal = iiot_confidence(model, stats, normal)
    c_fire = iiot_confidence(model, stats, wildfire)
    assert c_fire > c_normal, "wildfire-like IIoT reading must score as more anomalous"
    assert 0.0 <= c_normal <= 1.0 and 0.0 <= c_fire <= 1.0


def test_align_reading_tolerance():
    df = pd.DataFrame({"t": [1000.0, 2000.0], "temp_c": [1, 2], "rh_pct": [1, 2],
                       "pm25": [1, 2], "pm10": [1, 2], "gas_mq2": [1, 2]})
    assert align_reading(1010.0, df, tolerance_s=30.0)["t"] == 1000.0
    assert align_reading(1500.0, df, tolerance_s=30.0) is None   # outside ±30s window


def test_resample_history_drops_big_gaps():
    df = pd.DataFrame({
        "t": [0.0, 60.0, 120.0, 3600.0, 3660.0],   # 58-minute outage
        "temp_c": [1, 2, 3, 4, 5], "rh_pct": [1, 2, 3, 4, 5], "pm25": [1, 2, 3, 4, 5],
        "pm10": [1, 2, 3, 4, 5], "gas_mq2": [1, 2, 3, 4, 5],
    })
    out = resample_history(df, sample_interval_s=60, max_gap_s=300)
    assert out["t"].max() == 3660.0                      # post-outage data kept
    assert not ((out["t"] > 120.0) & (out["t"] < 3600.0)).any()  # outage NOT forward-filled
