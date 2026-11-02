"""§6 — decisional tier unit tests (pure functions, no ML deps)."""
import pytest

from src.decisional.evidential_fusion import (
    THETA, mass_function, dempster_combine, weighted_average_combine,
    belief_divergence, conflict_aware_fuse, decide, load_decisional_config,
)


def test_masses_sum_to_one():
    m = mass_function(0.7, 0.8)
    assert abs(sum(m.values()) - 1.0) < 1e-9
    assert set(m) == {"F", "not_F", "Θ"}


def test_eq4_values():
    m = mass_function(0.7, 0.8)
    assert m["F"] == pytest.approx(0.56)
    assert m["not_F"] == pytest.approx(0.24)
    assert m["Θ"] == pytest.approx(0.20)


def test_mass_function_clamps_inputs():
    m = mass_function(1.7, 0.8)
    assert m["F"] == pytest.approx(0.8)
    m = mass_function(0.5, -0.5)
    assert m == {"F": 0.0, "not_F": 0.0, "Θ": pytest.approx(1.0)}


def test_dempster_identity_with_vacuous_mass():
    vacuous = {"F": 0.0, "not_F": 0.0, "Θ": 1.0}
    m = mass_function(0.6, 0.8)
    combined, conflict = dempster_combine(m, vacuous)
    assert conflict == pytest.approx(0.0)
    for k in m:
        assert combined[k] == pytest.approx(m[k])


def test_dempster_reinforcing_evidence():
    m_a = mass_function(0.9, 0.9)
    m_b = mass_function(0.9, 0.9)
    combined, conflict = dempster_combine(m_a, m_b)
    assert combined["F"] > m_a["F"]           # agreement sharpens belief in F
    assert abs(sum(combined.values()) - 1.0) < 1e-9


def test_full_conflict_falls_back_to_weighted_average():
    certain_f = {"F": 1.0, "not_F": 0.0, "Θ": 0.0}
    certain_nf = {"F": 0.0, "not_F": 1.0, "Θ": 0.0}
    combined, conflict = dempster_combine(certain_f, certain_nf)
    assert conflict == pytest.approx(1.0)
    assert abs(sum(combined.values()) - 1.0) < 1e-9   # fallback, not division by ~0


def test_weighted_average_combiner():
    a = {"F": 0.6, "not_F": 0.2, "Θ": 0.2}
    b = {"F": 0.2, "not_F": 0.6, "Θ": 0.2}
    c = weighted_average_combine(a, b)
    assert c["F"] == pytest.approx(0.4)
    assert abs(sum(c.values()) - 1.0) < 1e-9


def test_divergence_bounds():
    m = mass_function(0.7, 0.8)
    assert belief_divergence(m, m) == pytest.approx(0.0, abs=1e-8)   # 1e-9 eps in denominator
    certain_f = {"F": 1.0, "not_F": 0.0, "Θ": 0.0}
    certain_nf = {"F": 0.0, "not_F": 1.0, "Θ": 0.0}
    assert belief_divergence(certain_f, certain_nf) == pytest.approx(1.0, abs=1e-8)


def test_conflict_aware_policies():
    masses = [mass_function(0.9, 0.9), mass_function(0.1, 0.9)]  # conflicting
    auto, d_auto = conflict_aware_fuse(masses)                    # diverges -> weighted
    weighted, _ = conflict_aware_fuse(masses, policy="weighted")
    assert auto == weighted
    dempster, _ = conflict_aware_fuse([mass_function(0.9, 0.9), mass_function(0.9, 0.9)],
                                      policy="dempster")
    assert dempster["F"] > 0.9
    with pytest.raises(ValueError):
        conflict_aware_fuse(masses, policy="nonsense")


def test_decide_threshold():
    alarm, info = decide([mass_function(0.95, 0.95), mass_function(0.95, 0.95)],
                         alarm_threshold=0.78)
    assert alarm is True
    assert info["fused_mass"]["F"] > 0.78
    alarm, info = decide([mass_function(0.1, 0.5)], alarm_threshold=0.78)
    assert alarm is False


def test_load_decisional_config_layering(tmp_path):
    cfg = load_decisional_config()          # defaults (no calibration file yet)
    assert set(cfg["reliability"]) == {"c", "d", "s"}
    assert "operational" in cfg
    custom = tmp_path / "cal.yaml"
    custom.write_text("reliability: {c: 0.42, d: 0.9, s: 0.1}\n"
                      "fire_alarm_threshold: 0.65\n")
    cfg2 = load_decisional_config(custom)
    assert cfg2["reliability"]["c"] == 0.42
    assert cfg2["fire_alarm_threshold"] == 0.65
