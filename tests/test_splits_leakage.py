"""§3.5 — the leakage unit test: zero event-ID overlap across splits, plus split-size
sanity. This is the concrete implementation of the paper's flagged risk that
"random frame-level splitting can overestimate generalization."
"""
import pytest

sklearn = pytest.importorskip("sklearn")

from scripts.build_splits import make_group_splits, group_id_for  # noqa: E402


def _synthetic_records(n_groups=40, frames_per_group=5, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    labels = ["fire", "smoke", "none"]
    records = []
    for g in range(n_groups):
        lab = labels[rng.integers(0, 3, size=1)[0]] if g % 10 else "fire"  # keep fire common
        for f in range(frames_per_group):
            records.append({"frame_path": f"event_{g:03d}_frame{f:04d}.jpg",
                            "label": lab, "event_id": f"event_{g:03d}"})
    return records


def test_zero_event_overlap_across_splits():
    records = _synthetic_records()
    assignment = make_group_splits(records, seed=42)
    by_split = {"train": set(), "val": set(), "test": set()}
    for r in records:
        by_split[assignment[r["frame_path"]]].add(r["event_id"])
    assert not (by_split["train"] & by_split["val"]), "LEAKAGE: train/val event overlap"
    assert not (by_split["train"] & by_split["test"]), "LEAKAGE: train/test event overlap"
    assert not (by_split["val"] & by_split["test"]), "LEAKAGE: val/test event overlap"
    for split, ids in by_split.items():
        assert ids, f"{split} got no event groups"


def test_split_proportions_approximate_70_15_15():
    records = _synthetic_records(n_groups=80)
    assignment = make_group_splits(records, seed=7)
    n = len(records)
    counts = {s: sum(1 for v in assignment.values() if v == s) for s in ("train", "val", "test")}
    assert counts["train"] / n == pytest.approx(0.70, abs=0.10)
    assert counts["val"] / n == pytest.approx(0.15, abs=0.08)
    assert counts["test"] / n == pytest.approx(0.15, abs=0.08)


def test_rare_class_survives_all_splits():
    records = _synthetic_records(n_groups=60)
    # add a rare pure-smoke group
    records += [{"frame_path": f"rare_frame{i}.jpg", "label": "smoke", "event_id": "rare"}
                for i in range(5)]
    assignment = make_group_splits(records, seed=3)
    seen = {assignment[r["frame_path"]] for r in records if r["event_id"] == "rare"}
    assert seen, "rare class group lost from splits"


def test_group_id_heuristics():
    import pathlib
    assert group_id_for(pathlib.Path("FLAME_D_video_frame_0042.jpg"), "flame",
                        r"^(.*?)_?[fF]rame") == "FLAME_D_video"
    # dfire: consecutive stems of one source video land in the SAME group, distinct
    # stems never collide across prefixes, unnumbered stills stay their own group
    assert group_id_for(pathlib.Path("AoF00000.jpg"), "dfire", None) == \
        group_id_for(pathlib.Path("AoF00001.jpg"), "dfire", None)
    assert group_id_for(pathlib.Path("AoF00000.jpg"), "dfire", None) != \
        group_id_for(pathlib.Path("AoF00032.jpg"), "dfire", None)
    assert group_id_for(pathlib.Path("WEB00099.jpg"), "dfire", None) != \
        group_id_for(pathlib.Path("AoF00099.jpg"), "dfire", None)
    assert group_id_for(pathlib.Path("unnumbered_still.jpg"), "dfire", None) == \
        "unnumbered_still"
    assert group_id_for(pathlib.Path("seq7/img_1.jpg"), "figlib", None) == "seq7"


def test_dfire_consecutive_frames_share_group():
    """The D-Fire regression test: consecutive stems of one source video must never
    straddle a split boundary (AoF00004 train / AoF00008 val was the leak)."""
    records = [{"frame_path": f"AoF{i:05d}.jpg", "label": "fire",
                "dataset": "dfire", "regex": None} for i in range(640)]
    records += [{"frame_path": f"WEB{i:05d}.jpg", "label": "none",
                 "dataset": "dfire", "regex": None} for i in range(640)]
    assignment = make_group_splits(records, seed=0)
    for lo in range(0, 640, 32):                     # every 32-stem bucket is one group
        splits = {assignment[f"AoF{i:05d}.jpg"] for i in range(lo, lo + 32)}
        assert len(splits) == 1, f"AoF {lo}..{lo + 31} straddles splits: {splits}"
