"""
Phase 15 — CSV Replay / Before-After Reproducibility
=====================================================
Phase: 15
Status: IMPLEMENTED

PURPOSE
-------
Replay the EXISTING Phase 4 evaluation CSV through the Phase 15 area gate
and assert that the documented before/after false-positive suppression and
genuine-object retention numbers are reproducible from the raw data.

This is the auditable link between:
    - the raw Phase 4 measurements (data/evaluation/phase4_results.csv), and
    - the config change (temporal.min_area_norm: 0.02 -> 0.15).

It is READ-ONLY on phase4_results.csv and phase4_report.txt.
It does NOT run the camera, detector, tracker, depth, or TTS.
It does NOT modify any config.

DEFINITIONS (must match tools/phase15_fp_analysis.py):
    RETAINED  = area_norm >= threshold  (would pass the area gate)
    SUPPRESSED / LOST = area_norm < threshold

Run:
    .venv/Scripts/python.exe -m pytest tests/evaluation/phase15_fp_replay.py -v -s
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "tools"))

import phase15_fp_analysis as p15  # noqa: E402

_CSV = _PROJECT_ROOT / "data" / "evaluation" / "phase4_results.csv"

# Old (pre-Phase-15) and new (Phase-15 calibrated) thresholds.
OLD_THRESHOLD = 0.02
NEW_THRESHOLD = 0.15


@pytest.fixture(scope="module")
def rows():
    if not _CSV.exists():
        pytest.skip("phase4_results.csv not present")
    return p15.load_accepted_rows(_CSV)


def _counts_at(rows, threshold):
    genuine = [r for r in rows if r.is_genuine]
    fps = [r for r in rows if r.is_false_positive]
    fp_suppressed = sum(1 for r in fps if r.area_norm < threshold)
    gen_retained = sum(1 for r in genuine if r.area_norm >= threshold)
    return {
        "fp_total": len(fps),
        "fp_suppressed": fp_suppressed,
        "fp_retained": len(fps) - fp_suppressed,
        "gen_total": len(genuine),
        "gen_retained": gen_retained,
        "gen_lost": len(genuine) - gen_retained,
    }


class TestBeforeAfterReplay:
    """Reproduce the exact before/after numbers reported for Phase 15."""

    def test_before_baseline_0_02(self, rows):
        c = _counts_at(rows, OLD_THRESHOLD)
        # Before: 0.02 suppressed 72/303 FP, retained all 114 genuine.
        assert c["fp_total"] == 303
        assert c["fp_suppressed"] == 72
        assert c["gen_total"] == 114
        assert c["gen_retained"] == 114
        assert c["gen_lost"] == 0

    def test_after_calibrated_0_15(self, rows):
        c = _counts_at(rows, NEW_THRESHOLD)
        # After: 0.15 suppressed 215/303 FP (71.0%), retained all 114 genuine.
        assert c["fp_total"] == 303
        assert c["fp_suppressed"] == 215
        assert c["gen_total"] == 114
        assert c["gen_retained"] == 114
        assert c["gen_lost"] == 0

    def test_fp_suppression_improvement(self, rows):
        before = _counts_at(rows, OLD_THRESHOLD)["fp_suppressed"]
        after = _counts_at(rows, NEW_THRESHOLD)["fp_suppressed"]
        gained = after - before
        # 215 - 72 = 143 additional FP suppressed, at zero genuine cost.
        assert gained == 143

    def test_zero_genuine_loss_at_new_threshold(self, rows):
        assert _counts_at(rows, NEW_THRESHOLD)["gen_lost"] == 0


class TestPerScenarioReplay:
    """Per-scenario before/after, focusing on the dominant FP scenario."""

    def test_empty_scene_dominant_fp_reduced(self, rows):
        scn = p15.per_scenario_breakdown(rows, NEW_THRESHOLD)
        empty = scn["empty_scene_fp_rate"]
        # 203 accepted empty-scene FP -> 22 retained, 181 suppressed.
        assert empty["total"] == 203
        assert empty["is_fp"] is True
        assert empty["suppressed"] == 181
        assert empty["retained"] == 22

    def test_genuine_scenarios_fully_retained(self, rows):
        scn = p15.per_scenario_breakdown(rows, NEW_THRESHOLD)
        for name in ("person_standing", "chair_visible",
                     "laptop_visible", "bottle_visible"):
            b = scn[name]
            assert b["is_genuine"] is True
            assert b["suppressed"] == 0, f"{name} lost genuine detections"
            assert b["retained"] == b["total"]

    def test_per_class_retention_full(self, rows):
        pcls = p15.per_class_genuine_retention(rows, NEW_THRESHOLD)
        for cls in ("person", "chair", "laptop", "bottle"):
            assert pcls[cls]["lost"] == 0
            assert pcls[cls]["retained"] == pcls[cls]["total"]


class TestHonestLimitations:
    """Assert the documented limitations remain true (not tuned away)."""

    def test_some_fp_remain_by_design(self, rows):
        """
        ~29% of empty-scene-family FP overlap the genuine area cluster and
        are NOT suppressed. This is expected and honest, not a failure.
        """
        c = _counts_at(rows, NEW_THRESHOLD)
        assert c["fp_retained"] > 0

    def test_mislabels_not_counted_as_solved(self, rows):
        """
        FP-3 mislabels (wrong class in TP scenes) are excluded from the FP
        group — the area gate does not claim to solve them.
        """
        mislabel = [r for r in rows
                    if not (r.is_genuine or r.is_false_positive)]
        assert len(mislabel) > 0
