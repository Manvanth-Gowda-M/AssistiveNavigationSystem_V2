"""
Unit Tests — Phase 15 False-Positive Reduction
================================================
Phase: 15
Status: IMPLEMENTED

Phase 15 changed exactly ONE runtime parameter:
    temporal.min_area_norm : 0.02  ->  0.15

The value was chosen by an OFFLINE, evidence-based analysis of the existing
Phase 4 evaluation CSV (tools/phase15_fp_analysis.py). These tests validate:

  [A] The area-analysis math is correct (area_norm recomputation).
  [B] The genuine/false-positive classification of Phase 4 rows is correct.
  [C] The calibrated threshold (0.15) hard-fails a phantom with the measured
      FP area profile (never confirms).
  [D] A genuine object with the measured TP area profile still confirms.
  [E] The threshold sweep is reproducible and monotonic in the expected way.
  [F] The recommendation logic never forces a change that loses genuine
      objects (approval condition #5 / #15).
  [G] The live config.yaml carries the calibrated value (0.15) and confidence
      is unchanged (0.35).

ALL tests use synthetic data or the existing CSV — no camera / model required.

IMPORTANT HONESTY NOTE:
    These tests validate the DECISION and the GATE MECHANISM. They do not
    claim the system is free of false positives. The area gate suppresses
    ~71% of the measured empty-scene FP with zero genuine loss; it does NOT
    address class-confusion mislabels (FP-3) or the door/stair COCO gap
    (FP-4). Those remain documented limitations.

Run:
    .venv/Scripts/python.exe -m pytest tests/unit/test_phase15.py -v
"""

import sys
from pathlib import Path

import pytest

# Make src/ and tools/ importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "tools"))

import phase15_fp_analysis as p15  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic ScoredObject factory (mirrors test_temporal.make_scored)
# ---------------------------------------------------------------------------

def make_scored(
    track_id: int = 1,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.90,
    center_x: float = 0.50,
    center_y: float = 0.50,
    area_norm: float = 0.30,
    stability: float = 0.9,
    nav_score: float = 0.75,
    proximity: str = "CLOSE",
    direction: str = "center",
    age: int = 10,
    last_seen: int = 0,
    fw: int = 640,
    fh: int = 480,
):
    """Create a synthetic ScoredObject using the real dataclass."""
    from assistive_navigation.navigation.priority import ScoredObject

    x1, y1 = (center_x - 0.1) * fw, (center_y - 0.1) * fh
    x2, y2 = (center_x + 0.1) * fw, (center_y + 0.1) * fh
    x1n, y1n = x1 / fw, y1 / fh
    x2n, y2n = x2 / fw, y2 / fh
    return ScoredObject(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
        bbox_norm=(x1n, y1n, x2n, y2n),
        center_x=center_x,
        center_y=center_y,
        age=age,
        last_seen=last_seen,
        stability=stability,
        area_norm=area_norm,
        frame_width=fw,
        frame_height=fh,
        raw_depth=0.65,
        proximity=proximity,
        priority="navigation_critical",
        depth_samples=50,
        depth_age=0,
        direction=direction,
        nav_score=nav_score,
    )


def _make_tcf(min_area_norm: float, confirmation_frames: int = 5):
    """TemporalConfirmationFilter with an explicit min_area_norm."""
    from assistive_navigation.navigation.temporal import TemporalConfirmationFilter
    cfg = {
        "temporal": {
            "confirmation_frames":            confirmation_frames,
            "miss_tolerance":                 3,
            "class_stability_threshold":      0.8,
            "class_history_window":           10,
            "position_stability_threshold":   0.15,
            "min_area_norm":                  min_area_norm,
            "min_nav_score_for_confirmation": 0.05,
        }
    }
    return TemporalConfirmationFilter(cfg)


# The calibrated Phase 15 value.
CALIBRATED_MIN_AREA_NORM = 0.15


# ---------------------------------------------------------------------------
# Module-scoped fixtures (avoid class-scoped instance-method deprecation)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def csv_rows():
    csv_path = _PROJECT_ROOT / "data" / "evaluation" / "phase4_results.csv"
    if not csv_path.exists():
        pytest.skip("phase4_results.csv not present")
    return p15.load_accepted_rows(csv_path)


@pytest.fixture(scope="module")
def analysis_result():
    csv_path = _PROJECT_ROOT / "data" / "evaluation" / "phase4_results.csv"
    if not csv_path.exists():
        pytest.skip("phase4_results.csv not present")
    return p15.run_analysis(csv_path)


@pytest.fixture(scope="module")
def live_config():
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


# ===========================================================================
# GROUP A — area_norm recomputation math
# ===========================================================================

class TestAreaMath:

    def test_full_frame_bbox_area_is_one(self):
        assert p15.compute_area_norm(0, 0, 640, 480) == pytest.approx(1.0)

    def test_half_frame_bbox(self):
        # 320x480 = half the frame width
        assert p15.compute_area_norm(0, 0, 320, 480) == pytest.approx(0.5)

    def test_quarter_area_bbox(self):
        # 320x240 = quarter of the frame area
        assert p15.compute_area_norm(0, 0, 320, 240) == pytest.approx(0.25)

    def test_zero_area_degenerate_bbox(self):
        assert p15.compute_area_norm(100, 100, 100, 100) == 0.0

    def test_inverted_bbox_clamped_to_zero(self):
        # x2 < x1 should not yield negative area
        assert p15.compute_area_norm(200, 200, 100, 100) == 0.0

    def test_known_phase4_person_bbox(self):
        # person_standing first row bbox (87.9,181.1,618.1,479.7)
        area = p15.compute_area_norm(87.9, 181.1, 618.1, 479.7)
        # (530.2 * 298.6) / 307200 ≈ 0.5153
        assert area == pytest.approx(0.5153, abs=0.001)


# ===========================================================================
# GROUP B — CSV classification correctness
# ===========================================================================

class TestCsvClassification:

    def test_only_accepted_rows_loaded(self, csv_rows):
        rows = csv_rows
        # load_accepted_rows must only return accepted detections
        assert len(rows) > 0
        # Every row has a recomputed area_norm in [0, 1]
        for r in rows:
            assert 0.0 <= r.area_norm <= 1.0

    def test_fp_scenarios_flagged_as_false_positive(self, csv_rows):
        for r in csv_rows:
            if r.scenario in p15.FP_SCENARIOS:
                assert r.is_false_positive is True
                assert r.is_genuine is False

    def test_genuine_requires_class_match(self, csv_rows):
        for r in csv_rows:
            if r.is_genuine:
                assert r.scenario not in p15.FP_SCENARIOS
                assert r.detected_class == r.ground_truth

    def test_expected_group_counts(self, csv_rows):
        # Documented Phase 15 evidence counts (from the analysis run).
        genuine = [r for r in csv_rows if r.is_genuine]
        fps = [r for r in csv_rows if r.is_false_positive]
        assert len(genuine) == 114, f"genuine count changed: {len(genuine)}"
        assert len(fps) == 303, f"FP count changed: {len(fps)}"


# ===========================================================================
# GROUP C — calibrated threshold suppresses FP-profile phantom
# ===========================================================================

class TestCalibratedThresholdSuppressesFP:

    def test_fp_area_profile_never_confirms_at_0_15(self):
        """
        The dominant empty-scene FP cluster has area_norm around the FP
        median (~0.0645). Such a phantom must HARD-FAIL the area gate at
        the calibrated threshold and never confirm.
        """
        tcf = _make_tcf(min_area_norm=CALIBRATED_MIN_AREA_NORM, confirmation_frames=5)
        fp_phantom = make_scored(track_id=1, class_name="person",
                                 area_norm=0.0645, center_x=0.79)
        last = None
        for _ in range(10):
            last = tcf.update([fp_phantom])[0]
        assert last.is_confirmed is False
        assert "area_too_small" in last.gate_failures
        assert last.confirmation_state == "NEW"

    def test_fp_at_old_threshold_would_have_confirmed(self):
        """
        Regression-of-behaviour: the SAME phantom (area 0.08, the old
        documented FP profile) WOULD confirm at the old 0.02 threshold.
        This documents WHY the change was necessary.
        """
        tcf_old = _make_tcf(min_area_norm=0.02, confirmation_frames=5)
        fp_phantom = make_scored(track_id=1, class_name="person",
                                 area_norm=0.08, center_x=0.79)
        last = None
        for _ in range(6):
            last = tcf_old.update([fp_phantom])[0]
        assert last.is_confirmed is True  # old behaviour — the bug we fixed

    def test_borderline_fp_just_below_threshold_suppressed(self):
        tcf = _make_tcf(min_area_norm=CALIBRATED_MIN_AREA_NORM, confirmation_frames=3)
        obj = make_scored(track_id=1, area_norm=0.149)
        last = None
        for _ in range(6):
            last = tcf.update([obj])[0]
        assert last.is_confirmed is False


# ===========================================================================
# GROUP D — genuine object still confirms
# ===========================================================================

class TestGenuineObjectStillConfirms:

    def test_genuine_min_area_profile_confirms(self):
        """
        The smallest genuine object measured in Phase 4 had area_norm
        ~0.1612 (just above the 0.15 gate). It must still confirm.
        """
        tcf = _make_tcf(min_area_norm=CALIBRATED_MIN_AREA_NORM, confirmation_frames=5)
        genuine = make_scored(track_id=1, class_name="chair", area_norm=0.1612)
        last = None
        for _ in range(5):
            last = tcf.update([genuine])[0]
        assert last.is_confirmed is True
        assert "area_too_small" not in last.gate_failures

    def test_genuine_median_area_profile_confirms(self):
        tcf = _make_tcf(min_area_norm=CALIBRATED_MIN_AREA_NORM, confirmation_frames=5)
        genuine = make_scored(track_id=1, class_name="person", area_norm=0.5156)
        last = None
        for _ in range(5):
            last = tcf.update([genuine])[0]
        assert last.is_confirmed is True

    def test_all_genuine_classes_retained(self):
        """person/chair/laptop/bottle at their genuine profiles all confirm."""
        for cls, area in [("person", 0.52), ("chair", 0.20),
                          ("laptop", 0.45), ("bottle", 0.17)]:
            tcf = _make_tcf(min_area_norm=CALIBRATED_MIN_AREA_NORM,
                            confirmation_frames=4)
            obj = make_scored(track_id=1, class_name=cls, area_norm=area)
            last = None
            for _ in range(4):
                last = tcf.update([obj])[0]
            assert last.is_confirmed is True, f"{cls} at area {area} failed to confirm"


# ===========================================================================
# GROUP E — sweep reproducibility and expected shape
# ===========================================================================

class TestSweepReproducibility:

    def test_sweep_deterministic(self, analysis_result):
        analysis = analysis_result
        csv_path = _PROJECT_ROOT / "data" / "evaluation" / "phase4_results.csv"
        again = p15.run_analysis(csv_path)
        for a, b in zip(analysis["sweep"], again["sweep"]):
            assert a["fp_suppressed"] == b["fp_suppressed"]
            assert a["gen_retained"] == b["gen_retained"]

    def test_fp_suppression_monotonic_nondecreasing(self, analysis_result):
        """As threshold rises, FP suppressed can only increase or stay equal."""
        supp = [s["fp_suppressed"] for s in analysis_result["sweep"]]
        for i in range(1, len(supp)):
            assert supp[i] >= supp[i - 1]

    def test_genuine_retention_monotonic_nonincreasing(self, analysis_result):
        """As threshold rises, genuine retained can only decrease or stay equal."""
        ret = [s["gen_retained"] for s in analysis_result["sweep"]]
        for i in range(1, len(ret)):
            assert ret[i] <= ret[i - 1]

    def test_0_15_zero_genuine_loss(self, analysis_result):
        s015 = next(s for s in analysis_result["sweep"] if abs(s["threshold"] - 0.15) < 1e-9)
        assert s015["gen_lost"] == 0
        assert s015["gen_retention_pct"] == pytest.approx(100.0)

    def test_0_20_has_genuine_loss(self, analysis_result):
        """Documents WHY we stopped at 0.15 and did not go to 0.20."""
        s020 = next(s for s in analysis_result["sweep"] if abs(s["threshold"] - 0.20) < 1e-9)
        assert s020["gen_lost"] > 0

    def test_0_15_suppresses_more_than_baseline(self, analysis_result):
        s002 = next(s for s in analysis_result["sweep"] if abs(s["threshold"] - 0.02) < 1e-9)
        s015 = next(s for s in analysis_result["sweep"] if abs(s["threshold"] - 0.15) < 1e-9)
        assert s015["fp_suppressed"] > s002["fp_suppressed"]


# ===========================================================================
# GROUP F — recommendation never forces genuine loss
# ===========================================================================

class TestRecommendationSafety:

    def test_recommendation_is_change_to_0_15(self):
        csv_path = _PROJECT_ROOT / "data" / "evaluation" / "phase4_results.csv"
        if not csv_path.exists():
            pytest.skip("phase4_results.csv not present")
        rec = p15.run_analysis(csv_path)["recommendation"]
        assert rec["action"] == "CHANGE"
        assert rec["threshold"] == pytest.approx(0.15)
        assert rec["sufficient"] is True

    def test_recommendation_only_picks_zero_loss_candidates(self):
        """
        Synthetic sweep where the only FP-improving option loses genuine
        objects → recommender must refuse (NO_CHANGE, insufficient).
        This guards approval condition #5 / #15.
        """
        fake_sweep = [
            {"threshold": 0.02, "fp_total": 100, "fp_suppressed": 10,
             "fp_retained": 90, "fp_suppression_pct": 10.0,
             "gen_total": 50, "gen_retained": 50, "gen_lost": 0,
             "gen_retention_pct": 100.0},
            {"threshold": 0.20, "fp_total": 100, "fp_suppressed": 80,
             "fp_retained": 20, "fp_suppression_pct": 80.0,
             "gen_total": 50, "gen_retained": 40, "gen_lost": 10,
             "gen_retention_pct": 80.0},
        ]
        rec = p15.recommend(fake_sweep, current_default=0.02)
        assert rec["action"] == "NO_CHANGE"
        assert rec["sufficient"] is False

    def test_recommendation_no_change_when_no_improvement(self):
        """If no zero-loss threshold beats baseline suppression → NO_CHANGE."""
        fake_sweep = [
            {"threshold": 0.02, "fp_total": 100, "fp_suppressed": 30,
             "fp_retained": 70, "fp_suppression_pct": 30.0,
             "gen_total": 50, "gen_retained": 50, "gen_lost": 0,
             "gen_retention_pct": 100.0},
            {"threshold": 0.05, "fp_total": 100, "fp_suppressed": 30,
             "fp_retained": 70, "fp_suppression_pct": 30.0,
             "gen_total": 50, "gen_retained": 50, "gen_lost": 0,
             "gen_retention_pct": 100.0},
        ]
        rec = p15.recommend(fake_sweep, current_default=0.02)
        assert rec["action"] == "NO_CHANGE"
        assert rec["sufficient"] is False


# ===========================================================================
# GROUP G — live config carries the calibrated value
# ===========================================================================

class TestLiveConfig:

    def test_config_min_area_norm_is_0_15(self, live_config):
        assert live_config["temporal"]["min_area_norm"] == pytest.approx(0.15)

    def test_confidence_threshold_unchanged(self, live_config):
        # Phase 15 must NOT touch confidence — stays 0.35.
        assert live_config["detection"]["confidence_threshold"] == pytest.approx(0.35)

    def test_confirmation_frames_unchanged(self, live_config):
        assert live_config["temporal"]["confirmation_frames"] == 8

    def test_filter_still_builds_and_confidence_intact(self, live_config):
        """DetectionFilter must still construct with confidence 0.35."""
        from assistive_navigation.detection.filter import DetectionFilter
        f = DetectionFilter(live_config)
        assert f._conf_threshold == pytest.approx(0.35)
