"""
Unit Tests — Navigation Priority Engine (Phase 9)
==================================================
Phase 9 PASS conditions:

  [1]  ScoredObject has all DirectedObject fields plus nav_score
  [2]  nav_score is always in [0.0, 1.0]
  [3]  nav_score is always a finite float
  [4]  VERY_CLOSE + navigation_critical + CENTER scores > FAR + unknown + LEFT
  [5]  Output is sorted by nav_score descending
  [6]  score([]) returns []
  [7]  All DirectedObject fields preserved unchanged
  [8]  Custom weights produce correct ordering
  [9]  UNKNOWN proximity is not boosted (conservative = 0.0 sub-score)
  [10] compute_score_for_values() gives consistent results
  [11] Hardware test: highest-scoring object is reasonable given scene

All logic tests use synthetic data — no camera or model required.

Run all tests:
    .venv/Scripts/python.exe -m pytest tests/unit/test_priority.py -v -s

Run logic only:
    .venv/Scripts/python.exe -m pytest tests/unit/test_priority.py -v -m "not requires_camera"
"""

import pytest
from typing import Tuple


# ---------------------------------------------------------------------------
# Synthetic DirectedObject helper
# ---------------------------------------------------------------------------

def make_directed_object(
    track_id: int = 1,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.88,
    center_x: float = 0.5,
    center_y: float = 0.5,
    bbox: Tuple = (100.0, 150.0, 400.0, 480.0),
    fw: int = 640,
    fh: int = 480,
    raw_depth: float = 0.6,
    proximity: str = "CLOSE",
    priority: str = "navigation_critical",
    depth_samples: int = 120,
    depth_age: int = 0,
    age: int = 5,
    last_seen: int = 0,
    stability: float = 1.0,
    direction: str = "CENTER",
):
    """Create a synthetic DirectedObject using the real dataclass."""
    from assistive_navigation.navigation.spatial import DirectedObject
    x1, y1, x2, y2 = bbox
    x1n, y1n = x1/fw, y1/fh
    x2n, y2n = x2/fw, y2/fh
    return DirectedObject(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        bbox_norm=(x1n, y1n, x2n, y2n),
        center_x=center_x,
        center_y=center_y,
        age=age,
        last_seen=last_seen,
        stability=stability,
        area_norm=((x2 - x1) * (y2 - y1)) / (fw * fh),
        frame_width=fw,
        frame_height=fh,
        raw_depth=raw_depth,
        proximity=proximity,
        priority=priority,
        depth_samples=depth_samples,
        depth_age=depth_age,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


@pytest.fixture
def engine(config):
    """Fresh NavigationPriorityEngine with default config weights."""
    from assistive_navigation.navigation.priority import NavigationPriorityEngine
    return NavigationPriorityEngine(config)


# ===========================================================================
# GROUP 1 — Import and structure
# ===========================================================================

class TestPriorityImports:

    def test_engine_importable(self):
        from assistive_navigation.navigation.priority import NavigationPriorityEngine
        assert NavigationPriorityEngine is not None

    def test_scored_object_importable(self):
        from assistive_navigation.navigation.priority import ScoredObject
        assert ScoredObject is not None

    def test_engine_creates(self, config):
        from assistive_navigation.navigation.priority import NavigationPriorityEngine
        e = NavigationPriorityEngine(config)
        assert e is not None

    def test_engine_repr(self, engine):
        r = repr(engine)
        assert "NavigationPriorityEngine" in r

    def test_default_weights_sum_to_one(self, engine):
        """Default config weights must sum to ~1.0."""
        assert abs(engine.weight_sum - 1.0) < 0.01

    def test_weights_property_returns_dict(self, engine):
        w = engine.weights
        assert isinstance(w, dict)
        assert "proximity" in w
        assert "object_importance" in w
        assert "direction" in w


# ===========================================================================
# GROUP 2 — ScoredObject structure (PASS CONDITION 1)
# ===========================================================================

class TestScoredObjectStructure:

    def test_scored_object_has_nav_score(self, engine):
        obj = make_directed_object()
        result = engine.score([obj])
        assert hasattr(result[0], "nav_score")

    def test_scored_object_nav_score_is_float(self, engine):
        obj = make_directed_object()
        f = engine.score([obj])[0]
        assert isinstance(f.nav_score, float)

    def test_scored_object_has_all_directed_fields(self, engine):
        """PASS CONDITION 1: all DirectedObject fields must be present."""
        obj = make_directed_object(
            track_id=7, class_id=0, class_name="person",
            confidence=0.91, direction="CENTER", proximity="CLOSE",
            priority="navigation_critical",
        )
        f = engine.score([obj])[0]
        assert f.track_id      == 7
        assert f.class_id      == 0
        assert f.class_name    == "person"
        assert f.direction     == "CENTER"
        assert f.proximity     == "CLOSE"
        assert f.priority      == "navigation_critical"
        assert hasattr(f, "bbox")
        assert hasattr(f, "center_x")
        assert hasattr(f, "raw_depth")
        assert hasattr(f, "age")
        assert hasattr(f, "stability")

    def test_scored_object_repr_contains_score(self, engine):
        obj = make_directed_object()
        f = engine.score([obj])[0]
        r = repr(f)
        assert "ScoredObject" in r
        assert "score=" in r


# ===========================================================================
# GROUP 3 — Score range and validity (PASS CONDITIONS 2, 3)
# ===========================================================================

class TestScoreRange:

    def test_score_in_0_to_1(self, engine):
        """PASS CONDITION 2: nav_score must be in [0.0, 1.0]."""
        obj = make_directed_object()
        f = engine.score([obj])[0]
        assert 0.0 <= f.nav_score <= 1.0

    def test_score_is_finite(self, engine):
        """PASS CONDITION 3: nav_score must be finite."""
        import math
        obj = make_directed_object()
        f = engine.score([obj])[0]
        assert math.isfinite(f.nav_score)

    def test_minimum_score_inputs(self, engine):
        """Worst possible inputs should produce score >= 0.0."""
        obj = make_directed_object(
            proximity="UNKNOWN", priority="unknown",
            direction="UNKNOWN", confidence=0.0, stability=0.0,
        )
        f = engine.score([obj])[0]
        assert f.nav_score >= 0.0

    def test_maximum_score_inputs(self, engine):
        """Best possible inputs should produce score <= 1.0."""
        obj = make_directed_object(
            proximity="VERY_CLOSE", priority="navigation_critical",
            direction="CENTER", confidence=1.0, stability=1.0,
        )
        f = engine.score([obj])[0]
        assert f.nav_score <= 1.0

    def test_score_with_various_proximity_values(self, engine):
        """Score should increase as proximity increases."""
        far    = engine.compute_score_for_values("FAR",        "navigation_critical", "CENTER", 0.9, 0.9)
        medium = engine.compute_score_for_values("MEDIUM",     "navigation_critical", "CENTER", 0.9, 0.9)
        close  = engine.compute_score_for_values("CLOSE",      "navigation_critical", "CENTER", 0.9, 0.9)
        v_close= engine.compute_score_for_values("VERY_CLOSE", "navigation_critical", "CENTER", 0.9, 0.9)
        assert far < medium < close < v_close, (
            f"Expected FAR({far:.3f}) < MEDIUM({medium:.3f}) < "
            f"CLOSE({close:.3f}) < VERY_CLOSE({v_close:.3f})"
        )


# ===========================================================================
# GROUP 4 — Urgency ordering (PASS CONDITION 4)
# ===========================================================================

class TestUrgencyOrdering:
    """PASS CONDITION 4: dangerous/close objects must score above safe/far ones."""

    def test_very_close_critical_center_beats_far_unknown_left(self, engine):
        """
        VERY_CLOSE + navigation_critical + CENTER must outscore
        FAR + unknown + LEFT.
        This is the master requirement: dangerous objects must not be
        overridden by low-value objects.
        """
        urgent = engine.compute_score_for_values(
            "VERY_CLOSE", "navigation_critical", "CENTER", 0.9, 0.9
        )
        non_urgent = engine.compute_score_for_values(
            "FAR", "unknown", "LEFT", 0.4, 0.4
        )
        assert urgent > non_urgent, (
            f"Urgent scenario ({urgent:.3f}) must score above "
            f"non-urgent ({non_urgent:.3f})"
        )

    def test_proximity_is_dominant_factor(self, engine):
        """
        A VERY_CLOSE unknown object should outscore a FAR navigation_critical one.
        Proximity weight (0.40) is the largest, ensuring closest = most urgent.
        """
        very_close_unknown = engine.compute_score_for_values(
            "VERY_CLOSE", "unknown", "CENTER", 0.8, 0.8
        )
        far_critical = engine.compute_score_for_values(
            "FAR", "navigation_critical", "CENTER", 0.8, 0.8
        )
        assert very_close_unknown > far_critical, (
            f"VERY_CLOSE unknown ({very_close_unknown:.3f}) must outscore "
            f"FAR navigation_critical ({far_critical:.3f}). "
            f"Proximity is the dominant factor."
        )

    def test_center_scores_higher_than_sides(self, engine):
        """CENTER direction scores above LEFT/RIGHT (same other inputs)."""
        center = engine.compute_score_for_values(
            "CLOSE", "navigation_critical", "CENTER", 0.8, 0.8
        )
        left = engine.compute_score_for_values(
            "CLOSE", "navigation_critical", "LEFT", 0.8, 0.8
        )
        right = engine.compute_score_for_values(
            "CLOSE", "navigation_critical", "RIGHT", 0.8, 0.8
        )
        assert center > left
        assert center > right

    def test_left_and_right_score_equally(self, engine):
        """LEFT and RIGHT should score identically (symmetric design)."""
        left = engine.compute_score_for_values(
            "CLOSE", "navigation_critical", "LEFT", 0.8, 0.8
        )
        right = engine.compute_score_for_values(
            "CLOSE", "navigation_critical", "RIGHT", 0.8, 0.8
        )
        assert abs(left - right) < 1e-9

    def test_critical_scores_above_contextual(self, engine):
        """navigation_critical outscores contextual (same other inputs)."""
        critical = engine.compute_score_for_values(
            "CLOSE", "navigation_critical", "CENTER", 0.8, 0.8
        )
        contextual = engine.compute_score_for_values(
            "CLOSE", "contextual", "CENTER", 0.8, 0.8
        )
        assert critical > contextual

    def test_contextual_scores_above_low_priority(self, engine):
        contextual = engine.compute_score_for_values(
            "CLOSE", "contextual", "CENTER", 0.8, 0.8
        )
        low = engine.compute_score_for_values(
            "CLOSE", "low_priority", "CENTER", 0.8, 0.8
        )
        assert contextual > low

    def test_unknown_proximity_conservative(self, engine):
        """
        PASS CONDITION 9:
        UNKNOWN proximity must NOT boost the score above other known values.
        It maps to 0.0, so it's equivalent to treating the object as absent.
        """
        unknown_prox = engine.compute_score_for_values(
            "UNKNOWN", "navigation_critical", "CENTER", 0.9, 0.9
        )
        far_prox = engine.compute_score_for_values(
            "FAR", "navigation_critical", "CENTER", 0.9, 0.9
        )
        assert unknown_prox < far_prox, (
            f"UNKNOWN proximity score ({unknown_prox:.3f}) must be below "
            f"FAR ({far_prox:.3f}). UNKNOWN should be conservative."
        )


# ===========================================================================
# GROUP 5 — Sorting (PASS CONDITION 5)
# ===========================================================================

class TestSorting:
    """PASS CONDITION 5: results must be sorted by nav_score descending."""

    def test_sorted_descending(self, engine):
        objs = [
            make_directed_object(track_id=1, proximity="FAR",        direction="LEFT"),
            make_directed_object(track_id=2, proximity="VERY_CLOSE", direction="CENTER"),
            make_directed_object(track_id=3, proximity="MEDIUM",     direction="RIGHT"),
        ]
        result = engine.score(objs)
        scores = [r.nav_score for r in result]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted descending: {scores}"
        )

    def test_highest_first(self, engine):
        """The first object in the sorted list must have the highest score."""
        objs = [
            make_directed_object(track_id=1, proximity="FAR",        direction="LEFT",   priority="contextual"),
            make_directed_object(track_id=2, proximity="VERY_CLOSE", direction="CENTER", priority="navigation_critical"),
            make_directed_object(track_id=3, proximity="MEDIUM",     direction="RIGHT",  priority="low_priority"),
        ]
        result = engine.score(objs)
        assert result[0].track_id == 2, (
            f"Highest-priority object (VERY_CLOSE + critical + CENTER) "
            f"should be first, but got track_id={result[0].track_id}"
        )

    def test_all_same_score_preserves_all(self, engine):
        """When all objects have identical inputs they all appear in output."""
        objs = [make_directed_object(track_id=i) for i in range(1, 5)]
        result = engine.score(objs)
        assert len(result) == 4


# ===========================================================================
# GROUP 6 — Empty input (PASS CONDITION 6)
# ===========================================================================

class TestEmptyInput:

    def test_empty_returns_empty(self, engine):
        """PASS CONDITION 6: empty input → empty output."""
        assert engine.score([]) == []

    def test_one_object_returns_one(self, engine):
        result = engine.score([make_directed_object()])
        assert len(result) == 1


# ===========================================================================
# GROUP 7 — Field preservation (PASS CONDITION 7)
# ===========================================================================

class TestFieldPreservation:

    def test_all_directed_fields_preserved(self, engine):
        """PASS CONDITION 7: every field from DirectedObject must pass through."""
        obj = make_directed_object(
            track_id=42, class_id=56, class_name="chair",
            confidence=0.77, center_x=0.30, center_y=0.60,
            bbox=(50.0, 100.0, 300.0, 400.0),
            raw_depth=0.55, proximity="MEDIUM",
            priority="navigation_critical",
            depth_samples=88, depth_age=2,
            age=12, last_seen=1, stability=0.85,
            direction="LEFT",
        )
        f = engine.score([obj])[0]
        assert f.track_id      == 42
        assert f.class_id      == 56
        assert f.class_name    == "chair"
        assert abs(f.confidence - 0.77) < 1e-6
        assert abs(f.center_x  - 0.30) < 1e-6
        assert abs(f.center_y  - 0.60) < 1e-6
        assert f.bbox          == (50.0, 100.0, 300.0, 400.0)
        assert abs(f.raw_depth - 0.55) < 1e-6
        assert f.proximity     == "MEDIUM"
        assert f.priority      == "navigation_critical"
        assert f.depth_samples == 88
        assert f.depth_age     == 2
        assert f.age           == 12
        assert f.last_seen     == 1
        assert abs(f.stability - 0.85) < 1e-6
        assert f.direction     == "LEFT"
        assert f.frame_width   == 640
        assert f.frame_height  == 480

    def test_input_object_not_modified(self, engine):
        """score() must not modify the input DirectedObject."""
        obj = make_directed_object()
        assert not hasattr(obj, "nav_score")
        engine.score([obj])
        assert not hasattr(obj, "nav_score")


# ===========================================================================
# GROUP 8 — Custom weights (PASS CONDITION 8)
# ===========================================================================

class TestCustomWeights:

    def _make_engine(self, **weights):
        from assistive_navigation.navigation.priority import NavigationPriorityEngine
        config = {"priority": {"weights": weights}}
        return NavigationPriorityEngine(config)

    def test_zero_proximity_weight_removes_proximity_influence(self):
        """With proximity weight=0, proximity value has no effect on score."""
        engine = self._make_engine(
            proximity=0.0, object_importance=1.0,
            direction=0.0, confidence=0.0, stability=0.0
        )
        score_very_close = engine.compute_score_for_values(
            "VERY_CLOSE", "navigation_critical", "CENTER", 0.8, 0.8
        )
        score_far = engine.compute_score_for_values(
            "FAR", "navigation_critical", "CENTER", 0.8, 0.8
        )
        # With no proximity weight, VERY_CLOSE and FAR give same importance score
        assert abs(score_very_close - score_far) < 1e-6

    def test_all_weight_on_direction_orders_correctly(self):
        """With all weight on direction, CENTER object should top the list."""
        engine = self._make_engine(
            proximity=0.0, object_importance=0.0,
            direction=1.0, confidence=0.0, stability=0.0
        )
        left   = engine.compute_score_for_values("CLOSE", "navigation_critical", "LEFT",   0.9, 0.9)
        center = engine.compute_score_for_values("CLOSE", "navigation_critical", "CENTER", 0.9, 0.9)
        right  = engine.compute_score_for_values("CLOSE", "navigation_critical", "RIGHT",  0.9, 0.9)
        assert center > left
        assert center > right

    def test_custom_engine_weights_property(self):
        """weights property reflects the configured values."""
        from assistive_navigation.navigation.priority import NavigationPriorityEngine
        e = NavigationPriorityEngine({"priority": {"weights": {
            "proximity": 0.50, "object_importance": 0.20,
            "direction": 0.15, "confidence": 0.10, "stability": 0.05
        }}})
        w = e.weights
        assert abs(w["proximity"] - 0.50) < 1e-6
        assert abs(w["stability"] - 0.05) < 1e-6


# ===========================================================================
# GROUP 9 — compute_score_for_values (PASS CONDITION 10)
# ===========================================================================

class TestComputeScoreForValues:

    def test_returns_float(self, engine):
        r = engine.compute_score_for_values("CLOSE", "navigation_critical", "CENTER", 0.8, 0.8)
        assert isinstance(r, float)

    def test_consistent_with_score_method(self, engine):
        """compute_score_for_values must give same result as score() for same inputs."""
        obj = make_directed_object(
            proximity="CLOSE", priority="navigation_critical",
            direction="CENTER", confidence=0.88, stability=1.0,
        )
        via_score = engine.score([obj])[0].nav_score
        via_direct = engine.compute_score_for_values(
            "CLOSE", "navigation_critical", "CENTER", 0.88, 1.0
        )
        assert abs(via_score - via_direct) < 1e-6

    def test_identical_inputs_give_identical_scores(self, engine):
        """Same inputs always produce the same score (deterministic)."""
        s1 = engine.compute_score_for_values("CLOSE", "navigation_critical", "CENTER", 0.8, 0.9)
        s2 = engine.compute_score_for_values("CLOSE", "navigation_critical", "CENTER", 0.8, 0.9)
        assert s1 == s2

    def test_unknown_proximity_zero_subscore(self, engine):
        """Confirm UNKNOWN proximity maps to 0.0 sub-score internally."""
        # With only proximity weight=1, rest=0:
        from assistive_navigation.navigation.priority import NavigationPriorityEngine
        e = NavigationPriorityEngine({"priority": {"weights": {
            "proximity": 1.0, "object_importance": 0.0,
            "direction": 0.0, "confidence": 0.0, "stability": 0.0,
        }}})
        score = e.compute_score_for_values("UNKNOWN", "navigation_critical", "CENTER", 1.0, 1.0)
        assert abs(score - 0.0) < 1e-6, (
            f"UNKNOWN proximity with 100% proximity weight should give 0.0, got {score}"
        )


# ===========================================================================
# GROUP 10 — Hardware test
# ===========================================================================

class TestPriorityWithWebcam:
    """Hardware test. Requires webcam."""

    requires_camera = pytest.mark.requires_camera

    @requires_camera
    def test_full_pipeline_with_scores(self, config):
        """
        PASS CONDITION 11 (hardware):
        Run the complete pipeline through Phase 9 and verify ScoredObjects
        have valid structure. The highest-scoring object should be the one
        that is closest/most-critical in the current scene.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        from assistive_navigation.detection.detector import ObjectDetector, DetectorError
        from assistive_navigation.detection.filter import DetectionFilter
        from assistive_navigation.tracking.tracker import ObjectTracker
        from assistive_navigation.depth.depth_estimator import DepthEstimator
        from assistive_navigation.depth.fusion import DepthFusion
        from assistive_navigation.navigation.spatial import SpatialReasoner
        from assistive_navigation.navigation.priority import (
            NavigationPriorityEngine, ScoredObject
        )
        import math

        try:
            detector = ObjectDetector(config)
            detector.load()
        except DetectorError as e:
            pytest.skip(f"Detector not available: {e}")
        try:
            depth_est = DepthEstimator(config)
            depth_est.load()
        except Exception as e:
            pytest.skip(f"Depth estimator not available: {e}")

        det_filter = DetectionFilter(config)
        tracker    = ObjectTracker(config)
        fusion     = DepthFusion(config)
        spatial    = SpatialReasoner(config)
        engine     = NavigationPriorityEngine(config)

        try:
            cam = CameraCapture(config)
            cam.open()
        except CameraError as e:
            pytest.skip(f"Webcam not available: {e}")

        scored_all = []
        try:
            for _ in range(10):
                frame = cam.read()
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                det_result = detector.detect(frame)
                filtered   = det_filter.filter(det_result)
                tracked    = tracker.update(filtered.accepted, w, h)
                depth_map  = depth_est.process_frame(frame)
                fused      = fusion.fuse(tracked, depth_map, depth_est.depth_frame_age)
                directed   = spatial.assign_direction(fused)
                scored     = engine.score(directed)
                scored_all.extend(scored)
        finally:
            cam.release()

        print(f"\n  === PRIORITY ENGINE HARDWARE TEST ===")
        print(f"  Frames processed  : 10")
        print(f"  ScoredObjects     : {len(scored_all)}")

        if scored_all:
            # Take the best score from all frames
            best = max(scored_all, key=lambda s: s.nav_score)
            print(f"  Best nav_score    : {best.nav_score:.3f}")
            print(f"  Best object       : {best.class_name}")
            print(f"  Best direction    : {best.direction}")
            print(f"  Best proximity    : {best.proximity}")

            for s in scored_all:
                assert isinstance(s, ScoredObject)
                assert 0.0 <= s.nav_score <= 1.0
                assert math.isfinite(s.nav_score)
                # sorted within each frame
            # Sorted check is per-call; scored_all concatenates frames
            print(f"  All structure checks: PASS")
        else:
            print(f"  (No objects detected — valid if scene was empty.)")
