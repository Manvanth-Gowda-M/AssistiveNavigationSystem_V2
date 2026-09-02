"""
Unit Tests — Spatial Reasoning Module (Phase 8)
================================================
Phase 8 PASS conditions:

  [1]  DirectedObject has all FusedObject fields plus 'direction'
  [2]  center_x < left_zone_end        → LEFT
  [3]  left_zone_end <= center_x < right_zone_start → CENTER
  [4]  center_x >= right_zone_start    → RIGHT
  [5]  Boundary: center_x = left_zone_end  → CENTER (not LEFT)
       Boundary: center_x = right_zone_start → RIGHT (not CENTER)
  [6]  All FusedObject fields preserved unchanged in DirectedObject
  [7]  Invalid center_x (NaN, inf) → UNKNOWN, no crash
  [8]  Custom thresholds produce correct zones
  [9]  Misconfigured thresholds (left >= right) raise ValueError
  [10] assign_direction([]) returns []
  [11] Hardware test: direction labels correct for objects in known positions

All logic tests run without camera or model.

Run all tests:
    .venv/Scripts/python.exe -m pytest tests/unit/test_spatial.py -v -s

Run logic-only tests:
    .venv/Scripts/python.exe -m pytest tests/unit/test_spatial.py -v -m "not requires_camera"
"""

import math
import pytest
from typing import Tuple


# ---------------------------------------------------------------------------
# Synthetic FusedObject helper
# ---------------------------------------------------------------------------

def make_fused_object(
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
):
    """Create a synthetic FusedObject using the real dataclass."""
    from assistive_navigation.depth.fusion import FusedObject
    x1, y1, x2, y2 = bbox
    x1n, y1n = x1/fw, y1/fh
    x2n, y2n = x2/fw, y2/fh
    return FusedObject(
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
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


@pytest.fixture
def spatial(config):
    """Fresh SpatialReasoner with default config thresholds."""
    from assistive_navigation.navigation.spatial import SpatialReasoner
    return SpatialReasoner(config)


# ===========================================================================
# GROUP 1 — Import and structure
# ===========================================================================

class TestSpatialImports:

    def test_spatial_reasoner_importable(self):
        from assistive_navigation.navigation.spatial import SpatialReasoner
        assert SpatialReasoner is not None

    def test_directed_object_importable(self):
        from assistive_navigation.navigation.spatial import DirectedObject
        assert DirectedObject is not None

    def test_direction_constants_importable(self):
        from assistive_navigation.navigation.spatial import Direction
        assert Direction.LEFT    == "LEFT"
        assert Direction.CENTER  == "CENTER"
        assert Direction.RIGHT   == "RIGHT"
        assert Direction.UNKNOWN == "UNKNOWN"

    def test_spatial_reasoner_creates(self, config):
        from assistive_navigation.navigation.spatial import SpatialReasoner
        r = SpatialReasoner(config)
        assert r is not None

    def test_repr_contains_zone_info(self, spatial):
        r = repr(spatial)
        assert "SpatialReasoner" in r
        assert "LEFT" in r
        assert "CENTER" in r
        assert "RIGHT" in r


# ===========================================================================
# GROUP 2 — Zone classification (classify_direction)
# ===========================================================================

class TestClassifyDirection:
    """
    PASS CONDITIONS 2, 3, 4:
    Test all three zones using classify_direction() directly.
    Default thresholds: left_end=0.35, right_start=0.65.
    """

    def test_far_left_is_left(self, spatial):
        """center_x = 0.0 → LEFT."""
        from assistive_navigation.navigation.spatial import Direction
        assert spatial.classify_direction(0.0) == Direction.LEFT

    def test_mid_left_is_left(self, spatial):
        """center_x = 0.20 → LEFT (below 0.35)."""
        from assistive_navigation.navigation.spatial import Direction
        assert spatial.classify_direction(0.20) == Direction.LEFT

    def test_just_below_left_end_is_left(self, spatial):
        """center_x = 0.349 → LEFT."""
        from assistive_navigation.navigation.spatial import Direction
        assert spatial.classify_direction(0.349) == Direction.LEFT

    def test_centre_is_center(self, spatial):
        """center_x = 0.5 → CENTER."""
        from assistive_navigation.navigation.spatial import Direction
        assert spatial.classify_direction(0.5) == Direction.CENTER

    def test_mid_right_zone_center(self, spatial):
        """center_x = 0.55 → CENTER (above 0.35, below 0.65)."""
        from assistive_navigation.navigation.spatial import Direction
        assert spatial.classify_direction(0.55) == Direction.CENTER

    def test_just_below_right_start_is_center(self, spatial):
        """center_x = 0.649 → CENTER."""
        from assistive_navigation.navigation.spatial import Direction
        assert spatial.classify_direction(0.649) == Direction.CENTER

    def test_mid_right_is_right(self, spatial):
        """center_x = 0.80 → RIGHT."""
        from assistive_navigation.navigation.spatial import Direction
        assert spatial.classify_direction(0.80) == Direction.RIGHT

    def test_far_right_is_right(self, spatial):
        """center_x = 1.0 → RIGHT."""
        from assistive_navigation.navigation.spatial import Direction
        assert spatial.classify_direction(1.0) == Direction.RIGHT

    def test_all_three_directions_reachable(self, spatial):
        """All three direction values must be reachable."""
        from assistive_navigation.navigation.spatial import Direction
        results = {
            spatial.classify_direction(0.10),   # LEFT
            spatial.classify_direction(0.50),   # CENTER
            spatial.classify_direction(0.80),   # RIGHT
        }
        assert Direction.LEFT   in results
        assert Direction.CENTER in results
        assert Direction.RIGHT  in results


# ===========================================================================
# GROUP 3 — Boundary values (PASS CONDITION 5)
# ===========================================================================

class TestBoundaryValues:
    """
    PASS CONDITION 5:
    Exact boundary values must map deterministically to the correct zone.
    Convention: [left_end, right_start) for CENTER.
    At left_end: CENTER. At right_start: RIGHT.
    """

    def test_exactly_left_zone_end_is_center(self, spatial):
        """
        PASS CONDITION 5a:
        center_x = left_zone_end (default 0.35) → CENTER, not LEFT.
        LEFT zone is [0.0, 0.35) — strictly below 0.35.
        """
        from assistive_navigation.navigation.spatial import Direction
        left_end = spatial.left_zone_end
        result = spatial.classify_direction(left_end)
        assert result == Direction.CENTER, (
            f"At exactly left_zone_end={left_end}, expected CENTER but got {result}. "
            f"LEFT zone must be strictly [0.0, {left_end}), not including the boundary."
        )

    def test_exactly_right_zone_start_is_right(self, spatial):
        """
        PASS CONDITION 5b:
        center_x = right_zone_start (default 0.65) → RIGHT, not CENTER.
        CENTER zone is [0.35, 0.65) — strictly below 0.65.
        """
        from assistive_navigation.navigation.spatial import Direction
        right_start = spatial.right_zone_start
        result = spatial.classify_direction(right_start)
        assert result == Direction.RIGHT, (
            f"At exactly right_zone_start={right_start}, expected RIGHT but got {result}. "
            f"CENTER zone must be [left_end, {right_start}), not including the boundary."
        )

    def test_just_above_left_end_is_center(self, spatial):
        """center_x = left_zone_end + epsilon → CENTER."""
        from assistive_navigation.navigation.spatial import Direction
        cx = spatial.left_zone_end + 1e-9
        assert spatial.classify_direction(cx) == Direction.CENTER

    def test_just_below_right_start_is_center(self, spatial):
        """center_x = right_zone_start - epsilon → CENTER."""
        from assistive_navigation.navigation.spatial import Direction
        cx = spatial.right_zone_start - 1e-9
        assert spatial.classify_direction(cx) == Direction.CENTER

    def test_just_above_right_start_is_right(self, spatial):
        """center_x = right_zone_start + epsilon → RIGHT."""
        from assistive_navigation.navigation.spatial import Direction
        cx = spatial.right_zone_start + 1e-9
        assert spatial.classify_direction(cx) == Direction.RIGHT


# ===========================================================================
# GROUP 4 — Invalid center_x handling (PASS CONDITION 7)
# ===========================================================================

class TestInvalidCenterX:
    """PASS CONDITION 7: invalid values handled without crash."""

    def test_nan_returns_unknown(self, spatial):
        """NaN center_x → UNKNOWN."""
        from assistive_navigation.navigation.spatial import Direction
        result = spatial.classify_direction(float("nan"))
        assert result == Direction.UNKNOWN

    def test_positive_inf_returns_unknown(self, spatial):
        """Positive infinity → UNKNOWN."""
        from assistive_navigation.navigation.spatial import Direction
        result = spatial.classify_direction(float("inf"))
        assert result == Direction.UNKNOWN

    def test_negative_inf_returns_unknown(self, spatial):
        """Negative infinity → UNKNOWN."""
        from assistive_navigation.navigation.spatial import Direction
        result = spatial.classify_direction(float("-inf"))
        assert result == Direction.UNKNOWN

    def test_out_of_range_low_clamps_to_left(self, spatial):
        """center_x < 0.0 is clamped to 0.0 → LEFT."""
        from assistive_navigation.navigation.spatial import Direction
        result = spatial.classify_direction(-0.5)
        assert result == Direction.LEFT

    def test_out_of_range_high_clamps_to_right(self, spatial):
        """center_x > 1.0 is clamped to 1.0 → RIGHT."""
        from assistive_navigation.navigation.spatial import Direction
        result = spatial.classify_direction(1.5)
        assert result == Direction.RIGHT


# ===========================================================================
# GROUP 5 — Configurable thresholds (PASS CONDITION 8)
# ===========================================================================

class TestCustomThresholds:
    """PASS CONDITION 8: custom thresholds produce correct zones."""

    def _make_spatial(self, left_end: float, right_start: float):
        """Create SpatialReasoner with custom thresholds."""
        from assistive_navigation.navigation.spatial import SpatialReasoner
        config = {
            "spatial": {
                "left_zone_end": left_end,
                "right_zone_start": right_start,
            }
        }
        return SpatialReasoner(config)

    def test_narrow_center_zone(self):
        """Very narrow center zone [0.49, 0.51) still works."""
        from assistive_navigation.navigation.spatial import Direction
        spatial = self._make_spatial(0.49, 0.51)
        assert spatial.classify_direction(0.30) == Direction.LEFT
        assert spatial.classify_direction(0.50) == Direction.CENTER
        assert spatial.classify_direction(0.70) == Direction.RIGHT
        # Boundary
        assert spatial.classify_direction(0.49) == Direction.CENTER
        assert spatial.classify_direction(0.51) == Direction.RIGHT

    def test_wide_center_zone(self):
        """Wide center zone [0.1, 0.9) still works."""
        from assistive_navigation.navigation.spatial import Direction
        spatial = self._make_spatial(0.10, 0.90)
        assert spatial.classify_direction(0.05) == Direction.LEFT
        assert spatial.classify_direction(0.50) == Direction.CENTER
        assert spatial.classify_direction(0.95) == Direction.RIGHT

    def test_custom_thresholds_properties(self):
        """Properties return the configured values."""
        spatial = self._make_spatial(0.25, 0.75)
        assert spatial.left_zone_end    == 0.25
        assert spatial.right_zone_start == 0.75
        assert abs(spatial.center_zone_width - 0.50) < 1e-9

    def test_default_thresholds_from_config(self, config):
        """Default config thresholds are 0.35 and 0.65."""
        from assistive_navigation.navigation.spatial import SpatialReasoner
        s = SpatialReasoner(config)
        assert s.left_zone_end    == 0.35
        assert s.right_zone_start == 0.65

    def test_center_zone_width_property(self, spatial):
        """center_zone_width = right_start - left_end."""
        expected = spatial.right_zone_start - spatial.left_zone_end
        assert abs(spatial.center_zone_width - expected) < 1e-9


# ===========================================================================
# GROUP 6 — Misconfigured threshold validation (PASS CONDITION 9)
# ===========================================================================

class TestThresholdValidation:
    """PASS CONDITION 9: misconfigured thresholds raise ValueError."""

    def _make_spatial(self, left_end, right_start):
        from assistive_navigation.navigation.spatial import SpatialReasoner
        return SpatialReasoner({"spatial": {
            "left_zone_end": left_end,
            "right_zone_start": right_start,
        }})

    def test_left_equal_right_raises(self):
        """left_zone_end == right_zone_start → ValueError."""
        with pytest.raises(ValueError, match="strictly less than"):
            self._make_spatial(0.5, 0.5)

    def test_left_greater_than_right_raises(self):
        """left_zone_end > right_zone_start → ValueError."""
        with pytest.raises(ValueError, match="strictly less than"):
            self._make_spatial(0.7, 0.3)

    def test_left_at_zero_raises(self):
        """left_zone_end = 0.0 is invalid."""
        with pytest.raises(ValueError):
            self._make_spatial(0.0, 0.65)

    def test_right_at_one_raises(self):
        """right_zone_start = 1.0 is invalid."""
        with pytest.raises(ValueError):
            self._make_spatial(0.35, 1.0)

    def test_negative_left_raises(self):
        """Negative left threshold is invalid."""
        with pytest.raises(ValueError):
            self._make_spatial(-0.1, 0.65)


# ===========================================================================
# GROUP 7 — assign_direction() method
# ===========================================================================

class TestAssignDirection:
    """Tests for the main public method."""

    def test_empty_list_returns_empty(self, spatial):
        """PASS CONDITION 10: empty input → empty output."""
        result = spatial.assign_direction([])
        assert result == []

    def test_one_object_returns_one_directed(self, spatial):
        obj = make_fused_object(center_x=0.5)
        result = spatial.assign_direction([obj])
        assert len(result) == 1

    def test_three_objects_three_directed(self, spatial):
        objs = [
            make_fused_object(track_id=1, center_x=0.10),
            make_fused_object(track_id=2, center_x=0.50),
            make_fused_object(track_id=3, center_x=0.90),
        ]
        result = spatial.assign_direction(objs)
        assert len(result) == 3

    def test_directions_assigned_correctly_for_three(self, spatial):
        """PASS CONDITIONS 2, 3, 4 via assign_direction."""
        from assistive_navigation.navigation.spatial import Direction
        objs = [
            make_fused_object(track_id=1, center_x=0.10),   # LEFT
            make_fused_object(track_id=2, center_x=0.50),   # CENTER
            make_fused_object(track_id=3, center_x=0.90),   # RIGHT
        ]
        result = spatial.assign_direction(objs)
        by_id = {r.track_id: r.direction for r in result}
        assert by_id[1] == Direction.LEFT
        assert by_id[2] == Direction.CENTER
        assert by_id[3] == Direction.RIGHT

    def test_output_is_directed_object_instances(self, spatial):
        """Results must be DirectedObject instances."""
        from assistive_navigation.navigation.spatial import DirectedObject
        obj = make_fused_object()
        result = spatial.assign_direction([obj])
        assert isinstance(result[0], DirectedObject)


# ===========================================================================
# GROUP 8 — FusedObject field preservation (PASS CONDITION 6)
# ===========================================================================

class TestFieldPreservation:
    """PASS CONDITION 6: all FusedObject fields must be preserved."""

    def test_all_fused_fields_preserved(self, spatial):
        obj = make_fused_object(
            track_id=42, class_id=0, class_name="person",
            confidence=0.91, center_x=0.50, center_y=0.45,
            bbox=(100.0, 150.0, 400.0, 450.0),
            fw=640, fh=480,
            raw_depth=0.72, proximity="CLOSE",
            priority="navigation_critical",
            depth_samples=200, depth_age=3,
            age=15, last_seen=0, stability=0.93,
        )
        result = spatial.assign_direction([obj])[0]

        assert result.track_id       == 42
        assert result.class_id       == 0
        assert result.class_name     == "person"
        assert abs(result.confidence - 0.91) < 1e-6
        assert result.center_x       == pytest.approx(0.50)
        assert result.center_y       == pytest.approx(0.45)
        assert result.bbox           == pytest.approx((100.0, 150.0, 400.0, 450.0))
        assert abs(result.raw_depth  - 0.72) < 1e-6
        assert result.proximity      == "CLOSE"
        assert result.priority       == "navigation_critical"
        assert result.depth_samples  == 200
        assert result.depth_age      == 3
        assert result.age            == 15
        assert result.last_seen      == 0
        assert abs(result.stability  - 0.93) < 1e-6
        assert result.frame_width    == 640
        assert result.frame_height   == 480

    def test_direction_field_is_string(self, spatial):
        obj = make_fused_object()
        result = spatial.assign_direction([obj])[0]
        assert isinstance(result.direction, str)
        assert len(result.direction) > 0

    def test_directed_object_has_direction_attribute(self, spatial):
        obj = make_fused_object()
        result = spatial.assign_direction([obj])[0]
        assert hasattr(result, "direction")

    def test_repr_contains_direction(self, spatial):
        obj = make_fused_object(center_x=0.50)
        r = repr(spatial.assign_direction([obj])[0])
        assert "DirectedObject" in r
        assert "CENTER" in r

    def test_input_fused_object_not_modified(self, spatial):
        """assign_direction must not modify the original FusedObject."""
        obj = make_fused_object(center_x=0.20)
        assert not hasattr(obj, "direction"), "FusedObject should not have direction"
        spatial.assign_direction([obj])
        # FusedObject should still lack a direction attribute
        assert not hasattr(obj, "direction")


# ===========================================================================
# GROUP 9 — Hardware test
# ===========================================================================

class TestSpatialWithWebcam:
    """Hardware test. Requires webcam."""

    requires_camera = pytest.mark.requires_camera

    @requires_camera
    def test_full_pipeline_with_direction(self, config):
        """
        PASS CONDITION 11 (hardware):
        Run the complete pipeline through Phase 8 on real camera frames.
        Verify DirectedObjects have valid structure and correct directions.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        from assistive_navigation.detection.detector import ObjectDetector, DetectorError
        from assistive_navigation.detection.filter import DetectionFilter
        from assistive_navigation.tracking.tracker import ObjectTracker
        from assistive_navigation.depth.depth_estimator import DepthEstimator
        from assistive_navigation.depth.fusion import DepthFusion
        from assistive_navigation.navigation.spatial import (
            SpatialReasoner, DirectedObject, Direction
        )

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

        try:
            cam = CameraCapture(config)
            cam.open()
        except CameraError as e:
            pytest.skip(f"Webcam not available: {e}")

        directed_all = []
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
                directed_all.extend(directed)
        finally:
            cam.release()

        print(f"\n  === SPATIAL REASONING HARDWARE TEST ===")
        print(f"  Frames processed   : 10")
        print(f"  DirectedObjects    : {len(directed_all)}")

        if directed_all:
            classes    = {d.class_name for d in directed_all}
            directions = {d.direction  for d in directed_all}
            print(f"  Classes detected   : {classes}")
            print(f"  Directions seen    : {directions}")
            print(f"  Zone thresholds    : LEFT<{spatial.left_zone_end}  "
                  f"CENTER<{spatial.right_zone_start}  RIGHT>=")

            for d in directed_all:
                assert isinstance(d, DirectedObject)
                assert d.direction in {
                    Direction.LEFT, Direction.CENTER,
                    Direction.RIGHT, Direction.UNKNOWN
                }, f"Unexpected direction: {d.direction}"
                # center_x and direction must be consistent
                if math.isfinite(d.center_x):
                    if d.center_x < spatial.left_zone_end:
                        assert d.direction == Direction.LEFT
                    elif d.center_x < spatial.right_zone_start:
                        assert d.direction == Direction.CENTER
                    else:
                        assert d.direction == Direction.RIGHT
        else:
            print(f"  (No objects detected — valid if scene was empty)")
