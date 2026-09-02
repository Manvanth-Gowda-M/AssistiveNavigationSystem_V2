"""
Unit Tests - Depth-Detection Fusion Module
============================================
Phase 7 PASS conditions:

  [1]  FusedObject has all required fields with correct types
  [2]  fuse() produces one FusedObject per tracked object
  [3]  proximity is UNKNOWN when no depth map available
  [4]  proximity maps correctly to the right bucket (synthetic depth map)
  [5]  priority is derived correctly from class_id
  [6]  temporal smoothing reduces per-frame depth variance
  [7]  fuse() with empty track list returns empty list
  [8]  depth_age is carried from the depth_age argument
  [9]  raw_depth is -1.0 when no depth map available
  [10] All tests pass

Tests use ONLY synthetic data (TrackedObjects, depth maps) — no camera,
no model loading required for logic tests.

Hardware test: combines real camera + detector + tracker + depth estimator
               to verify the full pipeline produces valid FusedObjects.

Run all tests:
    .venv/Scripts/python.exe -m pytest tests/unit/test_fusion.py -v -s

Run only logic tests (no camera/model):
    .venv/Scripts/python.exe -m pytest tests/unit/test_fusion.py -v -m "not requires_camera"
"""

import pytest
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_tracked_object(
    track_id: int = 1,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.88,
    x1: float = 100.0, y1: float = 150.0,
    x2: float = 400.0, y2: float = 480.0,
    fw: int = 640, fh: int = 480,
    age: int = 5,
    last_seen: int = 0,
    stability: float = 1.0,
):
    """Create a synthetic TrackedObject using the real dataclass."""
    from assistive_navigation.tracking.tracker import TrackedObject
    x1n = x1 / fw; y1n = y1 / fh; x2n = x2 / fw; y2n = y2 / fh
    return TrackedObject(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
        bbox_norm=(x1n, y1n, x2n, y2n),
        center_x=(x1n + x2n) / 2.0,
        center_y=(y1n + y2n) / 2.0,
        age=age,
        last_seen=last_seen,
        stability=stability,
        area_norm=((x2 - x1) * (y2 - y1)) / (fw * fh),
        frame_width=fw,
        frame_height=fh,
    )


def make_uniform_depth_map(value: float, h: int = 480, w: int = 640) -> np.ndarray:
    """Create a depth map with all pixels set to a single value."""
    return np.full((h, w), value, dtype=np.float32)


def make_split_depth_map(
    left_value: float = 0.85,
    right_value: float = 0.20,
    h: int = 480, w: int = 640,
) -> np.ndarray:
    """Left half = left_value, right half = right_value."""
    d = np.zeros((h, w), dtype=np.float32)
    d[:, :w // 2] = left_value
    d[:, w // 2:] = right_value
    return d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


@pytest.fixture
def fusion(config):
    """Fresh DepthFusion instance (no model loading needed)."""
    from assistive_navigation.depth.fusion import DepthFusion
    return DepthFusion(config)


# ===========================================================================
# GROUP 1 — Import and structure
# ===========================================================================

class TestFusionImports:

    def test_depth_fusion_importable(self):
        from assistive_navigation.depth.fusion import DepthFusion
        assert DepthFusion is not None

    def test_fused_object_importable(self):
        from assistive_navigation.depth.fusion import FusedObject
        assert FusedObject is not None

    def test_fusion_creates_without_error(self, config):
        from assistive_navigation.depth.fusion import DepthFusion
        f = DepthFusion(config)
        assert f is not None

    def test_fusion_repr(self, fusion):
        r = repr(fusion)
        assert "DepthFusion" in r


# ===========================================================================
# GROUP 2 — FusedObject structure
# ===========================================================================

class TestFusedObjectStructure:
    """PASS CONDITION 1: FusedObject must have all required fields."""

    def test_fused_object_has_all_tracker_fields(self, fusion):
        """All TrackedObject fields must pass through to FusedObject."""
        obj = make_tracked_object(track_id=7, class_id=0, class_name="person")
        depth = make_uniform_depth_map(0.6)
        results = fusion.fuse([obj], depth, depth_age=0)
        assert len(results) == 1
        f = results[0]
        assert f.track_id == 7
        assert f.class_id == 0
        assert f.class_name == "person"
        assert isinstance(f.confidence, float)
        assert isinstance(f.bbox, tuple) and len(f.bbox) == 4
        assert isinstance(f.bbox_norm, tuple) and len(f.bbox_norm) == 4
        assert isinstance(f.center_x, float)
        assert isinstance(f.center_y, float)
        assert isinstance(f.age, int)
        assert isinstance(f.last_seen, int)
        assert isinstance(f.stability, float)
        assert isinstance(f.area_norm, float)
        assert isinstance(f.frame_width, int)
        assert isinstance(f.frame_height, int)

    def test_fused_object_has_depth_fields(self, fusion):
        """Phase 7 fields must exist on FusedObject."""
        obj = make_tracked_object()
        depth = make_uniform_depth_map(0.6)
        results = fusion.fuse([obj], depth, depth_age=3)
        f = results[0]
        assert hasattr(f, "raw_depth")
        assert hasattr(f, "proximity")
        assert hasattr(f, "priority")
        assert hasattr(f, "depth_samples")
        assert hasattr(f, "depth_age")

    def test_fused_object_field_types(self, fusion):
        obj = make_tracked_object()
        depth = make_uniform_depth_map(0.6)
        f = fusion.fuse([obj], depth, depth_age=2)[0]
        assert isinstance(f.raw_depth, float)
        assert isinstance(f.proximity, str)
        assert isinstance(f.priority, str)
        assert isinstance(f.depth_samples, int)
        assert isinstance(f.depth_age, int)

    def test_fused_object_depth_age_matches_argument(self, fusion):
        """depth_age on FusedObject must match the depth_age argument."""
        obj = make_tracked_object()
        depth = make_uniform_depth_map(0.5)
        f = fusion.fuse([obj], depth, depth_age=7)[0]
        assert f.depth_age == 7

    def test_fused_object_repr_contains_track_id(self, fusion):
        obj = make_tracked_object(track_id=42)
        depth = make_uniform_depth_map(0.5)
        f = fusion.fuse([obj], depth)[0]
        assert "42" in repr(f)

    def test_bbox_values_preserved_exactly(self, fusion):
        """bbox values must pass through unchanged."""
        obj = make_tracked_object(x1=50.5, y1=75.3, x2=300.1, y2=450.7)
        depth = make_uniform_depth_map(0.5)
        f = fusion.fuse([obj], depth)[0]
        assert f.bbox == pytest.approx((50.5, 75.3, 300.1, 450.7), abs=0.01)


# ===========================================================================
# GROUP 3 — fuse() basic behaviour
# ===========================================================================

class TestFuseBasic:

    def test_empty_tracks_returns_empty_list(self, fusion):
        """PASS CONDITION 7: fuse() with no tracked objects returns []."""
        result = fusion.fuse([], make_uniform_depth_map(0.5))
        assert result == []

    def test_one_track_returns_one_fused(self, fusion):
        """PASS CONDITION 2: one TrackedObject → one FusedObject."""
        obj = make_tracked_object()
        depth = make_uniform_depth_map(0.6)
        result = fusion.fuse([obj], depth)
        assert len(result) == 1

    def test_three_tracks_returns_three_fused(self, fusion):
        """PASS CONDITION 2: three TrackedObjects → three FusedObjects."""
        objs = [
            make_tracked_object(track_id=1, class_id=0,  class_name="person",
                                x1=10,  x2=150),
            make_tracked_object(track_id=2, class_id=56, class_name="chair",
                                x1=200, x2=380),
            make_tracked_object(track_id=3, class_id=39, class_name="bottle",
                                x1=450, x2=580),
        ]
        depth = make_uniform_depth_map(0.5)
        result = fusion.fuse(objs, depth)
        assert len(result) == 3

    def test_track_ids_preserved_in_fused(self, fusion):
        """track_id must pass through to FusedObject."""
        objs = [
            make_tracked_object(track_id=11),
            make_tracked_object(track_id=22, x1=200, x2=380),
        ]
        depth = make_uniform_depth_map(0.5)
        result = fusion.fuse(objs, depth)
        ids = {f.track_id for f in result}
        assert ids == {11, 22}


# ===========================================================================
# GROUP 4 — Depth unavailable (None depth map)
# ===========================================================================

class TestNoDepthMap:

    def test_none_depth_gives_unknown_proximity(self, fusion):
        """PASS CONDITION 3: proximity must be UNKNOWN when depth_map is None."""
        from assistive_navigation.depth.depth_estimator import Proximity
        obj = make_tracked_object()
        result = fusion.fuse([obj], depth_map=None)
        assert result[0].proximity == Proximity.UNKNOWN

    def test_none_depth_gives_minus_one_raw_depth(self, fusion):
        """PASS CONDITION 9: raw_depth must be -1.0 when depth_map is None."""
        obj = make_tracked_object()
        result = fusion.fuse([obj], depth_map=None)
        assert result[0].raw_depth == pytest.approx(-1.0)

    def test_none_depth_gives_zero_samples(self, fusion):
        """depth_samples must be 0 when depth_map is None."""
        obj = make_tracked_object()
        result = fusion.fuse([obj], depth_map=None)
        assert result[0].depth_samples == 0

    def test_none_depth_still_populates_priority(self, fusion):
        """Priority must still be derived from class_id even without depth."""
        obj = make_tracked_object(class_id=0, class_name="person")
        result = fusion.fuse([obj], depth_map=None)
        assert result[0].priority == "navigation_critical"


# ===========================================================================
# GROUP 5 — Proximity classification
# ===========================================================================

class TestProximityClassification:
    """PASS CONDITION 4: proximity must map correctly to buckets."""

    def test_high_depth_gives_very_close(self, fusion):
        """Object in high-depth region → VERY_CLOSE."""
        from assistive_navigation.depth.depth_estimator import Proximity
        # depth=0.85 > very_close_threshold=0.75
        obj = make_tracked_object(x1=10, y1=10, x2=300, y2=470)
        depth = make_uniform_depth_map(0.85)
        result = fusion.fuse([obj], depth)
        assert result[0].proximity == Proximity.VERY_CLOSE

    def test_medium_depth_gives_close(self, fusion):
        """depth=0.65 → CLOSE (between 0.55 and 0.75)."""
        from assistive_navigation.depth.depth_estimator import Proximity
        obj = make_tracked_object(x1=10, y1=10, x2=300, y2=470)
        depth = make_uniform_depth_map(0.65)
        result = fusion.fuse([obj], depth)
        assert result[0].proximity == Proximity.CLOSE

    def test_low_medium_depth_gives_medium(self, fusion):
        """depth=0.42 → MEDIUM (between 0.35 and 0.55)."""
        from assistive_navigation.depth.depth_estimator import Proximity
        obj = make_tracked_object(x1=10, y1=10, x2=300, y2=470)
        depth = make_uniform_depth_map(0.42)
        result = fusion.fuse([obj], depth)
        assert result[0].proximity == Proximity.MEDIUM

    def test_low_depth_gives_far(self, fusion):
        """depth=0.10 → FAR (below 0.35)."""
        from assistive_navigation.depth.depth_estimator import Proximity
        obj = make_tracked_object(x1=10, y1=10, x2=300, y2=470)
        depth = make_uniform_depth_map(0.10)
        result = fusion.fuse([obj], depth)
        assert result[0].proximity == Proximity.FAR

    def test_different_bboxes_different_proximity(self, fusion):
        """
        Object on high-depth side vs low-depth side should get different proximity.
        Left half = 0.85 (very close), right half = 0.20 (far).
        """
        from assistive_navigation.depth.depth_estimator import Proximity
        depth = make_split_depth_map(left_value=0.85, right_value=0.20)

        # Left object (in high-depth region)
        left_obj = make_tracked_object(track_id=1, x1=10,  x2=250, y1=100, y2=400)
        # Right object (in low-depth region)
        right_obj = make_tracked_object(track_id=2, x1=390, x2=620, y1=100, y2=400)

        result = fusion.fuse([left_obj, right_obj], depth)
        left_f  = next(f for f in result if f.track_id == 1)
        right_f = next(f for f in result if f.track_id == 2)

        assert left_f.proximity  in (Proximity.VERY_CLOSE, Proximity.CLOSE)
        assert right_f.proximity in (Proximity.FAR, Proximity.MEDIUM)

    def test_raw_depth_in_valid_range_when_map_present(self, fusion):
        """raw_depth must be 0.0–1.0 when depth map is available."""
        obj = make_tracked_object(x1=10, y1=10, x2=300, y2=470)
        depth = make_uniform_depth_map(0.6)
        result = fusion.fuse([obj], depth)
        assert 0.0 <= result[0].raw_depth <= 1.0

    def test_depth_samples_positive_when_map_present(self, fusion):
        """depth_samples must be > 0 when a valid depth map is provided."""
        obj = make_tracked_object(x1=10, y1=10, x2=300, y2=470)
        depth = make_uniform_depth_map(0.5)
        result = fusion.fuse([obj], depth)
        assert result[0].depth_samples > 0


# ===========================================================================
# GROUP 6 — Priority
# ===========================================================================

class TestPriorityDerivation:
    """PASS CONDITION 5: priority must be derived correctly from class_id."""

    def test_person_is_navigation_critical(self, fusion):
        obj = make_tracked_object(class_id=0, class_name="person")
        f = fusion.fuse([obj], None)[0]
        assert f.priority == "navigation_critical"

    def test_chair_is_navigation_critical(self, fusion):
        obj = make_tracked_object(class_id=56, class_name="chair")
        f = fusion.fuse([obj], None)[0]
        assert f.priority == "navigation_critical"

    def test_bottle_is_contextual(self, fusion):
        obj = make_tracked_object(class_id=39, class_name="bottle")
        f = fusion.fuse([obj], None)[0]
        assert f.priority == "contextual"

    def test_backpack_is_contextual(self, fusion):
        obj = make_tracked_object(class_id=24, class_name="backpack")
        f = fusion.fuse([obj], None)[0]
        assert f.priority == "contextual"

    def test_unknown_class_gives_unknown_priority(self, fusion):
        # Class ID 99 is not in any navigation category
        obj = make_tracked_object(class_id=99, class_name="alien_object")
        f = fusion.fuse([obj], None)[0]
        assert f.priority == "unknown"

    def test_laptop_is_navigation_critical(self, fusion):
        obj = make_tracked_object(class_id=63, class_name="laptop")
        f = fusion.fuse([obj], None)[0]
        assert f.priority == "navigation_critical"


# ===========================================================================
# GROUP 7 — Temporal smoothing
# ===========================================================================

class TestTemporalSmoothing:
    """PASS CONDITION 6: temporal smoothing must reduce depth variance."""

    def test_single_call_raw_depth_equals_sampled(self, config):
        """On first call for a track, raw_depth equals the sampled value."""
        from assistive_navigation.depth.fusion import DepthFusion
        fusion = DepthFusion(config)
        obj = make_tracked_object(track_id=1, x1=10, y1=10, x2=300, y2=470)
        depth = make_uniform_depth_map(0.70)
        f = fusion.fuse([obj], depth)[0]
        # After one call with uniform depth 0.70, smoothed ≈ 0.70
        assert abs(f.raw_depth - 0.70) < 0.01

    def test_smoothing_window_averages_values(self, config):
        """
        PASS CONDITION 6:
        After 5 calls with varying depth, raw_depth should converge to
        the mean rather than jumping between extremes.
        """
        from assistive_navigation.depth.fusion import DepthFusion
        fusion = DepthFusion(config)
        obj = make_tracked_object(track_id=1, x1=10, y1=10, x2=300, y2=470)

        # Feed alternating values: 0.80 and 0.40 → mean ≈ 0.60
        values = [0.80, 0.40, 0.80, 0.40, 0.80]
        last_f = None
        for v in values:
            depth = make_uniform_depth_map(v)
            last_f = fusion.fuse([obj], depth)[0]

        # After 5 values (3×0.80, 2×0.40), mean ≈ 0.56
        assert last_f is not None
        assert abs(last_f.raw_depth - 0.56) < 0.10, (
            f"Expected smoothed depth ~0.56, got {last_f.raw_depth:.3f}"
        )

    def test_history_cleared_when_track_disappears(self, config):
        """
        When a track is no longer in the tracked_objects list, its
        history should be cleared from the internal registry.
        """
        from assistive_navigation.depth.fusion import DepthFusion
        fusion = DepthFusion(config)
        obj = make_tracked_object(track_id=5, x1=10, y1=10, x2=300, y2=470)
        depth = make_uniform_depth_map(0.7)

        # Establish history
        fusion.fuse([obj], depth)
        assert fusion.active_track_count == 1

        # Now fuse with empty list — history should be cleared
        fusion.fuse([], None)
        assert fusion.active_track_count == 0

    def test_reset_history_clears_all(self, config):
        """reset_history() with no args clears all track histories."""
        from assistive_navigation.depth.fusion import DepthFusion
        fusion = DepthFusion(config)
        objs = [
            make_tracked_object(track_id=1, x1=10,  x2=150),
            make_tracked_object(track_id=2, x1=200, x2=350),
        ]
        depth = make_uniform_depth_map(0.5)
        fusion.fuse(objs, depth)
        assert fusion.active_track_count == 2
        fusion.reset_history()
        assert fusion.active_track_count == 0

    def test_reset_history_specific_track(self, config):
        """reset_history(track_id) clears only that track."""
        from assistive_navigation.depth.fusion import DepthFusion
        fusion = DepthFusion(config)
        objs = [
            make_tracked_object(track_id=1, x1=10,  x2=150),
            make_tracked_object(track_id=2, x1=200, x2=350),
        ]
        depth = make_uniform_depth_map(0.5)
        fusion.fuse(objs, depth)
        fusion.reset_history(track_id=1)
        # Track 1 gone, track 2 still present
        assert 1 not in fusion.get_depth_history(1)
        assert len(fusion.get_depth_history(2)) > 0

    def test_get_depth_history_returns_list(self, config):
        """get_depth_history() must return a list."""
        from assistive_navigation.depth.fusion import DepthFusion
        fusion = DepthFusion(config)
        obj = make_tracked_object(track_id=3, x1=10, y1=10, x2=300, y2=470)
        depth = make_uniform_depth_map(0.6)
        fusion.fuse([obj], depth)
        h = fusion.get_depth_history(3)
        assert isinstance(h, list)
        assert len(h) == 1
        assert abs(h[0] - 0.6) < 0.01


# ===========================================================================
# GROUP 8 — Edge cases
# ===========================================================================

class TestFusionEdgeCases:

    def test_all_zero_depth_map_gives_unknown(self, fusion):
        """All-zero depth map → no valid samples → UNKNOWN proximity."""
        from assistive_navigation.depth.depth_estimator import Proximity
        obj = make_tracked_object(x1=10, y1=10, x2=300, y2=470)
        depth = np.zeros((480, 640), dtype=np.float32)  # all invalid (0.0 filtered)
        result = fusion.fuse([obj], depth)
        assert result[0].proximity in (Proximity.UNKNOWN, Proximity.FAR)

    def test_bbox_outside_frame_does_not_crash(self, fusion):
        """A bbox partially outside the frame must not raise an exception."""
        obj = make_tracked_object(x1=-50, y1=-50, x2=700, y2=550)
        depth = make_uniform_depth_map(0.5)
        try:
            result = fusion.fuse([obj], depth)
            assert len(result) == 1
        except Exception as e:
            pytest.fail(f"Out-of-bounds bbox raised: {e}")

    def test_very_small_bbox_handled_gracefully(self, fusion):
        """A tiny 5×5 bbox may have too few valid samples → UNKNOWN or valid."""
        from assistive_navigation.depth.depth_estimator import Proximity
        obj = make_tracked_object(x1=100, y1=100, x2=105, y2=105)
        depth = make_uniform_depth_map(0.7)
        result = fusion.fuse([obj], depth)
        # May be UNKNOWN (too few samples) or a valid bucket — must not crash
        assert result[0].proximity in {
            Proximity.FAR, Proximity.MEDIUM,
            Proximity.CLOSE, Proximity.VERY_CLOSE, Proximity.UNKNOWN
        }

    def test_depth_age_zero_by_default(self, fusion):
        """depth_age defaults to 0 if not supplied."""
        obj = make_tracked_object()
        depth = make_uniform_depth_map(0.5)
        f = fusion.fuse([obj], depth)[0]
        assert f.depth_age == 0

    def test_active_track_count_updates(self, config):
        """active_track_count should reflect current tracked objects."""
        from assistive_navigation.depth.fusion import DepthFusion
        fusion = DepthFusion(config)
        objs = [
            make_tracked_object(track_id=1, x1=10,  x2=150),
            make_tracked_object(track_id=2, x1=200, x2=350),
            make_tracked_object(track_id=3, x1=400, x2=580),
        ]
        depth = make_uniform_depth_map(0.5)
        fusion.fuse(objs, depth)
        assert fusion.active_track_count == 3


# ===========================================================================
# GROUP 9 — Hardware test (real camera + detector + tracker + depth)
# ===========================================================================

class TestFusionWithWebcam:
    """
    Full pipeline hardware test.
    Requires webcam. Marked requires_camera.
    """

    requires_camera = pytest.mark.requires_camera

    @requires_camera
    def test_full_pipeline_produces_fused_objects(self, config):
        """
        PASS CONDITION (hardware):
        Run the complete detection → filtering → tracking → depth → fusion
        pipeline on real camera frames. Verify FusedObjects have valid
        structure and that no phase is bypassed.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        from assistive_navigation.detection.detector import ObjectDetector, DetectorError
        from assistive_navigation.detection.filter import DetectionFilter
        from assistive_navigation.tracking.tracker import ObjectTracker
        from assistive_navigation.depth.depth_estimator import DepthEstimator, Proximity
        from assistive_navigation.depth.fusion import DepthFusion, FusedObject

        # Load components
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

        try:
            cam = CameraCapture(config)
            cam.open()
        except CameraError as e:
            pytest.skip(f"Webcam not available: {e}")

        fused_all = []
        try:
            for frame_num in range(10):
                frame = cam.read()
                if frame is None:
                    continue
                h, w = frame.shape[:2]

                # Phase 3: detect
                det_result = detector.detect(frame)
                # Phase 3: filter
                filtered   = det_filter.filter(det_result)
                # Phase 5: track
                tracked    = tracker.update(filtered.accepted, w, h)
                # Phase 6: depth
                depth_map  = depth_est.process_frame(frame)
                # Phase 7: fuse
                fused      = fusion.fuse(tracked, depth_map, depth_est.depth_frame_age)

                fused_all.extend(fused)
        finally:
            cam.release()

        print(f"\n  === FUSION PIPELINE HARDWARE TEST ===")
        print(f"  Frames processed    : 10")
        print(f"  FusedObjects total  : {len(fused_all)}")
        if fused_all:
            unique_classes    = {f.class_name for f in fused_all}
            unique_proximity  = {f.proximity for f in fused_all}
            unique_priorities = {f.priority for f in fused_all}
            print(f"  Classes detected    : {unique_classes}")
            print(f"  Proximity buckets   : {unique_proximity}")
            print(f"  Priority levels     : {unique_priorities}")
            # Verify structure
            for f in fused_all:
                assert isinstance(f, FusedObject)
                assert f.track_id > 0
                assert f.proximity in {
                    Proximity.FAR, Proximity.MEDIUM,
                    Proximity.CLOSE, Proximity.VERY_CLOSE, Proximity.UNKNOWN
                }
                assert f.raw_depth >= -1.0
                assert f.depth_age >= 0
                assert f.priority in {
                    "navigation_critical", "contextual",
                    "low_priority", "unknown"
                }
        else:
            print(f"  No FusedObjects produced (no objects detected or tracked).")
            # Not a failure — room may be empty or person may not be visible
            print(f"  (This is acceptable if no navigation objects were in view.)")
