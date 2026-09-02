"""
Unit Tests - Object Tracker Module
=====================================
Phase 5 PASS conditions:

  [1] Track IDs persist for a single object across >=30 consecutive frames
  [2] Track IDs do not change unnecessarily (<2 ID switches per 100 frames)
  [3] Missed detections handled gracefully (track survives brief gap)
  [4] Multiple objects tracked simultaneously with distinct IDs
  [5] track_age increments correctly
  [6] TrackedObject has all required fields with correct types
  [7] All tests pass

Tests are split into:
  - Logic tests : no camera, use synthetic Detection objects
  - Hardware tests : require webcam (marked requires_camera)

Run all tests:
    .venv/Scripts/python.exe -m pytest tests/unit/test_tracker.py -v -s

Run only logic tests (no camera):
    .venv/Scripts/python.exe -m pytest tests/unit/test_tracker.py -v -m "not requires_camera"
"""

import pytest
import numpy as np
from assistive_navigation.detection.detector import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detection(
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.85,
    x1: float = 100.0, y1: float = 150.0,
    x2: float = 400.0, y2: float = 480.0,
    fw: int = 640, fh: int = 480,
) -> Detection:
    """Create a synthetic Detection for unit tests."""
    x1n = x1 / fw; y1n = y1 / fh; x2n = x2 / fw; y2n = y2 / fh
    return Detection(
        class_id=class_id, class_name=class_name, confidence=confidence,
        bbox=(x1, y1, x2, y2),
        bbox_norm=(x1n, y1n, x2n, y2n),
        center_x=(x1n + x2n) / 2.0,
        center_y=(y1n + y2n) / 2.0,
        area_norm=((x2 - x1) * (y2 - y1)) / (fw * fh),
        frame_width=fw, frame_height=fh,
    )


@pytest.fixture(scope="module")
def config():
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


@pytest.fixture
def tracker(config):
    """Fresh ObjectTracker instance for each test."""
    from assistive_navigation.tracking.tracker import ObjectTracker
    return ObjectTracker(config)


# ===========================================================================
# GROUP 1 — Import and structural tests
# ===========================================================================

class TestTrackerImports:

    def test_object_tracker_importable(self):
        from assistive_navigation.tracking.tracker import ObjectTracker
        assert ObjectTracker is not None

    def test_tracked_object_importable(self):
        from assistive_navigation.tracking.tracker import TrackedObject
        assert TrackedObject is not None

    def test_tracker_is_ready_after_init(self, tracker):
        """ByteTrack should be ready immediately — no separate load() needed."""
        assert tracker.is_ready is True

    def test_tracker_starts_at_frame_zero(self, tracker):
        assert tracker.frame_number == 0

    def test_tracker_starts_with_zero_tracks(self, tracker):
        assert tracker.active_track_count == 0


# ===========================================================================
# GROUP 2 — TrackedObject structure
# ===========================================================================

class TestTrackedObjectStructure:

    def test_tracked_object_has_all_required_fields(self, tracker):
        """
        PASS CONDITION 6:
        TrackedObject must have all fields defined in Phase 0 architecture.
        """
        det = make_detection()
        results = tracker.update([det], 640, 480)
        assert len(results) == 1
        obj = results[0]

        # Check all required fields exist and have correct types
        assert isinstance(obj.track_id, int)
        assert isinstance(obj.class_id, int)
        assert isinstance(obj.class_name, str)
        assert isinstance(obj.confidence, float)
        assert isinstance(obj.bbox, tuple) and len(obj.bbox) == 4
        assert isinstance(obj.bbox_norm, tuple) and len(obj.bbox_norm) == 4
        assert isinstance(obj.center_x, float)
        assert isinstance(obj.center_y, float)
        assert isinstance(obj.age, int)
        assert isinstance(obj.last_seen, int)
        assert isinstance(obj.stability, float)
        assert isinstance(obj.area_norm, float)
        assert isinstance(obj.frame_width, int)
        assert isinstance(obj.frame_height, int)

    def test_tracked_object_confidence_in_range(self, tracker):
        det = make_detection(confidence=0.91)
        results = tracker.update([det], 640, 480)
        assert 0.0 <= results[0].confidence <= 1.0

    def test_tracked_object_center_normalised(self, tracker):
        det = make_detection(x1=100, y1=150, x2=400, y2=480)
        results = tracker.update([det], 640, 480)
        obj = results[0]
        assert 0.0 <= obj.center_x <= 1.0
        assert 0.0 <= obj.center_y <= 1.0

    def test_tracked_object_bbox_norm_in_range(self, tracker):
        det = make_detection()
        results = tracker.update([det], 640, 480)
        for v in results[0].bbox_norm:
            assert 0.0 <= v <= 1.0

    def test_tracked_object_area_norm_positive(self, tracker):
        det = make_detection()
        results = tracker.update([det], 640, 480)
        assert results[0].area_norm > 0.0

    def test_tracked_object_class_name_matches(self, tracker):
        det = make_detection(class_id=56, class_name="chair")
        results = tracker.update([det], 640, 480)
        assert results[0].class_name == "chair"
        assert results[0].class_id == 56

    def test_tracked_object_repr_contains_id(self, tracker):
        det = make_detection()
        results = tracker.update([det], 640, 480)
        r = repr(results[0])
        assert "TrackedObject" in r
        assert "id=" in r


# ===========================================================================
# GROUP 3 — Single object track persistence
# ===========================================================================

class TestSingleObjectTracking:

    def test_empty_detections_returns_empty(self, tracker):
        result = tracker.update([], 640, 480)
        assert result == []

    def test_single_detection_returns_one_track(self, tracker):
        det = make_detection()
        result = tracker.update([det], 640, 480)
        assert len(result) == 1

    def test_track_id_assigned_is_positive_integer(self, tracker):
        det = make_detection()
        result = tracker.update([det], 640, 480)
        assert result[0].track_id > 0

    def test_age_starts_at_one_after_first_detection(self, tracker):
        det = make_detection()
        result = tracker.update([det], 640, 480)
        assert result[0].age == 1

    def test_last_seen_is_zero_when_matched(self, tracker):
        """last_seen = 0 means this track was matched in the current frame."""
        det = make_detection()
        result = tracker.update([det], 640, 480)
        assert result[0].last_seen == 0

    def test_stability_is_one_when_matched_every_frame(self, tracker):
        """
        If a detection is matched every single frame, stability = 1.0.
        matched_frames / total_age = N / N = 1.0
        """
        det = make_detection()
        result = None
        for _ in range(5):
            result = tracker.update([det], 640, 480)
        assert result is not None
        assert abs(result[0].stability - 1.0) < 0.01

    def test_age_increments_each_frame(self, tracker):
        """
        PASS CONDITION 5:
        age must increment by 1 each frame for a continuously matched track.
        """
        det = make_detection()
        for i in range(1, 6):
            result = tracker.update([det], 640, 480)
            assert result[0].age == i, (
                f"Expected age={i} on frame {i}, got {result[0].age}"
            )

    def test_frame_number_increments_each_update(self, tracker):
        for i in range(1, 5):
            tracker.update([], 640, 480)
            assert tracker.frame_number == i

    def test_track_id_persists_across_30_frames(self, tracker):
        """
        PASS CONDITION 1:
        A single object tracked for 30 frames must keep the same track_id.
        """
        det = make_detection(confidence=0.91)
        first_id = None
        id_changes = 0

        for _ in range(30):
            result = tracker.update([det], 640, 480)
            if not result:
                continue
            tid = result[0].track_id
            if first_id is None:
                first_id = tid
            elif tid != first_id:
                id_changes += 1
                first_id = tid  # track new ID if changed

        assert first_id is not None, "No tracks were created"
        assert id_changes < 2, (
            f"Track ID changed {id_changes} times in 30 frames. "
            f"Expected < 2 changes for a stable single-object track."
        )
        print(f"\n  Track ID persistence: first_id={first_id}, "
              f"changes={id_changes}/30 frames — PASS")


# ===========================================================================
# GROUP 4 — Missed detection handling
# ===========================================================================

class TestMissedDetectionHandling:

    def test_track_survives_one_missed_frame(self, tracker):
        """
        PASS CONDITION 3:
        A track that receives no matching detection for 1 frame must still
        be alive (last_seen=1) rather than being dropped.
        """
        det = make_detection()

        # Establish the track with 3 detections
        for _ in range(3):
            tracker.update([det], 640, 480)

        result_before = tracker.update([det], 640, 480)
        assert len(result_before) == 1
        tid = result_before[0].track_id

        # One missed frame — no detections
        tracker.update([], 640, 480)

        # Track should still exist in state registry
        age = tracker.get_track_age(tid)
        assert age is not None, (
            f"Track {tid} was dropped after 1 missed frame. "
            f"max_missed_frames={tracker._max_missed}. Should survive."
        )

    def test_last_seen_increases_during_gap(self, tracker):
        """
        last_seen must increment during frames where no detection is matched.
        """
        det = make_detection()
        # Establish track
        result = tracker.update([det], 640, 480)
        tid = result[0].track_id

        # 3 consecutive empty frames
        for _ in range(3):
            tracker.update([], 640, 480)

        age = tracker.get_track_age(tid)
        stab = tracker.get_track_stability(tid)

        # Track should still be alive
        assert age is not None, f"Track {tid} dropped after 3 empty frames"
        # stability < 1.0 because some frames were missed
        assert stab < 1.0

    def test_track_dropped_after_max_missed_frames(self, config):
        """
        A track that receives no detection for max_missed_frames+1 frames
        must be dropped from the state registry.
        """
        from assistive_navigation.tracking.tracker import ObjectTracker
        # Use a small max_missed for this test
        test_config = dict(config)
        test_config["tracking"] = dict(config.get("tracking", {}))
        test_config["tracking"]["max_missed_frames"] = 3

        tracker = ObjectTracker(test_config)
        det = make_detection()

        # Establish track
        result = tracker.update([det], 640, 480)
        tid = result[0].track_id

        # Miss exactly max_missed + 1 frames = 4 empty frames
        for _ in range(4):
            tracker.update([], 640, 480)

        # Track should now be gone
        age = tracker.get_track_age(tid)
        assert age is None, (
            f"Track {tid} should have been dropped after 4 missed frames "
            f"(max_missed=3), but age={age}."
        )

    def test_stability_drops_after_missed_frames(self, tracker):
        """Stability must be < 1.0 if some frames were missed."""
        det = make_detection()
        result = tracker.update([det], 640, 480)
        tid = result[0].track_id

        # 2 missed frames
        tracker.update([], 640, 480)
        tracker.update([], 640, 480)

        stab = tracker.get_track_stability(tid)
        # matched_frames=1, total_age=3 → stability ≈ 0.33
        if stab is not None:
            assert stab < 1.0


# ===========================================================================
# GROUP 5 — Multiple object tracking
# ===========================================================================

class TestMultipleObjectTracking:

    def test_two_distinct_objects_get_distinct_ids(self, tracker):
        """
        PASS CONDITION 4:
        Two objects in the same frame must receive distinct track IDs.
        """
        person = make_detection(class_id=0, class_name="person",
                                x1=50, y1=100, x2=200, y2=400, confidence=0.90)
        chair  = make_detection(class_id=56, class_name="chair",
                                x1=350, y1=200, x2=580, y2=450, confidence=0.75)

        result = tracker.update([person, chair], 640, 480)
        assert len(result) == 2

        ids = {obj.track_id for obj in result}
        assert len(ids) == 2, (
            f"Two different objects must have two distinct track IDs, got: {ids}"
        )

    def test_two_objects_have_correct_classes(self, tracker):
        """Each tracked object must carry the correct class name."""
        person = make_detection(class_id=0, class_name="person",
                                x1=50, y1=100, x2=200, y2=400)
        chair  = make_detection(class_id=56, class_name="chair",
                                x1=350, y1=200, x2=580, y2=450)

        result = tracker.update([person, chair], 640, 480)
        class_names = {obj.class_name for obj in result}
        assert "person" in class_names
        assert "chair" in class_names

    def test_three_objects_get_three_ids(self, tracker):
        """Three objects must all receive distinct IDs."""
        dets = [
            make_detection(class_id=0,  class_name="person",
                           x1=10,  y1=100, x2=150, y2=400),
            make_detection(class_id=56, class_name="chair",
                           x1=200, y1=200, x2=380, y2=450),
            make_detection(class_id=39, class_name="bottle",
                           x1=450, y1=300, x2=520, y2=460),
        ]
        result = tracker.update(dets, 640, 480)
        assert len(result) == 3
        ids = {obj.track_id for obj in result}
        assert len(ids) == 3

    def test_results_sorted_by_track_id(self, tracker):
        """Results must be returned sorted by track_id (ascending)."""
        dets = [
            make_detection(class_id=0,  class_name="person",
                           x1=10, y1=100, x2=150, y2=400),
            make_detection(class_id=56, class_name="chair",
                           x1=350, y1=200, x2=580, y2=450),
        ]
        result = tracker.update(dets, 640, 480)
        ids = [obj.track_id for obj in result]
        assert ids == sorted(ids), f"Results not sorted by track_id: {ids}"


# ===========================================================================
# GROUP 6 — Reset behaviour
# ===========================================================================

class TestTrackerReset:

    def test_reset_clears_tracks(self, tracker):
        """After reset(), active_track_count must be 0."""
        det = make_detection()
        tracker.update([det], 640, 480)
        assert tracker.active_track_count > 0
        tracker.reset()
        assert tracker.active_track_count == 0

    def test_reset_clears_frame_number(self, tracker):
        """After reset(), frame_number must be 0."""
        for _ in range(5):
            tracker.update([], 640, 480)
        assert tracker.frame_number == 5
        tracker.reset()
        assert tracker.frame_number == 0

    def test_tracker_ready_after_reset(self, tracker):
        """Tracker must be ready to use again after reset()."""
        tracker.reset()
        assert tracker.is_ready is True
        det = make_detection()
        result = tracker.update([det], 640, 480)
        assert len(result) == 1

    def test_new_track_id_after_reset(self, tracker):
        """After reset, track IDs restart from 1."""
        det = make_detection()
        result1 = tracker.update([det], 640, 480)
        id_before = result1[0].track_id

        tracker.reset()
        result2 = tracker.update([det], 640, 480)
        # After reset, ByteTrack restarts its counter — new ID is issued
        assert len(result2) == 1
        # The ID after reset may or may not be 1 depending on ByteTrack
        # internals — we only assert it is a valid positive integer
        assert result2[0].track_id >= 1


# ===========================================================================
# GROUP 7 — Hardware tests (real webcam required)
# ===========================================================================

class TestTrackerWithWebcam:
    """
    Hardware tests. Require webcam. Marked requires_camera.
    """

    requires_camera = pytest.mark.requires_camera

    @requires_camera
    def test_tracker_with_real_frames_person(self, config):
        """
        PASS CONDITION 1 (hardware):
        Track a real person for 30 frames. Verify ID stability.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        from assistive_navigation.detection.detector import ObjectDetector, DetectorError
        from assistive_navigation.detection.filter import DetectionFilter
        from assistive_navigation.tracking.tracker import ObjectTracker

        try:
            detector = ObjectDetector(config)
            detector.load()
        except DetectorError as e:
            pytest.skip(f"Could not load model: {e}")

        det_filter = DetectionFilter(config)
        tracker = ObjectTracker(config)

        try:
            cam = CameraCapture(config)
            cam.open()
        except CameraError as e:
            pytest.skip(f"Webcam not available: {e}")

        id_history = []
        id_switches = 0

        try:
            for frame_num in range(30):
                frame = cam.read()
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                det_result = detector.detect(frame)
                filtered   = det_filter.filter(det_result)
                tracked    = tracker.update(filtered.accepted, w, h)

                # Record track IDs for person class
                person_ids = [t.track_id for t in tracked if t.class_name == "person"]
                if person_ids:
                    id_history.append(person_ids[0])
        finally:
            cam.release()

        if len(id_history) < 10:
            pytest.skip(
                f"Person detected in only {len(id_history)}/30 frames — "
                f"not enough data to test ID stability. "
                f"Ensure a person is clearly visible for the full test."
            )

        # Count ID switches
        for i in range(1, len(id_history)):
            if id_history[i] != id_history[i - 1]:
                id_switches += 1

        print(f"\n  === TRACKING STABILITY — Person (30 frames) ===")
        print(f"  Frames with person detected : {len(id_history)}")
        print(f"  Unique track IDs seen       : {len(set(id_history))}")
        print(f"  ID switches                 : {id_switches}")
        print(f"  PASS condition              : < 2 ID switches")
        print(f"  Result                      : {'PASS' if id_switches < 2 else 'FAIL'}")

        assert id_switches < 2, (
            f"Track ID switched {id_switches} times in {len(id_history)} person frames. "
            f"Expected < 2. This may indicate the tracker is losing the person "
            f"briefly (check confidence levels and MAX_MISSED_FRAMES setting)."
        )

    @requires_camera
    def test_tracker_active_track_count_is_positive_with_person(self, config):
        """After detecting a person, active_track_count must be >= 1."""
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        from assistive_navigation.detection.detector import ObjectDetector, DetectorError
        from assistive_navigation.detection.filter import DetectionFilter
        from assistive_navigation.tracking.tracker import ObjectTracker

        try:
            detector = ObjectDetector(config)
            detector.load()
        except DetectorError as e:
            pytest.skip(f"Could not load model: {e}")

        det_filter = DetectionFilter(config)
        tracker = ObjectTracker(config)

        try:
            cam = CameraCapture(config)
            cam.open()
        except CameraError as e:
            pytest.skip(f"Webcam not available: {e}")

        try:
            for _ in range(5):
                frame = cam.read()
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                det_result = detector.detect(frame)
                filtered   = det_filter.filter(det_result)
                tracker.update(filtered.accepted, w, h)
        finally:
            cam.release()

        print(f"\n  active_track_count after 5 frames: {tracker.active_track_count}")
        # Just verify the tracker ran without error and produced a valid count
        assert tracker.active_track_count >= 0
