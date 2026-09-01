"""
Unit Tests - Object Detector and Detection Filter
===================================================
Phase 3 PASS conditions:

  [1] Detector loads YOLO11n without error
  [2] Known objects detected in a real frame
  [3] Irrelevant/ignored detections logged, not passed through filter
  [4] Confidence recorded on every detection
  [5] Inference latency measured and reported (benchmark target: < 200ms,
      but actual result is reported honestly — not a hard failure)
  [6] All tests pass

Tests are split into groups:
  - Logic tests (no model/camera needed): always run
  - Model tests (model required, no camera): run after load()
  - Hardware tests (camera required): marked requires_camera

How to run all tests:
    .venv/Scripts/python.exe -m pytest tests/unit/test_detector.py -v -s

How to run only logic tests (no download, no camera):
    .venv/Scripts/python.exe -m pytest tests/unit/test_detector.py -v -m "not requires_camera and not requires_model"
"""

import pytest
import numpy as np
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    """Load real project config once per module."""
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


@pytest.fixture(scope="module")
def loaded_detector(config):
    """
    Load YOLO11n once per test module.
    This may download ~6 MB on first run.
    Shared by all model/hardware tests to avoid reloading repeatedly.
    """
    from assistive_navigation.detection.detector import ObjectDetector, DetectorError
    detector = ObjectDetector(config)
    try:
        detector.load()
    except DetectorError as e:
        pytest.skip(f"Could not load YOLO model: {e}")
    return detector


@pytest.fixture(scope="module")
def det_filter(config):
    """DetectionFilter instance shared across all filter tests."""
    from assistive_navigation.detection.filter import DetectionFilter
    return DetectionFilter(config)


@pytest.fixture(scope="module")
def camera_frame(config):
    """
    Open the webcam, capture one frame, close the camera.
    Used for hardware tests. Skipped if webcam unavailable.
    """
    from assistive_navigation.camera.capture import CameraCapture, CameraError
    try:
        with CameraCapture(config) as cam:
            frame = cam.read()
    except CameraError as e:
        pytest.skip(f"Webcam not available: {e}")
    if frame is None:
        pytest.skip("Webcam returned None frame.")
    return frame


# ===========================================================================
# GROUP 1 — Logic tests (no model or camera required)
# ===========================================================================

class TestDetectorImports:
    """Verify all Phase 3 classes are importable and structured correctly."""

    def test_detector_error_importable(self):
        from assistive_navigation.detection.detector import DetectorError
        assert issubclass(DetectorError, Exception)

    def test_detection_importable(self):
        from assistive_navigation.detection.detector import Detection
        assert Detection is not None

    def test_detection_result_importable(self):
        from assistive_navigation.detection.detector import DetectionResult
        assert DetectionResult is not None

    def test_object_detector_importable(self):
        from assistive_navigation.detection.detector import ObjectDetector
        assert ObjectDetector is not None

    def test_detection_filter_importable(self):
        from assistive_navigation.detection.filter import DetectionFilter
        assert DetectionFilter is not None

    def test_filter_result_importable(self):
        from assistive_navigation.detection.filter import FilterResult
        assert FilterResult is not None

    def test_priority_constants_importable(self):
        from assistive_navigation.detection.filter import Priority
        assert Priority.CRITICAL == "navigation_critical"
        assert Priority.IGNORED == "ignored"
        assert Priority.UNKNOWN == "unknown"

    def test_rejection_reason_constants_importable(self):
        from assistive_navigation.detection.filter import RejectionReason
        assert RejectionReason.LOW_CONFIDENCE == "low_confidence"
        assert RejectionReason.CLASS_IGNORED == "class_ignored"
        assert RejectionReason.CLASS_NOT_IN_LIST == "class_not_in_list"


class TestDetectionDataclass:
    """Verify Detection and DetectionResult have the correct fields."""

    def test_detection_has_required_fields(self):
        from assistive_navigation.detection.detector import Detection
        det = Detection(
            class_id=0, class_name="person", confidence=0.91,
            bbox=(10, 20, 100, 200), bbox_norm=(0.01, 0.04, 0.16, 0.42),
            center_x=0.09, center_y=0.23, area_norm=0.06,
            frame_width=640, frame_height=480
        )
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == 0.91
        assert det.bbox == (10, 20, 100, 200)
        assert det.frame_width == 640

    def test_detection_repr_contains_class_name(self):
        from assistive_navigation.detection.detector import Detection
        det = Detection(
            class_id=56, class_name="chair", confidence=0.75,
            bbox=(0, 0, 50, 100), bbox_norm=(0, 0, 0.08, 0.21),
            center_x=0.04, center_y=0.10, area_norm=0.02,
            frame_width=640, frame_height=480
        )
        assert "chair" in repr(det)
        assert "0.75" in repr(det)


class TestDetectorBeforeLoad:
    """Tests for ObjectDetector before load() is called."""

    def test_is_loaded_false_before_load(self, config):
        from assistive_navigation.detection.detector import ObjectDetector
        d = ObjectDetector(config)
        assert d.is_loaded is False

    def test_detect_before_load_raises_detector_error(self, config):
        from assistive_navigation.detection.detector import ObjectDetector, DetectorError
        d = ObjectDetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(DetectorError, match="detect\\(\\) called before load"):
            d.detect(frame)

    def test_detect_none_frame_raises_value_error(self, config, loaded_detector):
        """Passing None as a frame must raise ValueError, not crash."""
        with pytest.raises(ValueError, match="None frame"):
            loaded_detector.detect(None)

    def test_detect_wrong_shape_raises_value_error(self, config, loaded_detector):
        """A grayscale (H, W) frame must raise ValueError."""
        gray = np.zeros((480, 640), dtype=np.uint8)
        with pytest.raises(ValueError):
            loaded_detector.detect(gray)


class TestDetectionFilterLogic:
    """
    Unit tests for DetectionFilter using synthetic (mock) detections.
    No model or camera needed — we create Detection objects directly.
    """

    def _make_detection(self, class_id, class_name, confidence=0.8):
        """Helper: create a synthetic Detection with given class and confidence."""
        from assistive_navigation.detection.detector import Detection
        return Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox=(10, 10, 100, 100),
            bbox_norm=(0.015, 0.02, 0.156, 0.208),
            center_x=0.086,
            center_y=0.115,
            area_norm=0.022,
            frame_width=640,
            frame_height=480
        )

    def _make_result(self, detections):
        """Helper: wrap a list of Detections in a DetectionResult."""
        from assistive_navigation.detection.detector import DetectionResult
        return DetectionResult(
            detections=detections,
            latency_ms=42.0,
            frame_width=640,
            frame_height=480,
            raw_count=len(detections)
        )

    def test_person_is_accepted(self, det_filter):
        """Person (class 0) is navigation_critical and must be accepted."""
        det = self._make_detection(0, "person", confidence=0.85)
        result = det_filter.filter(self._make_result([det]))
        assert result.accepted_count == 1
        assert result.accepted[0].class_name == "person"

    def test_chair_is_accepted(self, det_filter):
        """Chair (class 56) is navigation_critical and must be accepted."""
        det = self._make_detection(56, "chair", confidence=0.72)
        result = det_filter.filter(self._make_result([det]))
        assert result.accepted_count == 1

    def test_bottle_is_accepted_as_contextual(self, det_filter):
        """Bottle (class 39) is contextual priority and must be accepted."""
        det = self._make_detection(39, "bottle", confidence=0.65)
        result = det_filter.filter(self._make_result([det]))
        assert result.accepted_count == 1
        from assistive_navigation.detection.filter import Priority
        assert result.get_priority(result.accepted[0]) == Priority.CONTEXTUAL

    def test_person_has_critical_priority(self, det_filter):
        """Person must have CRITICAL priority after filtering."""
        from assistive_navigation.detection.filter import Priority
        det = self._make_detection(0, "person", confidence=0.9)
        result = det_filter.filter(self._make_result([det]))
        assert result.accepted_count == 1
        assert result.get_priority(result.accepted[0]) == Priority.CRITICAL

    # --- PASS CONDITION 3: ignored classes must be rejected ---

    def test_toothbrush_is_rejected_class_ignored(self, det_filter):
        """
        PASS CONDITION 3a:
        Toothbrush (class 79) caused phantom detections in V1.
        It must be rejected with reason 'class_ignored', not silently dropped.
        """
        from assistive_navigation.detection.filter import RejectionReason
        det = self._make_detection(79, "toothbrush", confidence=0.82)
        result = det_filter.filter(self._make_result([det]))
        assert result.accepted_count == 0, "Toothbrush must not be accepted"
        assert result.rejected_count == 1
        assert result.rejected[0]["reason"] == RejectionReason.CLASS_IGNORED
        assert result.rejected[0]["class_name"] == "toothbrush"

    def test_tie_is_rejected_class_ignored(self, det_filter):
        """
        PASS CONDITION 3b:
        Tie (class 27) caused phantom detections in V1.
        Must be rejected as class_ignored.
        """
        from assistive_navigation.detection.filter import RejectionReason
        det = self._make_detection(27, "tie", confidence=0.77)
        result = det_filter.filter(self._make_result([det]))
        assert result.accepted_count == 0
        assert result.rejected[0]["reason"] == RejectionReason.CLASS_IGNORED

    def test_remote_is_rejected_class_ignored(self, det_filter):
        """
        Remote (class 65) was confused with phone in V1.
        Must be rejected as class_ignored.
        """
        from assistive_navigation.detection.filter import RejectionReason
        det = self._make_detection(65, "remote", confidence=0.80)
        result = det_filter.filter(self._make_result([det]))
        assert result.accepted_count == 0
        assert result.rejected[0]["reason"] == RejectionReason.CLASS_IGNORED

    def test_baseball_bat_is_rejected_class_ignored(self, det_filter):
        """
        Baseball bat (class 34) was triggered by pens/pencils in V1.
        Must be rejected as class_ignored.
        """
        from assistive_navigation.detection.filter import RejectionReason
        det = self._make_detection(34, "baseball bat", confidence=0.73)
        result = det_filter.filter(self._make_result([det]))
        assert result.accepted_count == 0
        assert result.rejected[0]["reason"] == RejectionReason.CLASS_IGNORED

    def test_low_confidence_is_rejected(self, det_filter):
        """
        PASS CONDITION 4 (inverse):
        A detection below confidence_threshold (0.35) must be rejected
        with reason 'low_confidence', even if it is a navigation class.
        """
        from assistive_navigation.detection.filter import RejectionReason
        # Person at 0.20 conf — below the 0.35 threshold
        det = self._make_detection(0, "person", confidence=0.20)
        result = det_filter.filter(self._make_result([det]))
        assert result.accepted_count == 0
        assert result.rejected[0]["reason"] == RejectionReason.LOW_CONFIDENCE

    def test_unknown_class_is_rejected_not_in_list(self, det_filter):
        """
        A COCO class not in any navigation category must be rejected
        with reason 'class_not_in_list'.
        E.g. kite (33) is in ignored list, but what about class 99?
        """
        from assistive_navigation.detection.filter import RejectionReason
        # Class 99 does not exist in COCO and is not in any nav list
        det = self._make_detection(99, "unknown_object", confidence=0.6)
        result = det_filter.filter(self._make_result([det]))
        assert result.accepted_count == 0
        assert result.rejected[0]["reason"] in (
            RejectionReason.CLASS_IGNORED,
            RejectionReason.CLASS_NOT_IN_LIST
        )

    def test_mixed_detections_separated_correctly(self, det_filter):
        """
        Multiple detections: some accepted, some rejected.
        Verify correct separation and that rejected contains reason codes.
        """
        from assistive_navigation.detection.filter import RejectionReason
        detections = [
            self._make_detection(0, "person", confidence=0.90),     # ACCEPT
            self._make_detection(79, "toothbrush", confidence=0.85), # REJECT: ignored
            self._make_detection(56, "chair", confidence=0.70),      # ACCEPT
            self._make_detection(0, "person", confidence=0.15),      # REJECT: low conf
        ]
        result = det_filter.filter(self._make_result(detections))
        assert result.accepted_count == 2
        assert result.rejected_count == 2
        rejection_reasons = {r["reason"] for r in result.rejected}
        assert RejectionReason.CLASS_IGNORED in rejection_reasons
        assert RejectionReason.LOW_CONFIDENCE in rejection_reasons

    def test_empty_input_returns_empty_result(self, det_filter):
        """Filtering zero detections must return a valid empty FilterResult."""
        result = det_filter.filter(self._make_result([]))
        assert result.accepted_count == 0
        assert result.rejected_count == 0
        assert result.raw_count == 0

    def test_filter_result_summary_is_string(self, det_filter):
        """FilterResult.summary() must return a non-empty string."""
        det = self._make_detection(0, "person", confidence=0.9)
        result = det_filter.filter(self._make_result([det]))
        s = result.summary()
        assert isinstance(s, str) and len(s) > 0


# ===========================================================================
# GROUP 2 — Model tests (require model load, no camera needed)
# ===========================================================================

class TestDetectorWithModel:
    """Tests requiring the model to be loaded but no camera."""

    requires_model = pytest.mark.requires_model

    def test_is_loaded_true_after_load(self, loaded_detector):
        """PASS CONDITION 1: is_loaded must be True after load()."""
        assert loaded_detector.is_loaded is True

    def test_model_name_matches_config(self, loaded_detector, config):
        """Model name property must match what was configured."""
        assert loaded_detector.model_name == config["detection"]["model_name"]

    def test_detect_returns_detection_result(self, loaded_detector):
        """detect() on a valid frame must return a DetectionResult."""
        from assistive_navigation.detection.detector import DetectionResult
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = loaded_detector.detect(frame)
        assert isinstance(result, DetectionResult)

    def test_detect_blank_frame_returns_no_detections(self, loaded_detector):
        """
        A completely black frame should produce zero detections.
        (A model should not hallucinate objects in a blank image.)
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = loaded_detector.detect(frame)
        assert result.raw_count == 0, (
            f"Detected {result.raw_count} objects in a completely black frame. "
            f"Detections: {[d.class_name for d in result.detections]}"
        )

    def test_detect_records_latency(self, loaded_detector):
        """
        PASS CONDITION 5:
        Latency must be recorded and be a positive number.
        We report the actual value honestly.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = loaded_detector.detect(frame)
        assert result.latency_ms > 0, "latency_ms must be positive"
        print(f"\n  Inference latency on blank frame: {result.latency_ms:.1f} ms")
        print(f"  Performance target: < 200 ms")
        print(f"  Result: {'on target' if result.latency_ms < 200 else 'above target — acceptable on CPU-only'}")

    def test_detect_records_frame_dimensions(self, loaded_detector):
        """DetectionResult must record the correct frame dimensions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = loaded_detector.detect(frame)
        assert result.frame_width == 640
        assert result.frame_height == 480

    def test_average_latency_increases_after_inferences(self, loaded_detector):
        """average_latency_ms must return a positive value after inferences."""
        # After the tests above, at least 3 inferences have been run
        assert loaded_detector.average_latency_ms > 0


# ===========================================================================
# GROUP 3 — Hardware tests (require webcam)
# ===========================================================================

class TestDetectorWithWebcam:
    """
    Hardware tests — require a real webcam and loaded model.
    Tests detection on real camera frames.
    """

    requires_camera = pytest.mark.requires_camera

    @requires_camera
    def test_detect_real_frame_returns_result(self, loaded_detector, camera_frame):
        """detect() on a real camera frame must return a DetectionResult."""
        from assistive_navigation.detection.detector import DetectionResult
        result = loaded_detector.detect(camera_frame)
        assert isinstance(result, DetectionResult)

    @requires_camera
    def test_real_frame_latency_is_measured(self, loaded_detector, camera_frame):
        """
        PASS CONDITION 5 (hardware):
        Measure and print actual CPU inference latency on a real frame.
        This is the benchmark number — not a hard pass/fail criterion.
        """
        result = loaded_detector.detect(camera_frame)
        assert result.latency_ms > 0
        print(f"\n  === INFERENCE LATENCY BENCHMARK ===")
        print(f"  Single frame latency: {result.latency_ms:.1f} ms")
        print(f"  Equivalent FPS:       {1000/result.latency_ms:.1f}")
        print(f"  Performance target:   < 200 ms")
        print(f"  Hardware:             Intel i5-12450H (CPU-only)")
        if result.latency_ms < 200:
            print(f"  Assessment:           ON TARGET")
        else:
            print(f"  Assessment:           ABOVE TARGET — acceptable on CPU-only")
            print(f"  Note: Phase 14 will optimise to ONNX Runtime for better CPU perf.")

    @requires_camera
    def test_detections_have_valid_structure(self, loaded_detector, camera_frame):
        """
        PASS CONDITION 2 (partial):
        All returned detections must have valid fields.
        Tests structural correctness — not which objects are detected
        (that depends on what is in the camera view).
        """
        result = loaded_detector.detect(camera_frame)
        for det in result.detections:
            assert 0 <= det.class_id <= 79, f"Invalid COCO class_id: {det.class_id}"
            assert 0.0 <= det.confidence <= 1.0, f"Confidence out of range: {det.confidence}"
            x1, y1, x2, y2 = det.bbox
            assert x1 >= 0 and y1 >= 0, "Bbox top-left must be >= 0"
            assert x2 > x1, "Bbox x2 must be > x1"
            assert y2 > y1, "Bbox y2 must be > y1"
            assert x2 <= result.frame_width, f"x2 {x2} > frame_width {result.frame_width}"
            assert y2 <= result.frame_height, f"y2 {y2} > frame_height {result.frame_height}"
            assert 0.0 <= det.center_x <= 1.0
            assert 0.0 <= det.center_y <= 1.0
            assert 0.0 < det.area_norm <= 1.0

    @requires_camera
    def test_all_detections_have_confidence(self, loaded_detector, camera_frame):
        """
        PASS CONDITION 4:
        Every detection must have a confidence score recorded.
        """
        result = loaded_detector.detect(camera_frame)
        for det in result.detections:
            assert det.confidence > 0.0, (
                f"{det.class_name} detection has confidence=0.0"
            )

    @requires_camera
    def test_detection_fps_over_30_frames(self, loaded_detector, config):
        """
        PASS CONDITION 5 (sustained):
        Measure detection FPS over 30 consecutive real frames.
        Reports actual result — not a hard pass/fail on FPS.
        """
        import time
        from assistive_navigation.camera.capture import CameraCapture, CameraError

        try:
            with CameraCapture(config) as cam:
                # Warm up
                for _ in range(3):
                    cam.read()

                frames_processed = 0
                total_latency = 0.0
                n = 30
                t_start = time.perf_counter()

                for _ in range(n):
                    frame = cam.read()
                    if frame is None:
                        continue
                    result = loaded_detector.detect(frame)
                    total_latency += result.latency_ms
                    frames_processed += 1

        except CameraError:
            pytest.skip("Webcam not available for FPS benchmark.")

        elapsed = time.perf_counter() - t_start
        pipeline_fps = frames_processed / elapsed
        avg_latency = total_latency / frames_processed if frames_processed else 0

        print(f"\n  === DETECTOR BENCHMARK (30 frames) ===")
        print(f"  Frames processed:     {frames_processed}")
        print(f"  Total elapsed time:   {elapsed:.2f} s")
        print(f"  Pipeline FPS:         {pipeline_fps:.1f}")
        print(f"  Avg inference latency:{avg_latency:.1f} ms")
        print(f"  Performance target:   > 5 FPS (navigation), > 10 FPS (good)")
        if pipeline_fps >= 10:
            print(f"  Assessment:           GOOD")
        elif pipeline_fps >= 5:
            print(f"  Assessment:           ACCEPTABLE for navigation")
        else:
            print(f"  Assessment:           BELOW TARGET — check Phase 14 ONNX optimisation")

        assert frames_processed > 0, "No frames were processed."
        assert pipeline_fps > 0, "Pipeline FPS must be positive."

    @requires_camera
    def test_filter_on_real_frame(self, loaded_detector, det_filter, camera_frame, config):
        """
        PASS CONDITION 3 (hardware):
        Run filter on a real frame. Print what was accepted and rejected.
        Verifies the three-layer model works end-to-end on real data.
        """
        result = loaded_detector.detect(camera_frame)
        filtered = det_filter.filter(result)

        print(f"\n  === FILTER RESULT ON REAL FRAME ===")
        print(f"  Raw detections:  {result.raw_count}")
        print(f"  Accepted:        {filtered.accepted_count}")
        print(f"  Rejected:        {filtered.rejected_count}")
        print(f"  Accepted classes: {[d.class_name for d in filtered.accepted]}")
        if filtered.rejected:
            by_reason: dict = {}
            for r in filtered.rejected:
                by_reason.setdefault(r["reason"], []).append(r["class_name"])
            for reason, names in by_reason.items():
                print(f"  Rejected [{reason}]: {names}")

        # Structural checks — independent of what is in the camera view
        conf_threshold = config["detection"]["confidence_threshold"]
        assert isinstance(filtered.accepted, list)
        assert isinstance(filtered.rejected, list)
        assert filtered.accepted_count + filtered.rejected_count <= result.raw_count
        for det in filtered.accepted:
            assert det.confidence >= conf_threshold
