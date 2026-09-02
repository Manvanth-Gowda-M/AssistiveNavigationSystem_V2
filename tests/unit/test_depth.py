"""
Unit Tests — Depth Estimator Module
=====================================
Phase 6 PASS conditions:

  [1] Model loads without error
  [2] Depth map produced with correct shape (H, W)
  [3] Depth values in range 0.0–1.0 after normalisation
  [4] Invalid/edge-case frames handled without crash
  [5] get_proximity_at_bbox() returns correct proximity bucket
  [6] classify_proximity() returns correct strings at boundaries
  [7] Frame-skip logic works (inference only every N frames)
  [8] Cached depth map returned for non-update frames
  [9] Inference latency measured and reported honestly
 [10] Depth visualisation (colour map) works

Tests are split into:
  - Logic tests   : no model required — test pure Python logic
  - Model tests   : model must be loaded (no camera)
  - Hardware tests: requires webcam (marked requires_camera)

Run all tests:
    .venv/Scripts/python.exe -m pytest tests/unit/test_depth.py -v -s

Run only logic tests (fastest — no model download needed):
    .venv/Scripts/python.exe -m pytest tests/unit/test_depth.py -v -m "not requires_camera" -k "not Model"
"""

import pytest
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


@pytest.fixture(scope="module")
def loaded_estimator(config):
    """
    Load the depth estimator once per test module.
    Skipped if the model file is not present.
    """
    from assistive_navigation.depth.depth_estimator import (
        DepthEstimator, DepthEstimatorError
    )
    est = DepthEstimator(config)
    try:
        est.load()
    except DepthEstimatorError as e:
        pytest.skip(f"Depth model not available: {e}")
    return est


@pytest.fixture
def fresh_estimator(config):
    """Fresh (unloaded) DepthEstimator for each test."""
    from assistive_navigation.depth.depth_estimator import DepthEstimator
    return DepthEstimator(config)


def make_blank_frame(h=480, w=640):
    """All-zero BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_random_frame(h=480, w=640):
    """Random BGR frame."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def make_gradient_frame(h=480, w=640):
    """Frame with a horizontal gradient (darker left, brighter right)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        val = int(x / w * 255)
        frame[:, x, :] = val
    return frame


# ===========================================================================
# GROUP 1 — Import and structure tests
# ===========================================================================

class TestDepthImports:

    def test_depth_estimator_importable(self):
        from assistive_navigation.depth.depth_estimator import DepthEstimator
        assert DepthEstimator is not None

    def test_depth_estimator_error_importable(self):
        from assistive_navigation.depth.depth_estimator import DepthEstimatorError
        assert issubclass(DepthEstimatorError, Exception)

    def test_proximity_constants_importable(self):
        from assistive_navigation.depth.depth_estimator import Proximity
        assert Proximity.FAR        == "FAR"
        assert Proximity.MEDIUM     == "MEDIUM"
        assert Proximity.CLOSE      == "CLOSE"
        assert Proximity.VERY_CLOSE == "VERY_CLOSE"
        assert Proximity.UNKNOWN    == "UNKNOWN"

    def test_estimator_not_loaded_before_load(self, fresh_estimator):
        assert fresh_estimator.is_loaded is False

    def test_has_no_depth_map_before_load(self, fresh_estimator):
        assert fresh_estimator.has_depth_map is False

    def test_process_before_load_raises_error(self, fresh_estimator):
        from assistive_navigation.depth.depth_estimator import DepthEstimatorError
        frame = make_blank_frame()
        with pytest.raises(DepthEstimatorError, match="before load"):
            fresh_estimator.process_frame(frame)

    def test_repr_contains_model_name(self, fresh_estimator):
        r = repr(fresh_estimator)
        assert "DepthEstimator" in r


# ===========================================================================
# GROUP 2 — classify_proximity logic (no model required)
# ===========================================================================

class TestClassifyProximity:
    """
    PASS CONDITION 6:
    classify_proximity() must return correct strings for all threshold regions.
    These tests only require config — no model loading.
    """

    def test_very_close_above_threshold(self, config):
        from assistive_navigation.depth.depth_estimator import (
            DepthEstimator, Proximity
        )
        est = DepthEstimator(config)
        # Default very_close_threshold = 0.75
        assert est.classify_proximity(0.80) == Proximity.VERY_CLOSE
        assert est.classify_proximity(0.76) == Proximity.VERY_CLOSE
        assert est.classify_proximity(1.00) == Proximity.VERY_CLOSE

    def test_close_between_thresholds(self, config):
        from assistive_navigation.depth.depth_estimator import (
            DepthEstimator, Proximity
        )
        est = DepthEstimator(config)
        # close_threshold=0.55, very_close=0.75
        assert est.classify_proximity(0.60) == Proximity.CLOSE
        assert est.classify_proximity(0.56) == Proximity.CLOSE
        assert est.classify_proximity(0.74) == Proximity.CLOSE

    def test_medium_between_thresholds(self, config):
        from assistive_navigation.depth.depth_estimator import (
            DepthEstimator, Proximity
        )
        est = DepthEstimator(config)
        # medium_threshold=0.35, close=0.55
        assert est.classify_proximity(0.40) == Proximity.MEDIUM
        assert est.classify_proximity(0.36) == Proximity.MEDIUM
        assert est.classify_proximity(0.54) == Proximity.MEDIUM

    def test_far_below_medium_threshold(self, config):
        from assistive_navigation.depth.depth_estimator import (
            DepthEstimator, Proximity
        )
        est = DepthEstimator(config)
        assert est.classify_proximity(0.00) == Proximity.FAR
        assert est.classify_proximity(0.10) == Proximity.FAR
        assert est.classify_proximity(0.34) == Proximity.FAR

    def test_classify_returns_string(self, config):
        from assistive_navigation.depth.depth_estimator import DepthEstimator
        est = DepthEstimator(config)
        result = est.classify_proximity(0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_all_four_buckets_reachable(self, config):
        """All four proximity levels must be reachable with valid inputs."""
        from assistive_navigation.depth.depth_estimator import (
            DepthEstimator, Proximity
        )
        est = DepthEstimator(config)
        buckets = {
            est.classify_proximity(0.10),  # FAR
            est.classify_proximity(0.40),  # MEDIUM
            est.classify_proximity(0.60),  # CLOSE
            est.classify_proximity(0.80),  # VERY_CLOSE
        }
        assert Proximity.FAR        in buckets
        assert Proximity.MEDIUM     in buckets
        assert Proximity.CLOSE      in buckets
        assert Proximity.VERY_CLOSE in buckets


# ===========================================================================
# GROUP 3 — Frame skip / cache logic (no model required)
# ===========================================================================

class TestFrameSkipLogic:
    """
    PASS CONDITIONS 7 and 8:
    Inference must only run every depth_update_interval frames.
    The cached result must be returned on skipped frames.
    """

    def test_update_interval_readable(self, config):
        from assistive_navigation.depth.depth_estimator import DepthEstimator
        est = DepthEstimator(config)
        assert est.update_interval > 0

    def test_frame_counter_increments(self, loaded_estimator):
        """frame_counter increments every process_frame() call."""
        before = loaded_estimator.frame_counter
        loaded_estimator.process_frame(make_blank_frame())
        assert loaded_estimator.frame_counter == before + 1

    def test_inference_count_does_not_run_every_frame(self, loaded_estimator):
        """
        PASS CONDITION 7:
        Inference count must NOT increase on every call.
        If interval=5, after 5 additional calls, inference_count rises by 1.
        """
        interval = loaded_estimator.update_interval

        # Force an inference first to establish baseline
        loaded_estimator.process_frame(make_blank_frame(), force=True)
        baseline = loaded_estimator.inference_count

        # Run (interval - 1) more frames — inference should NOT run again
        for _ in range(interval - 1):
            loaded_estimator.process_frame(make_blank_frame())

        assert loaded_estimator.inference_count == baseline, (
            f"Expected inference_count={baseline} after {interval-1} frames, "
            f"got {loaded_estimator.inference_count}. "
            f"Inference ran too often."
        )

    def test_force_flag_runs_inference_immediately(self, loaded_estimator):
        """force=True must run inference even on a skipped frame."""
        before = loaded_estimator.inference_count
        loaded_estimator.process_frame(make_blank_frame(), force=True)
        assert loaded_estimator.inference_count == before + 1

    def test_cached_map_returned_between_updates(self, loaded_estimator):
        """
        PASS CONDITION 8:
        The depth map returned on a skipped frame must be identical to the
        cached map from the last real inference.
        """
        # Force a fresh inference to set the cache
        loaded_estimator.process_frame(make_blank_frame(), force=True)
        cached = loaded_estimator._cached_depth_map.copy()

        # Next call (should be a skip unless interval=1)
        if loaded_estimator.update_interval > 1:
            result = loaded_estimator.process_frame(make_blank_frame())
            if result is not None:
                assert np.array_equal(result, cached), (
                    "Returned depth map on skipped frame differs from cache."
                )

    def test_depth_frame_age_resets_after_inference(self, loaded_estimator):
        """depth_frame_age must be 0 immediately after an inference runs."""
        loaded_estimator.process_frame(make_blank_frame(), force=True)
        assert loaded_estimator.depth_frame_age == 0

    def test_depth_frame_age_increases_on_skipped_frames(self, loaded_estimator):
        """depth_frame_age must increment on frames where inference was skipped."""
        loaded_estimator.process_frame(make_blank_frame(), force=True)
        age_before = loaded_estimator.depth_frame_age  # 0

        # At least one skipped frame
        if loaded_estimator.update_interval > 1:
            loaded_estimator.process_frame(make_blank_frame())
            assert loaded_estimator.depth_frame_age > age_before


# ===========================================================================
# GROUP 4 — Model tests (model must be loaded, no camera)
# ===========================================================================

class TestDepthModelLoaded:
    """Tests requiring the loaded depth model but not a live camera."""

    def test_is_loaded_true_after_load(self, loaded_estimator):
        """PASS CONDITION 1: is_loaded must be True after load()."""
        assert loaded_estimator.is_loaded is True

    def test_process_frame_returns_array(self, loaded_estimator):
        """PASS CONDITION 2 (part a): process_frame must return a numpy array."""
        frame = make_random_frame()
        result = loaded_estimator.process_frame(frame, force=True)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_depth_map_is_2d(self, loaded_estimator):
        """PASS CONDITION 2 (part b): depth map must be 2D (H, W)."""
        frame = make_random_frame()
        result = loaded_estimator.process_frame(frame, force=True)
        assert result is not None
        assert result.ndim == 2, (
            f"Expected 2D depth map, got shape {result.shape}"
        )

    def test_depth_map_matches_frame_size(self, loaded_estimator):
        """PASS CONDITION 2 (part c): depth map must match input frame dimensions."""
        frame = make_random_frame(h=480, w=640)
        result = loaded_estimator.process_frame(frame, force=True)
        assert result is not None
        assert result.shape == (480, 640), (
            f"Expected (480, 640), got {result.shape}"
        )

    def test_depth_values_in_0_to_1_range(self, loaded_estimator):
        """PASS CONDITION 3: all depth values must be in [0.0, 1.0]."""
        frame = make_random_frame()
        result = loaded_estimator.process_frame(frame, force=True)
        assert result is not None
        assert result.min() >= 0.0, f"Min value {result.min()} < 0"
        assert result.max() <= 1.0, f"Max value {result.max()} > 1"

    def test_depth_dtype_float32(self, loaded_estimator):
        """Depth map must be float32."""
        frame = make_random_frame()
        result = loaded_estimator.process_frame(frame, force=True)
        assert result is not None
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_depth_map_not_all_zeros_on_random_frame(self, loaded_estimator):
        """A random frame should not produce an all-zero depth map."""
        frame = make_random_frame()
        result = loaded_estimator.process_frame(frame, force=True)
        assert result is not None
        assert result.max() > 0.0, "Depth map is all zeros on random input."

    def test_blank_frame_handled_without_crash(self, loaded_estimator):
        """
        PASS CONDITION 4:
        A blank (all-zero) frame must not crash — it should return a valid
        (possibly all-zero) depth map.
        """
        frame = make_blank_frame()
        try:
            result = loaded_estimator.process_frame(frame, force=True)
            # Result may be None or all zeros — neither is an error
            if result is not None:
                assert result.ndim == 2
        except Exception as e:
            pytest.fail(f"Blank frame caused an unexpected exception: {e}")

    def test_gradient_frame_has_depth_variation(self, loaded_estimator):
        """
        A gradient frame (horizontal brightness variation) should produce
        some variation in the depth map (not perfectly flat).
        This is a sanity check — the model should respond to scene content.
        """
        frame = make_gradient_frame()
        result = loaded_estimator.process_frame(frame, force=True)
        assert result is not None
        depth_range = result.max() - result.min()
        assert depth_range > 0.01, (
            f"Depth map on gradient frame has very low variation ({depth_range:.4f}). "
            f"Model may not be responding to scene content."
        )

    def test_latency_is_measured(self, loaded_estimator):
        """
        PASS CONDITION 9:
        last_latency_ms must be positive after an inference run.
        We report the actual value — no assertion on speed.
        """
        frame = make_random_frame()
        loaded_estimator.process_frame(frame, force=True)
        lat = loaded_estimator.last_latency_ms
        assert lat > 0, "last_latency_ms must be positive after inference"
        print(f"\n  Depth inference latency (random frame): {lat:.1f} ms")
        print(f"  Input size: {loaded_estimator.input_size}px")
        print(f"  Performance target: < 500ms for interval-based usage")
        if lat < 200:
            print(f"  Assessment: FAST")
        elif lat < 500:
            print(f"  Assessment: ACCEPTABLE for interval-based usage")
        else:
            print(f"  Assessment: SLOW — consider reducing input_size")

    def test_average_latency_computed_after_multiple_inferences(self, loaded_estimator):
        """average_latency_ms must be positive after multiple inferences."""
        for _ in range(3):
            loaded_estimator.process_frame(make_random_frame(), force=True)
        assert loaded_estimator.average_latency_ms > 0

    def test_inference_count_increments(self, loaded_estimator):
        """inference_count increments each time inference actually runs."""
        before = loaded_estimator.inference_count
        loaded_estimator.process_frame(make_random_frame(), force=True)
        assert loaded_estimator.inference_count == before + 1

    def test_has_depth_map_after_inference(self, loaded_estimator):
        """has_depth_map must be True after at least one inference."""
        loaded_estimator.process_frame(make_random_frame(), force=True)
        assert loaded_estimator.has_depth_map is True


# ===========================================================================
# GROUP 5 — Proximity at bbox tests
# ===========================================================================

class TestProximityAtBbox:
    """
    PASS CONDITION 5:
    get_proximity_at_bbox() must return sensible results for
    bboxes placed in high-depth and low-depth regions of a synthetic map.
    """

    def _make_synthetic_depth(self, h=480, w=640):
        """
        Synthetic depth map:
          Left half  (x < 320): depth = 0.85 (VERY_CLOSE)
          Right half (x >= 320): depth = 0.20 (FAR)
        """
        d = np.zeros((h, w), dtype=np.float32)
        d[:, :w // 2] = 0.85   # left = very close
        d[:, w // 2:] = 0.20   # right = far
        return d

    def test_high_depth_region_returns_very_close_or_close(self, config):
        """Bbox in a high-depth region must return VERY_CLOSE or CLOSE."""
        from assistive_navigation.depth.depth_estimator import (
            DepthEstimator, Proximity
        )
        est = DepthEstimator(config)
        depth = self._make_synthetic_depth()
        # Bbox fully in left half (depth=0.85)
        result = est.get_proximity_at_bbox((50, 100, 250, 380), depth_map=depth)
        assert result in (Proximity.VERY_CLOSE, Proximity.CLOSE), (
            f"Expected VERY_CLOSE or CLOSE for high-depth region, got {result}"
        )

    def test_low_depth_region_returns_far_or_medium(self, config):
        """Bbox in a low-depth region must return FAR or MEDIUM."""
        from assistive_navigation.depth.depth_estimator import (
            DepthEstimator, Proximity
        )
        est = DepthEstimator(config)
        depth = self._make_synthetic_depth()
        # Bbox fully in right half (depth=0.20)
        result = est.get_proximity_at_bbox((380, 100, 590, 380), depth_map=depth)
        assert result in (Proximity.FAR, Proximity.MEDIUM), (
            f"Expected FAR or MEDIUM for low-depth region, got {result}"
        )

    def test_no_depth_map_returns_unknown(self, config):
        from assistive_navigation.depth.depth_estimator import (
            DepthEstimator, Proximity
        )
        est = DepthEstimator(config)  # unloaded, no cache
        result = est.get_proximity_at_bbox((50, 50, 200, 200))
        assert result == Proximity.UNKNOWN

    def test_out_of_bounds_bbox_returns_unknown(self, config):
        from assistive_navigation.depth.depth_estimator import (
            DepthEstimator, Proximity
        )
        est = DepthEstimator(config)
        depth = np.ones((480, 640), dtype=np.float32) * 0.5
        # Completely outside frame
        result = est.get_proximity_at_bbox((700, 600, 900, 900), depth_map=depth)
        assert result in (Proximity.UNKNOWN, Proximity.MEDIUM), (
            "Out-of-bounds bbox should return UNKNOWN or gracefully clamp"
        )

    def test_invalid_bbox_zero_area_returns_unknown(self, config):
        from assistive_navigation.depth.depth_estimator import (
            DepthEstimator, Proximity
        )
        est = DepthEstimator(config)
        depth = np.ones((480, 640), dtype=np.float32) * 0.5
        # x1 == x2, y1 == y2 (zero area)
        result = est.get_proximity_at_bbox((100, 100, 100, 100), depth_map=depth)
        assert result == Proximity.UNKNOWN

    def test_returns_string(self, config):
        from assistive_navigation.depth.depth_estimator import DepthEstimator
        est = DepthEstimator(config)
        depth = np.full((480, 640), 0.6, dtype=np.float32)
        result = est.get_proximity_at_bbox((100, 100, 300, 300), depth_map=depth)
        assert isinstance(result, str)


# ===========================================================================
# GROUP 6 — Visualisation tests
# ===========================================================================

class TestDepthVisualisation:
    """PASS CONDITION 10: colour map generation."""

    def test_colourmap_returns_array(self, loaded_estimator):
        """get_colourmap() must return a BGR uint8 numpy array."""
        loaded_estimator.process_frame(make_random_frame(), force=True)
        coloured = loaded_estimator.get_colourmap()
        assert coloured is not None
        assert isinstance(coloured, np.ndarray)

    def test_colourmap_is_3_channel(self, loaded_estimator):
        """Colour map output must have 3 channels (BGR)."""
        loaded_estimator.process_frame(make_random_frame(), force=True)
        coloured = loaded_estimator.get_colourmap()
        assert coloured is not None
        assert coloured.ndim == 3
        assert coloured.shape[2] == 3

    def test_colourmap_dtype_uint8(self, loaded_estimator):
        loaded_estimator.process_frame(make_random_frame(), force=True)
        coloured = loaded_estimator.get_colourmap()
        assert coloured is not None
        assert coloured.dtype == np.uint8

    def test_colourmap_none_without_cache(self, fresh_estimator):
        """get_colourmap() must return None if no depth map has been computed."""
        result = fresh_estimator.get_colourmap()
        assert result is None

    def test_colourmap_accepts_explicit_depth_map(self, config):
        """get_colourmap() must accept an explicit depth map argument."""
        from assistive_navigation.depth.depth_estimator import DepthEstimator
        est = DepthEstimator(config)
        synthetic = np.linspace(0, 1, 480 * 640).reshape(480, 640).astype(np.float32)
        coloured = est.get_colourmap(depth_map=synthetic)
        assert coloured is not None
        assert coloured.shape == (480, 640, 3)


# ===========================================================================
# GROUP 7 — Hardware tests (require webcam)
# ===========================================================================

class TestDepthWithWebcam:
    """Hardware tests — require webcam. Marked requires_camera."""

    requires_camera = pytest.mark.requires_camera

    @requires_camera
    def test_depth_map_from_real_frame(self, loaded_estimator, config):
        """
        PASS CONDITION 2 + 3 (hardware):
        Run depth estimation on a real camera frame.
        Verify shape, value range, and that the map is not all zeros.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        try:
            cam = CameraCapture(config)
            cam.open()
            frame = cam.read()
            cam.release()
        except CameraError as e:
            pytest.skip(f"Webcam not available: {e}")

        if frame is None:
            pytest.skip("Webcam returned None frame.")

        result = loaded_estimator.process_frame(frame, force=True)

        assert result is not None
        assert result.ndim == 2
        assert result.shape == (frame.shape[0], frame.shape[1])
        assert result.dtype == np.float32
        assert 0.0 <= result.min()
        assert result.max() <= 1.0
        assert result.max() > 0.01, "Depth map is near-zero on real frame."

        # Print benchmark
        print(f"\n  === DEPTH ESTIMATION — REAL FRAME ===")
        print(f"  Frame size          : {frame.shape[1]}x{frame.shape[0]}")
        print(f"  Depth map size      : {result.shape[1]}x{result.shape[0]}")
        print(f"  Depth value range   : [{result.min():.3f}, {result.max():.3f}]")
        print(f"  Inference latency   : {loaded_estimator.last_latency_ms:.1f} ms")
        print(f"  Input size (config) : {loaded_estimator.input_size}px")

    @requires_camera
    def test_proximity_on_real_frame(self, loaded_estimator, config):
        """
        PASS CONDITION 5 (hardware):
        Run proximity estimation on the centre of a real camera frame.
        Report the observed proximity bucket.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        try:
            cam = CameraCapture(config)
            cam.open()
            frame = cam.read()
            cam.release()
        except CameraError as e:
            pytest.skip(f"Webcam not available: {e}")

        if frame is None:
            pytest.skip("Webcam returned None frame.")

        depth_map = loaded_estimator.process_frame(frame, force=True)
        if depth_map is None:
            pytest.skip("Depth map was None.")

        h, w = frame.shape[:2]
        # Sample centre region
        cx = w // 4
        cy = h // 4
        centre_bbox = (cx, cy, w - cx, h - cy)
        prox = loaded_estimator.get_proximity_at_bbox(centre_bbox, depth_map)

        print(f"\n  === PROXIMITY ON REAL FRAME ===")
        print(f"  Bbox tested         : {centre_bbox}")
        print(f"  Proximity result    : {prox}")
        print(f"  Note: result is RELATIVE PROXIMITY, not metric distance")
        print(f"  Depth mean (centre) : {depth_map[cy:h-cy, cx:w-cx].mean():.3f}")

        from assistive_navigation.depth.depth_estimator import Proximity
        valid = {
            Proximity.FAR, Proximity.MEDIUM,
            Proximity.CLOSE, Proximity.VERY_CLOSE, Proximity.UNKNOWN
        }
        assert prox in valid, f"Unexpected proximity value: {prox}"

    @requires_camera
    def test_depth_benchmark_10_frames(self, loaded_estimator, config):
        """
        PASS CONDITION 9 (sustained hardware):
        Measure depth inference latency over 10 real frames.
        Report actual numbers — no assertion on speed.
        """
        import time
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        try:
            cam = CameraCapture(config)
            cam.open()
        except CameraError as e:
            pytest.skip(f"Webcam not available: {e}")

        latencies = []
        try:
            for _ in range(10):
                frame = cam.read()
                if frame is None:
                    continue
                loaded_estimator.process_frame(frame, force=True)
                latencies.append(loaded_estimator.last_latency_ms)
        finally:
            cam.release()

        if not latencies:
            pytest.skip("No frames captured.")

        avg = sum(latencies) / len(latencies)
        mn  = min(latencies)
        mx  = max(latencies)

        print(f"\n  === DEPTH LATENCY BENCHMARK (10 frames) ===")
        print(f"  Input size          : {loaded_estimator.input_size}px")
        print(f"  Average latency     : {avg:.1f} ms")
        print(f"  Min latency         : {mn:.1f} ms")
        print(f"  Max latency         : {mx:.1f} ms")
        print(f"  Target (guideline)  : < 500ms (interval-based usage)")
        print(f"  Update interval     : every {loaded_estimator.update_interval} frames")
        if avg < 200:
            print(f"  Assessment: FAST — suitable even for shorter intervals")
        elif avg < 500:
            print(f"  Assessment: ACCEPTABLE — interval-based usage works well")
        else:
            print(f"  Assessment: SLOW — use larger depth_update_interval")

        assert len(latencies) > 0
        assert avg > 0
