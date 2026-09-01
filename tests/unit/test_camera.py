"""
Unit Tests - Camera Capture Module
=====================================
Phase 2 PASS conditions tested here:

  [1] Camera opens without error
  [2] Frames are non-null with the expected shape (H, W, 3)
  [3] Actual FPS is measured and exceeds 15 FPS
  [4] Camera releases cleanly (is_open = False after release)
  [5] Invalid device index raises CameraError (not a crash)
  [6] All tests pass

Tests are split into two groups:
  - Hardware tests  : require a real webcam (marked requires_camera)
  - Logic tests     : no webcam needed, always run

How to run:
    # Run all tests (hardware tests skipped if no webcam):
    .venv/Scripts/python.exe -m pytest tests/unit/test_camera.py -v

    # Run only hardware tests (webcam must be connected):
    .venv/Scripts/python.exe -m pytest tests/unit/test_camera.py -v -m requires_camera

    # Run only logic tests (no webcam needed):
    .venv/Scripts/python.exe -m pytest tests/unit/test_camera.py -v -m "not requires_camera"
"""

import time
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Pytest marker registration
# Marks tests that require a physical webcam.
# If pytest.ini / pyproject.toml does not register this marker,
# pytest will warn but still run correctly.
# ---------------------------------------------------------------------------
requires_camera = pytest.mark.requires_camera


# ---------------------------------------------------------------------------
# Shared config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    """
    Load and return the real project config.
    Uses the actual config/config.yaml so tests reflect real settings.
    Scope=module: loaded once per test file (not once per test).
    """
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


@pytest.fixture(scope="module")
def invalid_device_config(config):
    """
    A config copy that points to device 99 — a camera that does not exist.
    Used to test graceful failure handling.
    """
    import copy
    bad_cfg = copy.deepcopy(config)
    bad_cfg["camera"]["device_id"] = 99
    return bad_cfg


# ===========================================================================
# GROUP 1 — Logic tests (no webcam required)
# ===========================================================================

class TestCameraErrorClass:
    """Tests for the CameraError exception — no hardware needed."""

    def test_camera_error_is_importable(self):
        """CameraError must be importable from the camera module."""
        from assistive_navigation.camera.capture import CameraError
        assert CameraError is not None

    def test_camera_error_is_exception(self):
        """CameraError must inherit from Exception."""
        from assistive_navigation.camera.capture import CameraError
        assert issubclass(CameraError, Exception)

    def test_camera_error_has_message(self):
        """CameraError must carry a message string."""
        from assistive_navigation.camera.capture import CameraError
        err = CameraError("test message")
        assert str(err) == "test message"


class TestCameraCaptureClass:
    """Tests for CameraCapture class structure — no hardware needed."""

    def test_camera_capture_is_importable(self):
        """CameraCapture must be importable."""
        from assistive_navigation.camera.capture import CameraCapture
        assert CameraCapture is not None

    def test_camera_capture_reads_config_values(self, config):
        """
        CameraCapture must read device_id, width, height, max_fps
        from config without raising an error.
        The camera is NOT opened here.
        """
        from assistive_navigation.camera.capture import CameraCapture
        cam = CameraCapture(config)
        assert cam.device_id == config["camera"]["device_id"]
        assert cam.requested_resolution == (
            config["camera"]["width"],
            config["camera"]["height"]
        )

    def test_read_before_open_raises_camera_error(self, config):
        """
        Calling read() before open() must raise CameraError,
        not crash with an AttributeError or NoneType error.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        cam = CameraCapture(config)
        with pytest.raises(CameraError, match="read\\(\\) called before open"):
            cam.read()

    def test_is_open_false_before_open(self, config):
        """is_open must be False before open() is called."""
        from assistive_navigation.camera.capture import CameraCapture
        cam = CameraCapture(config)
        assert cam.is_open is False

    def test_frame_count_zero_before_open(self, config):
        """frame_count must be 0 before any frames are read."""
        from assistive_navigation.camera.capture import CameraCapture
        cam = CameraCapture(config)
        assert cam.frame_count == 0

    def test_release_before_open_is_safe(self, config):
        """
        Calling release() before open() must be safe — no exception.
        release() should be idempotent.
        """
        from assistive_navigation.camera.capture import CameraCapture
        cam = CameraCapture(config)
        cam.release()  # should not raise
        assert cam.is_open is False

    def test_repr_contains_device_id(self, config):
        """repr() should include the device ID for easy debugging."""
        from assistive_navigation.camera.capture import CameraCapture
        cam = CameraCapture(config)
        r = repr(cam)
        assert str(config["camera"]["device_id"]) in r


# ===========================================================================
# GROUP 2 — Hardware tests (require a real webcam)
# These are marked @requires_camera. If the webcam is unavailable,
# they will be skipped automatically (see autoskip fixture below).
# ===========================================================================

@pytest.fixture(scope="module")
def camera_or_skip(config):
    """
    Opens a CameraCapture for the entire test module.
    If the camera cannot be opened, all hardware tests are skipped
    with a clear message — they are not failed.

    scope=module: the camera is opened once and shared across all
    hardware tests in this file (faster and avoids repeated open/close).
    The camera is released after all tests in the module finish.
    """
    from assistive_navigation.camera.capture import CameraCapture, CameraError
    cam = CameraCapture(config)
    try:
        cam.open()
    except CameraError as e:
        pytest.skip(f"Webcam not available — hardware tests skipped: {e}")
    yield cam
    cam.release()


class TestCameraHardware:
    """
    Hardware tests — require a real webcam (device 0).
    All tests in this class share one open CameraCapture instance.
    """

    # -------------------------------------------------------------------
    # PASS CONDITION 1: Camera opens without error
    # -------------------------------------------------------------------
    @requires_camera
    def test_camera_opens(self, camera_or_skip):
        """
        PASS CONDITION 1:
        Camera must open without raising an exception.
        is_open must be True after open().
        """
        cam = camera_or_skip
        assert cam.is_open is True, (
            "Camera is_open should be True after open(). "
            "Check that the webcam is not in use by another app."
        )

    # -------------------------------------------------------------------
    # PASS CONDITION 2: Frame shape
    # -------------------------------------------------------------------
    @requires_camera
    def test_frame_is_not_none(self, camera_or_skip):
        """read() must return a non-None value."""
        cam = camera_or_skip
        frame = cam.read()
        assert frame is not None, (
            "read() returned None. Camera may have disconnected."
        )

    @requires_camera
    def test_frame_is_numpy_array(self, camera_or_skip):
        """Frame must be a NumPy array."""
        cam = camera_or_skip
        frame = cam.read()
        assert isinstance(frame, np.ndarray), (
            f"Expected np.ndarray, got {type(frame)}"
        )

    @requires_camera
    def test_frame_has_3_dimensions(self, camera_or_skip):
        """
        PASS CONDITION 2 (part a):
        Frame must have exactly 3 dimensions: (height, width, channels).
        """
        cam = camera_or_skip
        frame = cam.read()
        assert frame is not None
        assert frame.ndim == 3, (
            f"Expected 3D array (H, W, C), got shape {frame.shape}"
        )

    @requires_camera
    def test_frame_has_3_colour_channels(self, camera_or_skip):
        """
        PASS CONDITION 2 (part b):
        Frame must have 3 colour channels (BGR — OpenCV default).
        """
        cam = camera_or_skip
        frame = cam.read()
        assert frame is not None
        assert frame.shape[2] == 3, (
            f"Expected 3 colour channels (BGR), got {frame.shape[2]}"
        )

    @requires_camera
    def test_frame_dimensions_are_positive(self, camera_or_skip):
        """
        PASS CONDITION 2 (part c):
        Frame height and width must both be > 0.
        """
        cam = camera_or_skip
        frame = cam.read()
        assert frame is not None
        h, w, _ = frame.shape
        assert h > 0, f"Frame height is 0"
        assert w > 0, f"Frame width is 0"

    @requires_camera
    def test_frame_resolution_matches_actual(self, camera_or_skip):
        """
        PASS CONDITION 2 (part d):
        Frame dimensions must match the actual resolution
        reported by cam.actual_resolution.
        """
        cam = camera_or_skip
        frame = cam.read()
        assert frame is not None
        h, w, _ = frame.shape
        exp_w, exp_h = cam.actual_resolution
        assert h == exp_h, (
            f"Frame height {h} != actual_height {exp_h}"
        )
        assert w == exp_w, (
            f"Frame width {w} != actual_width {exp_w}"
        )

    @requires_camera
    def test_frame_dtype_is_uint8(self, camera_or_skip):
        """Frame pixel values must be uint8 (0-255 range)."""
        cam = camera_or_skip
        frame = cam.read()
        assert frame is not None
        assert frame.dtype == np.uint8, (
            f"Expected dtype uint8, got {frame.dtype}"
        )

    @requires_camera
    def test_frame_is_not_all_zeros(self, camera_or_skip):
        """
        Frame must not be a black/empty array.
        An all-zero frame usually means the driver returned a blank frame.
        """
        cam = camera_or_skip
        frame = cam.read()
        assert frame is not None
        assert frame.max() > 0, (
            "Frame is all zeros — camera may be covered or driver is returning "
            "blank frames."
        )

    # -------------------------------------------------------------------
    # PASS CONDITION 3: Measured FPS
    # -------------------------------------------------------------------
    @requires_camera
    def test_fps_above_minimum(self, camera_or_skip):
        """
        PASS CONDITION 3:
        Raw capture speed must exceed 15 FPS.

        This uses the ACTUAL measured result — no artificial weakening.
        15 FPS is the minimum threshold for this phase.
        The target on this hardware is 20-30+ FPS.

        What we measure:
          measure_capture_fps(30) captures 30 frames as fast as possible
          (bypassing the max_fps cap) and returns frames/second.

        Why 15 FPS minimum?
          Human walking pace changes slowly enough that 15 FPS is
          sufficient for auditory navigation alerts. Below 15 FPS,
          the delay between scene change and alert would become
          noticeable and potentially unsafe.
        """
        cam = camera_or_skip
        MIN_ACCEPTABLE_FPS = 15.0

        fps = cam.measure_capture_fps(n_frames=30)

        print(f"\n  Measured capture FPS: {fps:.2f}")
        print(f"  Minimum required:     {MIN_ACCEPTABLE_FPS:.1f}")
        print(f"  Result:               {'PASS' if fps >= MIN_ACCEPTABLE_FPS else 'FAIL'}")

        assert fps >= MIN_ACCEPTABLE_FPS, (
            f"Capture FPS {fps:.1f} is below the minimum {MIN_ACCEPTABLE_FPS} FPS. "
            f"This may indicate: slow camera driver, USB bandwidth issue, "
            f"or system resource constraint. Check camera settings."
        )

    @requires_camera
    def test_smoothed_fps_updates_after_reads(self, camera_or_skip):
        """
        get_smoothed_fps() must return a non-zero value after
        multiple frames have been read.
        """
        cam = camera_or_skip
        # Read enough frames to populate the smoothed FPS window
        for _ in range(5):
            cam.read()
        fps = cam.get_smoothed_fps()
        assert fps > 0, (
            "get_smoothed_fps() returned 0.0 after 5 reads. "
            "Timestamp tracking may be broken."
        )

    @requires_camera
    def test_multiple_frames_are_captured(self, camera_or_skip):
        """Reading 10 frames must all be non-null."""
        cam = camera_or_skip
        failures = []
        for i in range(10):
            frame = cam.read()
            if frame is None:
                failures.append(i)
        assert len(failures) == 0, (
            f"Frames {failures} returned None out of 10 reads."
        )

    @requires_camera
    def test_frame_count_increments(self, camera_or_skip):
        """
        frame_count must increase after each successful read().
        """
        cam = camera_or_skip
        count_before = cam.frame_count
        cam.read()
        cam.read()
        cam.read()
        assert cam.frame_count == count_before + 3, (
            f"Expected frame_count to increase by 3, "
            f"before={count_before}, after={cam.frame_count}"
        )

    # -------------------------------------------------------------------
    # PASS CONDITION 4: Camera releases cleanly
    # -------------------------------------------------------------------
    @requires_camera
    def test_camera_releases_cleanly(self, config):
        """
        PASS CONDITION 4:
        After release(), is_open must be False.
        Opening and releasing a second camera instance to test isolation.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        cam = CameraCapture(config)
        try:
            cam.open()
        except CameraError:
            pytest.skip("Webcam not available for release test.")

        assert cam.is_open is True
        cam.release()
        assert cam.is_open is False, (
            "is_open should be False after release()."
        )

    @requires_camera
    def test_context_manager_releases_on_normal_exit(self, config):
        """
        The 'with' statement must release the camera after the block ends.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        try:
            with CameraCapture(config) as cam:
                assert cam.is_open is True
                _ = cam.read()
            # After 'with' block, camera must be closed
            assert cam.is_open is False, (
                "Camera still open after 'with' block exited."
            )
        except CameraError:
            pytest.skip("Webcam not available for context manager test.")

    @requires_camera
    def test_context_manager_releases_on_exception(self, config):
        """
        The 'with' statement must release the camera even if an
        exception occurs inside the block.
        This is the main advantage of the context manager pattern.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        cam_ref = None
        try:
            with pytest.raises(ValueError):
                with CameraCapture(config) as cam:
                    cam_ref = cam
                    assert cam.is_open is True
                    raise ValueError("Simulated error inside 'with' block")
        except CameraError:
            pytest.skip("Webcam not available for exception release test.")

        if cam_ref is not None:
            assert cam_ref.is_open is False, (
                "Camera was not released after exception inside 'with' block."
            )

    @requires_camera
    def test_release_is_idempotent(self, config):
        """
        Calling release() twice must not raise an error.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        try:
            cam = CameraCapture(config)
            cam.open()
            cam.release()
            cam.release()  # second call — must be safe
        except CameraError:
            pytest.skip("Webcam not available for idempotent release test.")


# ===========================================================================
# GROUP 3 — Invalid device test (no working camera needed, but needs
#            OpenCV to be installed — always runs)
# ===========================================================================

class TestCameraInvalidDevice:
    """
    PASS CONDITION 5:
    Attempting to open a non-existent camera device must raise CameraError
    with a helpful message, not crash the process.
    """

    def test_invalid_device_raises_camera_error(self, invalid_device_config):
        """
        PASS CONDITION 5:
        Device 99 does not exist. open() must raise CameraError,
        not a generic OpenCV error, AttributeError, or segfault.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        cam = CameraCapture(invalid_device_config)
        with pytest.raises(CameraError):
            cam.open()

    def test_invalid_device_leaves_camera_closed(self, invalid_device_config):
        """
        After a failed open(), is_open must still be False.
        The failed open must not leave the camera in an inconsistent state.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        cam = CameraCapture(invalid_device_config)
        try:
            cam.open()
        except CameraError:
            pass  # expected
        assert cam.is_open is False, (
            "is_open should remain False after a failed open()."
        )

    def test_invalid_device_error_message_is_helpful(self, invalid_device_config):
        """
        The CameraError message must mention the device ID.
        A helpful error message reduces debugging time.
        """
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        cam = CameraCapture(invalid_device_config)
        with pytest.raises(CameraError) as exc_info:
            cam.open()
        error_message = str(exc_info.value)
        assert "99" in error_message, (
            f"Error message should mention device 99, got: {error_message!r}"
        )
