"""
Camera Capture Module
=====================
Handles webcam frame capture using OpenCV VideoCapture.

Phase: 2
Status: IMPLEMENTED

Usage (simple):
    from assistive_navigation.utils.config_loader import load_config
    from assistive_navigation.camera.capture import CameraCapture

    config = load_config()
    with CameraCapture(config) as cam:
        frame = cam.read()
        if frame is not None:
            # frame is a NumPy array, shape (height, width, 3), BGR colour order

Usage (benchmark):
    with CameraCapture(config) as cam:
        fps = cam.measure_capture_fps(n_frames=30)
        print(f"Capture FPS: {fps:.1f}")

Design decisions (documented for transparency):
  - Synchronous capture in Phase 2. No background thread yet.
    Threading is a Phase 13 optimisation once correctness is proven.
  - Class-based with context manager support so the camera is always
    released even if an exception occurs.
  - CameraError is a specific exception so callers can handle camera
    failures separately from generic Python errors.
  - Requested resolution vs. actual resolution are both stored and logged.
    The camera driver may not honour the requested size.
  - FPS is measured two ways:
      measure_capture_fps() — a clean benchmark (Phase 2 PASS check)
      get_smoothed_fps()    — running average for live display (Phase 13)
"""

import time
import logging
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class CameraError(Exception):
    """
    Raised when the camera cannot be opened or produces an invalid frame.

    Using a specific exception (not a bare RuntimeError) lets callers
    catch camera problems without accidentally swallowing unrelated errors.

    Example:
        try:
            cam = CameraCapture(config)
            cam.open()
        except CameraError as e:
            print(f"Camera problem: {e}")
            # handle gracefully — do not crash
    """
    pass


# ---------------------------------------------------------------------------
# CameraCapture class
# ---------------------------------------------------------------------------

class CameraCapture:
    """
    Opens a webcam, reads frames, measures FPS, and releases resources cleanly.

    All settings come from the 'camera' section of config.yaml:
        device_id  : 0 = first camera (usually the laptop webcam)
        width      : requested frame width in pixels
        height     : requested frame height in pixels
        max_fps    : cap on capture rate (reduces CPU load)
        buffer_size: number of frames OpenCV buffers internally

    Example:
        with CameraCapture(config) as cam:
            for _ in range(10):
                frame = cam.read()

    Args:
        config (dict): The full configuration dictionary from config.yaml.
    """

    def __init__(self, config: dict) -> None:
        cam_cfg = config.get("camera", {})

        # --- Settings read from config (with safe defaults) ---
        self._device_id: int = cam_cfg.get("device_id", 0)
        self._requested_width: int = cam_cfg.get("width", 640)
        self._requested_height: int = cam_cfg.get("height", 480)
        self._max_fps: float = float(cam_cfg.get("max_fps", 30))
        self._buffer_size: int = cam_cfg.get("buffer_size", 2)

        # Minimum gap between frames to enforce the max_fps cap.
        # e.g. max_fps=30 → min gap = 1/30 ≈ 0.033 s
        self._min_frame_gap: float = 1.0 / self._max_fps

        # --- Internal state ---
        self._cap: Optional[cv2.VideoCapture] = None  # OpenCV capture object
        self._is_open: bool = False

        # Actual resolution reported by the driver after opening
        self._actual_width: int = 0
        self._actual_height: int = 0

        # Frame counter and timing
        self._frame_count: int = 0
        self._last_frame_time: float = 0.0

        # Smoothed FPS: keep timestamps of the last 30 frames
        # Used by get_smoothed_fps() for a live rolling average
        self._frame_timestamps: deque = deque(maxlen=30)

        logger.debug(
            "CameraCapture created — device=%d, requested=%dx%d, max_fps=%g",
            self._device_id, self._requested_width, self._requested_height,
            self._max_fps
        )

    # -----------------------------------------------------------------------
    # Context manager support — guarantees the camera is always released
    # -----------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        """Open the camera when entering a 'with' block."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Release the camera when leaving a 'with' block.
        Always releases, even if an exception occurred.
        Returns False so exceptions are NOT suppressed.
        """
        self.release()
        return False  # do not suppress exceptions

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def open(self) -> None:
        """
        Open the webcam.

        What this does:
          1. Creates an OpenCV VideoCapture for the configured device_id.
          2. Requests the configured resolution from the driver.
          3. Sets the internal frame buffer size.
          4. Reads back the actual resolution (driver may differ from request).
          5. Logs a warning if the actual resolution differs from requested.
          6. Captures and discards one frame to "warm up" the camera pipeline.

        Raises:
            CameraError: If the device cannot be opened, or the first
                         warm-up frame is invalid.
        """
        if self._is_open:
            logger.warning("open() called but camera is already open.")
            return

        logger.info("Opening camera device %d ...", self._device_id)

        # CAP_DSHOW = DirectShow backend on Windows.
        # More reliable than the default MSMF backend for integrated cameras.
        self._cap = cv2.VideoCapture(self._device_id, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise CameraError(
                f"Cannot open camera device {self._device_id}.\n"
                f"Check that the webcam is connected and not in use by "
                f"another application (e.g. Teams, Zoom, Camera app)."
            )

        # Request resolution and buffer size from the driver.
        # These are hints — the driver may ignore them.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   self._buffer_size)

        # Read back what the driver actually gave us
        self._actual_width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (self._actual_width  != self._requested_width or
                self._actual_height != self._requested_height):
            logger.warning(
                "Camera resolution: requested %dx%d, actual %dx%d. "
                "The driver did not honour the requested size. "
                "Zone thresholds in Phase 8 must use the actual resolution.",
                self._requested_width,  self._requested_height,
                self._actual_width,     self._actual_height
            )
        else:
            logger.info(
                "Camera resolution: %dx%d (as requested)",
                self._actual_width, self._actual_height
            )

        # Warm-up read: discard the first frame.
        # Many webcam drivers produce a dark or corrupt first frame while
        # the sensor adjusts exposure. Discarding it avoids a misleading
        # first detection result.
        ret, warmup_frame = self._cap.read()
        if not ret or warmup_frame is None:
            self._cap.release()
            self._cap = None
            raise CameraError(
                f"Camera device {self._device_id} opened but produced no "
                f"frames. The camera may be blocked or misconfigured."
            )

        self._is_open = True
        self._frame_count = 0
        self._last_frame_time = time.perf_counter()

        logger.info(
            "Camera ready — device=%d, resolution=%dx%d",
            self._device_id, self._actual_width, self._actual_height
        )

    def read(self) -> Optional[np.ndarray]:
        """
        Capture and return one frame from the webcam.

        This is synchronous — it waits for the next frame.
        The frame is a NumPy array of shape (height, width, 3) in BGR
        colour order (the OpenCV default).

        What "BGR" means for a beginner:
          OpenCV stores colours as Blue-Green-Red, not Red-Green-Blue.
          This is a historical OpenCV convention. It does not affect
          navigation — we only need shapes and positions, not colours.

        FPS cap:
          If the camera delivers frames faster than max_fps, this method
          waits the necessary time before reading. This prevents the CPU
          from being fully consumed by the capture loop.

        Returns:
            np.ndarray: A valid frame (height x width x 3, dtype uint8),
                        or None if the camera failed to produce a frame.

        Raises:
            CameraError: If read() is called before open().
        """
        if not self._is_open or self._cap is None:
            raise CameraError(
                "read() called before open(). "
                "Use 'with CameraCapture(config) as cam:' or call open() first."
            )

        # Enforce FPS cap: sleep if we are reading faster than max_fps
        now = time.perf_counter()
        elapsed = now - self._last_frame_time
        if elapsed < self._min_frame_gap:
            time.sleep(self._min_frame_gap - elapsed)

        ret, frame = self._cap.read()

        if not ret or frame is None:
            logger.warning(
                "Camera read() returned no frame (frame %d). "
                "Camera may have disconnected.",
                self._frame_count
            )
            return None

        # Update counters and timestamps
        self._frame_count += 1
        self._last_frame_time = time.perf_counter()
        self._frame_timestamps.append(self._last_frame_time)

        return frame

    def release(self) -> None:
        """
        Release the camera and free all resources.

        Safe to call multiple times — subsequent calls are no-ops.
        Always call this when done, or use the 'with' statement which
        calls it automatically.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(
                "Camera released — %d frames captured during this session.",
                self._frame_count
            )
        self._is_open = False

    def measure_capture_fps(self, n_frames: int = 30) -> float:
        """
        Measure raw capture speed by grabbing N frames as fast as possible.

        This temporarily bypasses the max_fps cap to measure what the
        hardware + driver can actually deliver. It is used for the
        Phase 2 PASS condition check.

        The FPS cap is restored after the measurement.

        Args:
            n_frames: Number of frames to capture for the measurement.
                      30 frames gives a stable result without taking
                      too long (~2 seconds at 15 FPS).

        Returns:
            float: Measured frames per second.

        Raises:
            CameraError: If the camera is not open.
            CameraError: If fewer than n_frames valid frames were captured.
        """
        if not self._is_open or self._cap is None:
            raise CameraError("measure_capture_fps() called before open().")

        logger.info("Measuring capture FPS over %d frames ...", n_frames)

        # Temporarily disable the FPS cap for the benchmark
        saved_gap = self._min_frame_gap
        self._min_frame_gap = 0.0

        frames_captured = 0
        start_time = time.perf_counter()

        for _ in range(n_frames):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                frames_captured += 1

        elapsed = time.perf_counter() - start_time

        # Restore the FPS cap
        self._min_frame_gap = saved_gap

        if frames_captured == 0:
            raise CameraError(
                "measure_capture_fps(): No valid frames received. "
                "Camera may have disconnected during benchmark."
            )

        fps = frames_captured / elapsed
        logger.info(
            "Capture FPS benchmark: %.1f FPS (%d frames in %.3f s)",
            fps, frames_captured, elapsed
        )
        return fps

    def get_smoothed_fps(self) -> float:
        """
        Return a smoothed FPS estimate based on the last N frame timestamps.

        This is a live estimate — it updates each time read() is called.
        Used for the FPS overlay in debug mode (Phase 13).

        Returns:
            float: Estimated FPS (0.0 if fewer than 2 frames captured yet).
        """
        if len(self._frame_timestamps) < 2:
            return 0.0
        elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_timestamps) - 1) / elapsed

    # -----------------------------------------------------------------------
    # Read-only properties
    # -----------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True if the camera is currently open and ready to read frames."""
        return self._is_open

    @property
    def actual_width(self) -> int:
        """Actual frame width reported by the camera driver (pixels)."""
        return self._actual_width

    @property
    def actual_height(self) -> int:
        """Actual frame height reported by the camera driver (pixels)."""
        return self._actual_height

    @property
    def actual_resolution(self) -> Tuple[int, int]:
        """(width, height) as reported by the camera driver."""
        return (self._actual_width, self._actual_height)

    @property
    def requested_resolution(self) -> Tuple[int, int]:
        """(width, height) as requested in config.yaml."""
        return (self._requested_width, self._requested_height)

    @property
    def frame_count(self) -> int:
        """Total number of valid frames read since open() was called."""
        return self._frame_count

    @property
    def device_id(self) -> int:
        """Camera device index as configured."""
        return self._device_id

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return (
            f"CameraCapture(device={self._device_id}, "
            f"resolution={self._actual_width}x{self._actual_height}, "
            f"status={status}, frames={self._frame_count})"
        )
