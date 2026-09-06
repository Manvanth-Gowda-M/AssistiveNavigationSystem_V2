"""
Object Detector Module
======================
Runs YOLO11n inference on a single video frame and returns all detections.

Phase: 3
Status: IMPLEMENTED

What this module does:
    - Loads a YOLO11n model (pretrained on COCO, 80 classes)
    - Accepts a BGR frame from OpenCV (numpy array)
    - Runs inference using the Ultralytics Python API (CPU)
    - Returns a list of Detection objects — one per detected object
    - Measures and records inference latency per frame
    - Logs EVERYTHING it detects, including low-confidence detections
      that will later be filtered out

What this module does NOT do:
    - Does not filter by navigation class list (that is DetectionFilter's job)
    - Does not track objects across frames (Phase 5)
    - Does not estimate depth (Phase 6)
    - Does not produce alerts (Phase 12)

IMPORTANT — On confidence scores:
    A high confidence score does NOT mean the detection is correct.
    Confidence means the model is "sure" about its guess, but its guess
    may still be wrong (e.g. a charger detected as a phone at 0.82 conf).
    This module logs all detections honestly. Do not raise the confidence
    floor to hide false positives — investigate the root cause instead.

Model note:
    This module uses the Ultralytics Python API with PyTorch (.pt weights).
    The model is downloaded automatically from the official Ultralytics hub
    on first use (~6 MB). The path yolo11n.pt is stored in the models/ dir.
    In Phase 14 (optimisation), this will be switched to ONNX Runtime
    for faster CPU inference.

Usage:
    from assistive_navigation.utils.config_loader import load_config
    from assistive_navigation.detection.detector import ObjectDetector

    config = load_config()
    detector = ObjectDetector(config)
    detector.load()

    frame = ...  # numpy array from OpenCV, shape (H, W, 3), BGR

    result = detector.detect(frame)
    for det in result.detections:
        print(det.class_name, det.confidence, det.bbox)
    print(f"Latency: {result.latency_ms:.1f} ms")
"""

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Suppress Ultralytics' own verbose logging during inference.
# It prints progress bars and model info to stdout by default.
# We capture what we need into our own structured log instead.
import os
os.environ.setdefault("YOLO_VERBOSE", "False")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class DetectorError(Exception):
    """
    Raised when the detector cannot be loaded or configured correctly.

    Examples of when this is raised:
      - Model file does not exist and cannot be downloaded
      - Model file is corrupt or not a valid YOLO model
      - detect() called before load()
    """
    pass


# ---------------------------------------------------------------------------
# Data classes — structured detection output
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    Represents a single detected object in one frame.

    All coordinates are provided in two forms:
      - Absolute (pixels): useful for drawing bounding boxes
      - Normalised (0.0–1.0): useful for zone calculations and
        comparisons across different frame resolutions

    Fields:
        class_id    : COCO class ID (0=person, 56=chair, etc.)
        class_name  : Human-readable name (e.g. "person")
        confidence  : Model's confidence in this detection (0.0–1.0)
                      WARNING: high confidence ≠ correct detection.
        bbox        : (x1, y1, x2, y2) in pixels (top-left, bottom-right)
        bbox_norm   : (x1, y1, x2, y2) normalised to frame dimensions
        center_x    : Horizontal centre of bbox, normalised (0.0=left, 1.0=right)
        center_y    : Vertical centre of bbox, normalised (0.0=top, 1.0=bottom)
        area_norm   : Bounding box area as a fraction of total frame area
        frame_width : Actual frame width in pixels (for reference)
        frame_height: Actual frame height in pixels (for reference)
    """
    class_id:     int
    class_name:   str
    confidence:   float
    bbox:         tuple        # (x1, y1, x2, y2) absolute pixels
    bbox_norm:    tuple        # (x1, y1, x2, y2) normalised 0–1
    center_x:     float        # normalised horizontal centre
    center_y:     float        # normalised vertical centre
    area_norm:    float        # normalised area (0.0–1.0)
    frame_width:  int
    frame_height: int

    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"center=({self.center_x:.2f},{self.center_y:.2f}), "
            f"area={self.area_norm:.3f})"
        )


@dataclass
class DetectionResult:
    """
    The complete output of one call to ObjectDetector.detect().

    Fields:
        detections  : All detections above the floor threshold, unfiltered.
                      These include classes that navigation does not care about.
                      Use DetectionFilter to get only navigation-relevant ones.
        latency_ms  : Time taken for YOLO inference on this frame (milliseconds).
                      Does not include time to read the frame from the camera.
        frame_width : Width of the input frame in pixels.
        frame_height: Height of the input frame in pixels.
        raw_count   : Total number of detections returned by YOLO before
                      the floor threshold was applied.
    """
    detections:   List[Detection]
    latency_ms:   float
    frame_width:  int
    frame_height: int
    raw_count:    int


# ---------------------------------------------------------------------------
# ObjectDetector class
# ---------------------------------------------------------------------------

class ObjectDetector:
    """
    Loads YOLO11n and runs inference on individual video frames.

    Lifecycle:
        detector = ObjectDetector(config)
        detector.load()        # downloads/loads model — do this once at startup
        result = detector.detect(frame)  # call every frame
        # ... use result.detections ...

    Or use as a context manager:
        with ObjectDetector(config) as detector:
            result = detector.detect(frame)

    Args:
        config (dict): Full configuration dictionary from config.yaml.
                       Uses the 'detection' section.
    """

    # Absolute minimum confidence below which detections are never returned.
    # This is a floor to remove truly garbage detections (sub-15% confidence)
    # before they even reach the filter. It is NOT the navigation threshold.
    # The navigation threshold (config: confidence_threshold, default 0.35)
    # is applied by DetectionFilter.
    _FLOOR_CONFIDENCE = 0.15

    def __init__(self, config: dict) -> None:
        det_cfg = config.get("detection", {})

        self._model_name: str  = det_cfg.get("model_name", "yolo11n")
        self._conf_threshold: float = float(
            det_cfg.get("confidence_threshold", 0.35)
        )
        self._input_size: int  = int(det_cfg.get("input_size", 640))
        self._device: str      = det_cfg.get("device", "cpu")
        self._num_threads: int = int(det_cfg.get("num_threads", 4))

        # Resolve model path relative to project root.
        # config_loader stores the project root; we reconstruct it here.
        from assistive_navigation.utils.config_loader import get_project_root
        project_root = get_project_root()
        model_path_str = det_cfg.get("model_path", f"models/{self._model_name}.pt")

        # Ultralytics uses .pt weights. The config stores .onnx path (for Phase 14).
        # We override the extension to .pt for Phase 3.
        model_path = project_root / Path(model_path_str).with_suffix(".pt")
        self._model_path: Path = model_path

        self._model = None  # Loaded by load()
        self._is_loaded: bool = False

        # Running stats
        self._inference_count: int = 0
        self._total_latency_ms: float = 0.0

        logger.debug(
            "ObjectDetector created — model=%s, device=%s, input_size=%d, "
            "floor_conf=%.2f, nav_conf=%.2f",
            self._model_name, self._device, self._input_size,
            self._FLOOR_CONFIDENCE, self._conf_threshold
        )

    # -----------------------------------------------------------------------
    # Context manager
    # -----------------------------------------------------------------------

    def __enter__(self) -> "ObjectDetector":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the YOLO11n model.

        On first call, Ultralytics will download yolo11n.pt (~6 MB) from
        the official Ultralytics model hub and save it to the models/
        directory. Subsequent calls load from the local file.

        Sets torch CPU thread count to config.num_threads to prevent
        the model from consuming all CPU cores.

        Raises:
            DetectorError: If the model cannot be loaded.
        """
        if self._is_loaded:
            logger.warning("load() called but model is already loaded.")
            return

        logger.info(
            "Loading YOLO model: %s (device=%s) ...",
            self._model_name, self._device
        )

        # Set CPU thread count before loading the model.
        # This prevents PyTorch from consuming all available cores,
        # leaving headroom for the camera capture and audio threads.
        try:
            import torch
            if self._num_threads > 0:
                torch.set_num_threads(self._num_threads)
                logger.debug(
                    "PyTorch CPU thread count set to %d", self._num_threads
                )
        except Exception as e:
            logger.warning("Could not set torch thread count: %s", e)

        # Load the model via Ultralytics API.
        # If self._model_path does not exist, Ultralytics automatically
        # downloads the weights from its official hub.
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._model_path), verbose=False)
            # Warm-up: run inference on a blank frame so the first real
            # frame doesn't incur JIT compilation overhead.
            _blank = np.zeros((self._input_size, self._input_size, 3),
                              dtype=np.uint8)
            self._model.predict(
                _blank,
                conf=self._FLOOR_CONFIDENCE,
                imgsz=self._input_size,
                device=self._device,
                verbose=False
            )
            self._is_loaded = True
            logger.info(
                "YOLO model loaded and warmed up — %s on %s",
                self._model_name, self._device
            )
        except Exception as e:
            self._model = None
            raise DetectorError(
                f"Failed to load YOLO model '{self._model_name}'.\n"
                f"Model path attempted: {self._model_path}\n"
                f"Error: {e}"
            ) from e

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run YOLO11n inference on one frame and return all detections.

        Returns ALL detections above the floor threshold (0.15 confidence).
        This includes classes not relevant to navigation — the caller
        should pass the result through DetectionFilter to get only
        navigation-relevant detections.

        The raw (unfiltered) output is intentionally preserved here so
        that false-positive analysis can see what the model actually
        detected, not just what the filter let through.

        Args:
            frame: A BGR numpy array from OpenCV, shape (H, W, 3),
                   dtype uint8. Must not be None or empty.

        Returns:
            DetectionResult containing all detections above floor threshold,
            inference latency, and frame dimensions.

        Raises:
            DetectorError: If detect() is called before load().
            ValueError: If the frame is None, empty, or wrong shape.
        """
        if not self._is_loaded or self._model is None:
            raise DetectorError(
                "detect() called before load(). Call load() first."
            )

        if frame is None:
            raise ValueError("detect() received None frame.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"detect() expected (H, W, 3) BGR frame, "
                f"got shape {frame.shape}"
            )

        frame_h, frame_w = frame.shape[:2]

        # --- Run inference and time it ---
        t_start = time.perf_counter()

        results = self._model.predict(
            frame,
            conf=self._FLOOR_CONFIDENCE,  # Use floor here, not nav threshold
            imgsz=self._input_size,
            device=self._device,
            verbose=False
        )

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # --- Update running stats ---
        self._inference_count += 1
        self._total_latency_ms += latency_ms

        # --- Parse Ultralytics result into our Detection dataclass ---
        detections: List[Detection] = []
        raw_count = 0

        if results and len(results) > 0:
            result = results[0]  # We always pass one frame
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                raw_count = len(boxes)

                for i in range(raw_count):
                    # Bounding box in absolute pixel coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                    # Clamp to frame bounds (model can sometimes predict
                    # boxes slightly outside the frame due to padding)
                    x1 = max(0.0, min(x1, frame_w))
                    y1 = max(0.0, min(y1, frame_h))
                    x2 = max(0.0, min(x2, frame_w))
                    y2 = max(0.0, min(y2, frame_h))

                    conf = float(boxes.conf[i].item())
                    cls_id = int(boxes.cls[i].item())

                    # Get class name from the model's names dictionary
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")

                    # Normalised coordinates (0.0–1.0)
                    x1n = x1 / frame_w
                    y1n = y1 / frame_h
                    x2n = x2 / frame_w
                    y2n = y2 / frame_h

                    # Centre of bounding box (normalised)
                    cx = (x1n + x2n) / 2.0
                    cy = (y1n + y2n) / 2.0

                    # Area of bbox as fraction of total frame area
                    area_norm = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)

                    det = Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        bbox_norm=(x1n, y1n, x2n, y2n),
                        center_x=cx,
                        center_y=cy,
                        area_norm=area_norm,
                        frame_width=frame_w,
                        frame_height=frame_h
                    )
                    detections.append(det)

        logger.debug(
            "detect(): %d detections in %.1f ms (frame %d)",
            len(detections), latency_ms, self._inference_count
        )

        return DetectionResult(
            detections=detections,
            latency_ms=latency_ms,
            frame_width=frame_w,
            frame_height=frame_h,
            raw_count=raw_count
        )

    # -----------------------------------------------------------------------
    # Properties and stats
    # -----------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded and is ready for inference."""
        return self._is_loaded

    @property
    def model_name(self) -> str:
        """Name of the model as configured."""
        return self._model_name

    @property
    def input_size(self) -> int:
        """Input image dimension (pixels) for YOLO inference."""
        return self._input_size

    @input_size.setter
    def input_size(self, size: int) -> None:
        """Set input image dimension (pixels) for YOLO inference."""
        self._input_size = max(160, int(size))

    @property
    def inference_count(self) -> int:
        """Total number of frames processed since load()."""
        return self._inference_count

    @property
    def average_latency_ms(self) -> float:
        """Average inference latency in milliseconds over all frames so far."""
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count

    def get_model(self):
        """
        Return the internal Ultralytics YOLO model object.

        This is used by ObjectTracker to reuse the same model instance for
        tracking — avoiding the cost of loading YOLO twice.

        Only call this after load() has been called.

        Returns:
            The Ultralytics YOLO model object, or None if not yet loaded.
        """
        return self._model

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"ObjectDetector(model={self._model_name}, "
            f"device={self._device}, status={status}, "
            f"frames={self._inference_count}, "
            f"avg_latency={self.average_latency_ms:.1f}ms)"
        )
