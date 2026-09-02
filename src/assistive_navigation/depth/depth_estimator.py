"""
Depth Estimator Module
======================
Generates a relative depth map from a single camera frame using
Depth Anything V2 Small via ONNX Runtime.

Phase: 6
Status: IMPLEMENTED

What this module does:
    - Loads Depth Anything V2 Small (ONNX, float32, ~99 MB)
    - Accepts a BGR frame from OpenCV
    - Runs inference every N frames (configurable); caches the last result
    - Returns a normalised depth map: 2D numpy array, float32, values 0.0–1.0
      where HIGHER = CLOSER to camera
    - Provides classify_proximity() and get_proximity_at_bbox() utilities
    - Reports proximity as: FAR / MEDIUM / CLOSE / VERY_CLOSE
    - Does NOT claim metric distances — only RELATIVE proximity

What this module does NOT do:
    - Does not combine depth with tracked objects (Phase 7 — fusion.py)
    - Does not assign direction (Phase 8)
    - Does not calculate navigation priority (Phase 9)
    - Does not claim to measure exact distances in metres

IMPORTANT — Relative vs. metric depth:
    Depth Anything V2 produces AFFINE-INVARIANT relative depth.
    This means the output tells you the ORDER of distances (what is closer,
    what is further) but NOT the absolute values in metres.
    The normalised value 0.75 does NOT mean "75 cm away."
    It means "this pixel is in the closer ~25% of the scene depth range."
    All proximity outputs must be labelled RELATIVE PROXIMITY, not distance.

Model details:
    Name    : Depth Anything V2 Small (ViT-S backbone, DINOv2)
    Variant : Float32 ONNX (single file, no external data)
    Source  : onnx-community/depth-anything-v2-small on HuggingFace
    License : Apache-2.0 (Small variant ONLY)
    Size    : ~99 MB
    Input   : pixel_values — (1, 3, H, W) float32, RGB, ImageNet-normalised
    Output  : predicted_depth — (1, H, W) float32, raw disparity values

Preprocessing (verified from official preprocessor_config.json):
    1. Resize to input_size × input_size (keeping aspect ratio, padding
       to nearest multiple of 14)
    2. Convert BGR → RGB
    3. Divide by 255.0 (rescale to 0–1)
    4. Subtract ImageNet mean [0.485, 0.456, 0.406]
    5. Divide by ImageNet std  [0.229, 0.224, 0.225]
    6. Add batch dimension → shape (1, 3, H, W)

Post-processing:
    1. Remove batch dim → (H, W)
    2. Resize back to original frame size
    3. Normalise to 0.0–1.0 range: (d - min) / (max - min)
       Result: higher = closer to camera

CPU inference benchmarks (i5-12450H, onnxruntime 1.23.2):
    input_size=252: ~118ms  ← default (recommended)
    input_size=364: ~260ms
    input_size=518: ~366ms  (native size)

Usage:
    from assistive_navigation.utils.config_loader import load_config
    from assistive_navigation.depth.depth_estimator import DepthEstimator

    config = load_config()
    estimator = DepthEstimator(config)
    estimator.load()

    # Call every frame — depth only runs every N frames internally
    depth_map = estimator.process_frame(frame)

    if depth_map is not None:
        # depth_map is (H, W) float32, 0.0–1.0, higher = closer
        prox = estimator.get_proximity_at_bbox((x1, y1, x2, y2))
        print(prox)  # 'FAR', 'MEDIUM', 'CLOSE', or 'VERY_CLOSE'
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Proximity constants
# ---------------------------------------------------------------------------

class Proximity:
    """
    Relative proximity labels.

    These are NOT metric distances. They describe where an object sits in the
    relative depth ordering of the current scene.

    FAR        : Object is among the more distant elements in the frame.
    MEDIUM     : Object is at a moderate relative distance.
    CLOSE      : Object is relatively close.
    VERY_CLOSE : Object is among the nearest elements in the frame.
    UNKNOWN    : Could not determine proximity (insufficient valid samples,
                 or depth map not yet available).
    """
    FAR        = "FAR"
    MEDIUM     = "MEDIUM"
    CLOSE      = "CLOSE"
    VERY_CLOSE = "VERY_CLOSE"
    UNKNOWN    = "UNKNOWN"


# ---------------------------------------------------------------------------
# DepthEstimator
# ---------------------------------------------------------------------------

class DepthEstimator:
    """
    Runs Depth Anything V2 Small to produce relative depth maps.

    Call load() once at startup. Then call process_frame() every frame.
    Inference only runs every depth_update_interval frames (configurable).
    Between updates, the cached depth map is returned with frame_age > 0.

    Args:
        config (dict): Full configuration dictionary from config.yaml.
                       Uses the 'depth' and 'proximity' sections.
    """

    # ImageNet normalisation constants (verified from preprocessor_config.json)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, config: dict) -> None:
        depth_cfg = config.get("depth", {})
        prox_cfg  = config.get("proximity", {})

        # Model settings
        from assistive_navigation.utils.config_loader import get_project_root
        project_root = get_project_root()
        model_path_str = depth_cfg.get(
            "model_path", "models/depth_anything_v2_small.onnx"
        )
        self._model_path: Path = project_root / model_path_str
        self._input_size: int  = int(depth_cfg.get("input_size", 252))
        self._update_interval: int = int(depth_cfg.get("depth_update_interval", 5))
        self._roi_fraction: float  = float(depth_cfg.get("roi_center_fraction", 0.5))
        self._min_valid_samples: int = int(depth_cfg.get("min_valid_samples", 10))

        # Thread count for ONNX Runtime (reuse detection thread setting)
        det_cfg = config.get("detection", {})
        self._num_threads: int = int(det_cfg.get("num_threads", 4))

        # Proximity thresholds (higher depth value = closer)
        self._thresh_very_close: float = float(
            prox_cfg.get("very_close_threshold", 0.75)
        )
        self._thresh_close: float = float(
            prox_cfg.get("close_threshold", 0.55)
        )
        self._thresh_medium: float = float(
            prox_cfg.get("medium_threshold", 0.35)
        )

        # Runtime state
        self._session = None           # ONNX Runtime InferenceSession
        self._is_loaded: bool = False
        self._input_name: str = ""
        self._output_name: str = ""

        # Cached depth map and metadata
        self._cached_depth_map: Optional[np.ndarray] = None
        self._cached_frame_h: int = 0
        self._cached_frame_w: int = 0
        self._frame_counter: int  = 0   # total process_frame() calls
        self._depth_frame_age: int = 0  # frames since last depth inference
        self._last_latency_ms: float = 0.0
        self._inference_count: int   = 0
        self._total_latency_ms: float = 0.0

        logger.debug(
            "DepthEstimator created — model=%s, input_size=%d, interval=%d",
            self._model_path.name, self._input_size, self._update_interval
        )

    # -----------------------------------------------------------------------
    # Context manager
    # -----------------------------------------------------------------------

    def __enter__(self) -> "DepthEstimator":
        self.load()
        return self

    def __exit__(self, *args) -> bool:
        return False

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the Depth Anything V2 Small ONNX model.

        Must be called once before process_frame().
        Raises DepthEstimatorError if the model file is missing or invalid.
        """
        if self._is_loaded:
            logger.warning("load() called but model already loaded.")
            return

        if not self._model_path.exists():
            raise DepthEstimatorError(
                f"Depth model not found: {self._model_path}\n"
                f"Run: python scripts/download_models.py --model depth_anything"
            )

        logger.info("Loading depth model: %s", self._model_path.name)

        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = self._num_threads
            options.inter_op_num_threads = 1
            # Suppress ONNX Runtime verbose output
            options.log_severity_level = 3

            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

            self._input_name  = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name

            # Warm-up pass: run on a small blank frame to prime the JIT
            dummy = np.zeros(
                (1, 3, self._input_size, self._input_size), dtype=np.float32
            )
            self._session.run(
                [self._output_name], {self._input_name: dummy}
            )

            self._is_loaded = True
            logger.info(
                "Depth model loaded: %s  input=%s  output=%s",
                self._model_path.name,
                self._input_name,
                self._output_name,
            )

        except ImportError:
            raise DepthEstimatorError(
                "onnxruntime is not installed. "
                "Run: pip install onnxruntime"
            )
        except Exception as e:
            self._session = None
            raise DepthEstimatorError(
                f"Failed to load depth model: {e}"
            ) from e

    # -----------------------------------------------------------------------
    # Core: process one frame
    # -----------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        force: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Process one camera frame and return the current depth map.

        Inference only runs every depth_update_interval frames.
        On skipped frames, the cached depth map is returned unchanged.
        On the first call (before any inference), returns None.

        Args:
            frame:  BGR numpy array from OpenCV, shape (H, W, 3), dtype uint8.
            force:  If True, run inference this frame regardless of interval.
                    Useful for testing or single-shot evaluation.

        Returns:
            numpy array (H, W) float32, values 0.0–1.0, higher = closer.
            None if no depth map has been computed yet (first few frames).

        Raises:
            DepthEstimatorError: If called before load().
        """
        if not self._is_loaded or self._session is None:
            raise DepthEstimatorError(
                "process_frame() called before load(). Call load() first."
            )
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected BGR (H, W, 3) frame, got shape "
                f"{frame.shape if frame is not None else None}"
            )

        self._frame_counter += 1
        self._depth_frame_age += 1

        # Decide whether to run inference
        should_update = force or (self._depth_frame_age >= self._update_interval)

        if should_update:
            depth_map = self._run_inference(frame)
            if depth_map is not None:
                self._cached_depth_map = depth_map
                self._cached_frame_h, self._cached_frame_w = frame.shape[:2]
                self._depth_frame_age = 0
        else:
            logger.debug(
                "Depth: skipping frame %d (age=%d < interval=%d), using cache",
                self._frame_counter,
                self._depth_frame_age,
                self._update_interval,
            )

        return self._cached_depth_map

    # -----------------------------------------------------------------------
    # Inference internals
    # -----------------------------------------------------------------------

    def _run_inference(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run ONNX inference on one frame.

        Returns normalised depth map (H_orig, W_orig) float32 or None on error.
        """
        h_orig, w_orig = frame.shape[:2]

        try:
            # 1. Preprocess
            tensor = self._preprocess(frame)

            # 2. Inference
            t_start = time.perf_counter()
            outputs = self._session.run(
                [self._output_name], {self._input_name: tensor}
            )
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            self._last_latency_ms  = latency_ms
            self._inference_count  += 1
            self._total_latency_ms += latency_ms

            logger.debug(
                "Depth inference: %.1f ms (frame %d, count %d)",
                latency_ms, self._frame_counter, self._inference_count
            )

            # 3. Post-process
            depth_raw = outputs[0]           # (1, H_inf, W_inf)
            depth_map = self._postprocess(depth_raw, h_orig, w_orig)

            return depth_map

        except Exception as e:
            logger.warning(
                "Depth inference failed on frame %d: %s",
                self._frame_counter, e
            )
            return None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert a BGR OpenCV frame to the normalised input tensor.

        Steps:
          1. Resize to (input_size, input_size), respecting aspect ratio
             and padding to a multiple of 14 (as required by DINOv2 backbone)
          2. Convert BGR → RGB
          3. Scale pixels to 0.0–1.0 (÷ 255)
          4. Subtract ImageNet mean, divide by ImageNet std
          5. Transpose to (C, H, W) and add batch dim → (1, C, H, W)

        Returns:
            float32 numpy array, shape (1, 3, input_size, input_size)
        """
        target = self._input_size

        # Resize with aspect-ratio preservation; pad to target
        h, w = frame.shape[:2]
        scale = target / max(h, w)
        new_h = int(round(h * scale / 14)) * 14
        new_w = int(round(w * scale / 14)) * 14
        new_h = max(14, new_h)
        new_w = max(14, new_w)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target × target if needed
        pad_h = target - new_h
        pad_w = target - new_w
        if pad_h > 0 or pad_w > 0:
            resized = np.pad(
                resized,
                ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        # Crop to exactly target × target (handles rounding)
        resized = resized[:target, :target]

        # BGR → RGB
        rgb = resized[:, :, ::-1].astype(np.float32)

        # Scale to 0–1
        rgb /= 255.0

        # ImageNet normalise
        rgb = (rgb - self._MEAN) / self._STD

        # (H, W, C) → (1, C, H, W)
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...]

        return tensor.astype(np.float32)

    def _postprocess(
        self,
        depth_raw: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        """
        Convert raw model output to a normalised (0–1) depth map at original size.

        Steps:
          1. Remove batch dim: (1, H, W) → (H, W)
          2. Resize to original frame dimensions
          3. Normalise to 0.0–1.0: (d - min) / (max - min)
             Higher value = closer to camera.

        Returns:
            float32 (orig_h, orig_w) array, values in [0.0, 1.0].
        """
        # Remove batch dim
        d = depth_raw[0]  # (H, W)

        # Resize to original frame size
        if d.shape != (orig_h, orig_w):
            d = cv2.resize(
                d.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )

        # Normalise to 0–1 (higher = closer)
        d_min = d.min()
        d_max = d.max()
        if d_max - d_min > 1e-8:
            d = (d - d_min) / (d_max - d_min)
        else:
            # Flat depth (uniform scene or blank frame)
            d = np.zeros_like(d, dtype=np.float32)

        return d.astype(np.float32)

    # -----------------------------------------------------------------------
    # Proximity utilities
    # -----------------------------------------------------------------------

    def classify_proximity(self, depth_value: float) -> str:
        """
        Map a single normalised depth value to a proximity bucket.

        This is a RELATIVE classification — not a metric distance.
        'Higher depth value' means 'closer in this frame', but the actual
        distance in metres is NOT known or claimed.

        Thresholds come from config.yaml [proximity] section.
        They are starting estimates and must be validated empirically.

        Args:
            depth_value: float in [0.0, 1.0], higher = closer.

        Returns:
            One of: Proximity.FAR, MEDIUM, CLOSE, VERY_CLOSE
        """
        if depth_value > self._thresh_very_close:
            return Proximity.VERY_CLOSE
        if depth_value > self._thresh_close:
            return Proximity.CLOSE
        if depth_value > self._thresh_medium:
            return Proximity.MEDIUM
        return Proximity.FAR

    def get_proximity_at_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        depth_map: Optional[np.ndarray] = None,
    ) -> str:
        """
        Estimate the relative proximity of a detected object using its bbox.

        Samples depth values from the CENTRAL region of the bounding box
        (roi_center_fraction × bbox size) to avoid background leakage.
        Uses the median of valid samples for robustness.

        Note: This method is provided for Phase 6 independent testing.
        Phase 7 (fusion.py) will implement this more fully with temporal
        smoothing and per-track history.

        Args:
            bbox:       (x1, y1, x2, y2) in absolute pixels.
            depth_map:  Optional depth map to use. If None, uses the cached
                        depth map from the last process_frame() call.

        Returns:
            Proximity string (FAR / MEDIUM / CLOSE / VERY_CLOSE / UNKNOWN).
            Returns UNKNOWN if no depth map is available or insufficient
            valid samples exist within the bbox region.
        """
        if depth_map is None:
            depth_map = self._cached_depth_map

        if depth_map is None:
            return Proximity.UNKNOWN

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape

        # Clamp to frame bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return Proximity.UNKNOWN

        # Central ROI
        bw = x2 - x1
        bh = y2 - y1
        shrink_x = bw * (1.0 - self._roi_fraction) / 2.0
        shrink_y = bh * (1.0 - self._roi_fraction) / 2.0

        cx1 = int(x1 + shrink_x)
        cy1 = int(y1 + shrink_y)
        cx2 = int(x2 - shrink_x)
        cy2 = int(y2 - shrink_y)

        # Safety clamp after shrink
        cx1 = max(0, min(cx1, w - 1))
        cy1 = max(0, min(cy1, h - 1))
        cx2 = max(cx1 + 1, min(cx2, w))
        cy2 = max(cy1 + 1, min(cy2, h))

        roi = depth_map[cy1:cy2, cx1:cx2]
        samples = roi.flatten()

        # Remove invalid values
        valid = samples[
            np.isfinite(samples) &
            (samples > 0.0) &
            (samples <= 1.0)
        ]

        if len(valid) < self._min_valid_samples:
            logger.debug(
                "get_proximity_at_bbox: only %d valid samples (min=%d)",
                len(valid), self._min_valid_samples
            )
            return Proximity.UNKNOWN

        # Use trimmed median: exclude top and bottom 5% to reduce outliers
        p5  = np.percentile(valid, 5)
        p95 = np.percentile(valid, 95)
        trimmed = valid[(valid >= p5) & (valid <= p95)]

        if len(trimmed) == 0:
            trimmed = valid  # fallback if trimming removed everything

        depth_value = float(np.median(trimmed))
        return self.classify_proximity(depth_value)

    def get_depth_at_point(
        self,
        x: int,
        y: int,
        depth_map: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """
        Return the normalised depth value at a single pixel coordinate.

        Useful for spot-checks and visualisation.

        Returns:
            float 0.0–1.0 (higher = closer), or None if out of bounds.
        """
        if depth_map is None:
            depth_map = self._cached_depth_map
        if depth_map is None:
            return None
        h, w = depth_map.shape
        if 0 <= y < h and 0 <= x < w:
            return float(depth_map[y, x])
        return None

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

    def get_colourmap(
        self,
        depth_map: Optional[np.ndarray] = None,
        colourmap: int = cv2.COLORMAP_PLASMA,
    ) -> Optional[np.ndarray]:
        """
        Convert a normalised depth map to a BGR colour image for display.

        The colour map makes depth structure visible on screen:
          Warm colours (yellow/white) = closer
          Cool colours (purple/dark)  = further

        Args:
            depth_map:  Depth array to visualise. Uses cache if None.
            colourmap:  OpenCV colour map constant (default: COLORMAP_PLASMA).

        Returns:
            BGR uint8 numpy array same shape as depth_map, or None.
        """
        if depth_map is None:
            depth_map = self._cached_depth_map
        if depth_map is None:
            return None

        # Scale to 0–255 uint8
        d8 = (depth_map * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(d8, colourmap)

    # -----------------------------------------------------------------------
    # Properties and stats
    # -----------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True after load() succeeded."""
        return self._is_loaded

    @property
    def has_depth_map(self) -> bool:
        """True if at least one depth map has been computed and cached."""
        return self._cached_depth_map is not None

    @property
    def depth_frame_age(self) -> int:
        """Frames since the last depth inference. 0 = ran this frame."""
        return self._depth_frame_age

    @property
    def last_latency_ms(self) -> float:
        """Inference time of the most recent depth inference (ms)."""
        return self._last_latency_ms

    @property
    def average_latency_ms(self) -> float:
        """Average inference time over all depth inferences so far (ms)."""
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count

    @property
    def inference_count(self) -> int:
        """Total number of depth inference calls made."""
        return self._inference_count

    @property
    def frame_counter(self) -> int:
        """Total number of process_frame() calls made."""
        return self._frame_counter

    @property
    def update_interval(self) -> int:
        """Configured depth update interval (frames)."""
        return self._update_interval

    @property
    def input_size(self) -> int:
        """Configured model input size (pixels, square)."""
        return self._input_size

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        has_map = "yes" if self.has_depth_map else "no"
        return (
            f"DepthEstimator(model={self._model_path.name}, "
            f"input_size={self._input_size}, "
            f"interval={self._update_interval}, "
            f"status={status}, "
            f"has_depth_map={has_map}, "
            f"inferences={self._inference_count}, "
            f"avg_latency={self.average_latency_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class DepthEstimatorError(Exception):
    """Raised when the depth model cannot be loaded or produces invalid output."""
    pass
