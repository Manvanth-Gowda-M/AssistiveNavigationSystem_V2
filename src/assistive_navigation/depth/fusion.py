"""
Depth-Detection Fusion Module
==============================
Combines tracked-object bounding boxes with the depth map to produce
a unified representation (FusedObject) for each tracked object.

Phase: 7
Status: IMPLEMENTED

What this module does:
    - Takes List[TrackedObject] from the tracker (Phase 5)
    - Takes a depth map (H×W float32) from DepthEstimator (Phase 6)
    - For each tracked object, samples depth values from the central
      region of its bounding box
    - Applies temporal smoothing per track_id (sliding window over
      the last N depth values to reduce frame-to-frame flicker)
    - Maps the smoothed depth value to a proximity bucket
    - Derives the navigation priority from the object's class_id
    - Returns a FusedObject combining all of the above

What this module does NOT do:
    - Does NOT claim metric distance — all proximity is RELATIVE
    - Does NOT re-run YOLO (detector is Phase 3, unchanged)
    - Does NOT re-run the tracker (Phase 5, unchanged)
    - Does NOT estimate direction (Phase 8 — not yet implemented)
    - Does NOT score navigation priority (Phase 9 — not yet implemented)
    - Does NOT apply temporal confirmation (Phase 10 — not yet implemented)
    - Does NOT produce audio alerts (Phase 11–12 — not yet implemented)

Data flow (one frame):
    TrackedObject(s)       — from ObjectTracker.update()
    depth_map (H×W)        — from DepthEstimator.process_frame()
         |
         v
    DepthFusion.fuse()
         |
         v
    FusedObject(s)         — TrackedObject fields + depth + proximity + priority

Priority source:
    TrackedObject does not carry a priority field. Priority is derived from
    class_id using the same navigation class list from config.yaml that
    DetectionFilter uses. DepthFusion builds a class_id → priority lookup
    table at construction time.

Depth sampling method (per object):
    1. Extract central ROI of bbox (roi_center_fraction from config)
    2. Collect valid depth values (NaN/zero filtered)
    3. Trimmed median (5th–95th percentile) for robustness
    4. Per-track temporal smoothing (sliding window average)
    5. Classify into FAR/MEDIUM/CLOSE/VERY_CLOSE via config thresholds

    This approach is documented in Phase 7 requirements. The design choice
    of trimmed-median was validated in Phase 6 testing.

Temporal smoothing rationale:
    Depth Anything V2 runs every N frames (interval-based).
    Between updates, the same depth map is reused.
    When a new depth map arrives, the sampled ROI value may jump slightly
    due to model variance. A short sliding window average (default: 5 values)
    reduces visible proximity flickering without introducing significant lag.

Proximity threshold calibration (important disclaimer):
    The thresholds in config.yaml (very_close=0.75, close=0.55, medium=0.35)
    are RELATIVE to the current scene, not metric distances.
    A depth value of 0.75 means "this object is among the closest ~25%
    of elements in this scene." It does NOT mean 75 cm.
    Full calibration requires placing known objects at known positions and
    observing the resulting depth values — this is documented as a Phase 7
    follow-up step.

Usage:
    from assistive_navigation.depth.fusion import DepthFusion, FusedObject
    from assistive_navigation.utils.config_loader import load_config

    config = load_config()
    fusion = DepthFusion(config)

    # Each frame:
    fused_objects = fusion.fuse(
        tracked_objects,   # List[TrackedObject] from tracker
        depth_map,         # np.ndarray (H, W) from DepthEstimator, or None
        depth_age,         # int — frames since last depth inference
    )
    for obj in fused_objects:
        print(obj.track_id, obj.class_name, obj.proximity, obj.priority)
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from assistive_navigation.depth.depth_estimator import DepthEstimator, Proximity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FusedObject dataclass
# ---------------------------------------------------------------------------

@dataclass
class FusedObject:
    """
    A tracked object with depth and priority information attached.

    Combines all fields from TrackedObject (Phase 5) with the depth
    and proximity data produced by Phase 7 fusion.

    Fields from TrackedObject (passed through unchanged):
        track_id     : Stable integer ID from the tracker.
        class_id     : COCO class ID.
        class_name   : Human-readable class name ("person", "chair", etc.)
        confidence   : YOLO detection confidence (0.0–1.0).
                       High confidence ≠ correct detection.
        bbox         : (x1, y1, x2, y2) absolute pixel coordinates.
        bbox_norm    : (x1, y1, x2, y2) normalised 0.0–1.0.
        center_x     : Normalised horizontal centre (0=left, 1=right).
        center_y     : Normalised vertical centre (0=top, 1=bottom).
        age          : Frames this track has existed.
        last_seen    : Frames since last matched detection.
        stability    : matched_frames / total_age (0.0–1.0).
        area_norm    : Bbox area as fraction of frame area.
        frame_width  : Frame width in pixels.
        frame_height : Frame height in pixels.

    Fields added by Phase 7 fusion:
        raw_depth    : Smoothed median depth value for this object's bbox ROI.
                       Range 0.0–1.0 (higher = closer to camera).
                       Value is -1.0 if depth map was unavailable or
                       insufficient valid samples existed.
                       This is RELATIVE DEPTH — NOT a metric distance.
        proximity    : Relative proximity bucket for this object.
                       One of: Proximity.FAR / MEDIUM / CLOSE / VERY_CLOSE / UNKNOWN.
                       Based on raw_depth and configurable thresholds.
                       Do NOT interpret as metric distance.
        priority     : Navigation priority level derived from class_id.
                       One of: "navigation_critical" / "contextual" /
                               "low_priority" / "unknown"
                       Does NOT account for proximity yet (Phase 9 does that).
        depth_samples: Number of valid depth samples used to compute raw_depth.
                       0 if proximity is UNKNOWN.
        depth_age    : Frames since the last depth inference ran.
                       0 = depth map is current (updated this frame).
                       >0 = depth map is from a previous frame (cached).
                       A large depth_age means the proximity value may be stale.
    """
    # ── Passed through from TrackedObject ──────────────────────────────────
    track_id:     int
    class_id:     int
    class_name:   str
    confidence:   float
    bbox:         Tuple[float, float, float, float]
    bbox_norm:    Tuple[float, float, float, float]
    center_x:     float
    center_y:     float
    age:          int
    last_seen:    int
    stability:    float
    area_norm:    float
    frame_width:  int
    frame_height: int

    # ── Added by Phase 7 ───────────────────────────────────────────────────
    raw_depth:    float    # -1.0 if unavailable; else 0.0–1.0 (higher = closer)
    proximity:    str      # Proximity constant string
    priority:     str      # Priority category string
    depth_samples: int     # valid samples used for raw_depth (0 if unavailable)
    depth_age:    int      # frames since last depth inference

    def __repr__(self) -> str:
        return (
            f"FusedObject(id={self.track_id}, class={self.class_name}, "
            f"conf={self.confidence:.2f}, proximity={self.proximity}, "
            f"priority={self.priority}, depth={self.raw_depth:.3f}, "
            f"age={self.age})"
        )


# ---------------------------------------------------------------------------
# DepthFusion
# ---------------------------------------------------------------------------

class DepthFusion:
    """
    Fuses TrackedObject list with a depth map to produce FusedObject list.

    Call fuse() every frame (or whenever you have updated tracking results).
    The depth map may be None if the depth estimator hasn't produced one yet.

    Args:
        config (dict): Full configuration from config.yaml.
                       Uses: depth, proximity, object_classes sections.
    """

    # Sentinel value for "depth unavailable"
    _NO_DEPTH: float = -1.0

    def __init__(self, config: dict) -> None:
        depth_cfg = config.get("depth", {})
        prox_cfg  = config.get("proximity", {})

        # ROI sampling parameters (same as DepthEstimator)
        self._roi_fraction: float     = float(depth_cfg.get("roi_center_fraction", 0.5))
        self._min_valid_samples: int  = int(depth_cfg.get("min_valid_samples", 10))
        self._smoothing_window: int   = int(depth_cfg.get("smoothing_window", 5))

        # Proximity thresholds (same as DepthEstimator — single source of truth)
        self._thresh_very_close: float = float(prox_cfg.get("very_close_threshold", 0.75))
        self._thresh_close:      float = float(prox_cfg.get("close_threshold", 0.55))
        self._thresh_medium:     float = float(prox_cfg.get("medium_threshold", 0.35))

        # Priority lookup: class_id (int) → priority string
        # Built from the same config that DetectionFilter uses.
        self._priority_by_class: Dict[int, str] = self._build_priority_map(config)

        # Per-track depth history: track_id → deque of recent depth values
        # Used for temporal smoothing.
        self._depth_history: Dict[int, deque] = {}

        logger.debug(
            "DepthFusion created — roi_fraction=%.2f, smoothing_window=%d, "
            "priority_classes=%d",
            self._roi_fraction, self._smoothing_window,
            len(self._priority_by_class)
        )

    # -----------------------------------------------------------------------
    # Main fuse method
    # -----------------------------------------------------------------------

    def fuse(
        self,
        tracked_objects: list,                    # List[TrackedObject]
        depth_map: Optional[np.ndarray],          # (H, W) float32 or None
        depth_age: int = 0,                       # frames since last depth inference
    ) -> List[FusedObject]:
        """
        Combine tracked objects with depth information.

        For each TrackedObject:
          1. Sample depth from the central ROI of its bounding box.
          2. Apply trimmed-median to get a robust depth value.
          3. Apply temporal smoothing (sliding window per track).
          4. Classify into a proximity bucket.
          5. Look up the navigation priority from class_id.
          6. Return a FusedObject with all fields combined.

        If depth_map is None:
          - raw_depth is set to -1.0
          - proximity is set to UNKNOWN
          - smoothing history is not updated

        Stale depth maps (large depth_age) are still used — the caller
        is responsible for deciding whether to act on stale depth.
        The depth_age field on each FusedObject makes staleness visible.

        Args:
            tracked_objects:  List[TrackedObject] from ObjectTracker.update().
            depth_map:        (H, W) float32 numpy array, 0.0–1.0, higher=closer.
                              None if depth estimator has not run yet.
            depth_age:        Frames since the last depth inference.

        Returns:
            List[FusedObject], one per tracked object. Empty list if no tracks.
        """
        if not tracked_objects:
            # Clean up history for tracks no longer active
            self._depth_history.clear()
            return []

        # Build a set of currently active track IDs for history cleanup
        active_ids = {obj.track_id for obj in tracked_objects}

        # Remove history for tracks no longer present
        stale_ids = set(self._depth_history.keys()) - active_ids
        for sid in stale_ids:
            del self._depth_history[sid]

        fused: List[FusedObject] = []

        for obj in tracked_objects:
            raw_depth, n_samples, proximity = self._compute_depth_proximity(
                obj, depth_map
            )

            priority = self._get_priority(obj.class_id)

            fused.append(FusedObject(
                # Pass-through from TrackedObject
                track_id=obj.track_id,
                class_id=obj.class_id,
                class_name=obj.class_name,
                confidence=obj.confidence,
                bbox=obj.bbox,
                bbox_norm=obj.bbox_norm,
                center_x=obj.center_x,
                center_y=obj.center_y,
                age=obj.age,
                last_seen=obj.last_seen,
                stability=obj.stability,
                area_norm=obj.area_norm,
                frame_width=obj.frame_width,
                frame_height=obj.frame_height,
                # Phase 7 additions
                raw_depth=raw_depth,
                proximity=proximity,
                priority=priority,
                depth_samples=n_samples,
                depth_age=depth_age,
            ))

        logger.debug(
            "DepthFusion.fuse(): %d objects, depth_age=%d",
            len(fused), depth_age
        )

        return fused

    # -----------------------------------------------------------------------
    # Depth sampling and smoothing
    # -----------------------------------------------------------------------

    def _compute_depth_proximity(
        self,
        obj,                               # TrackedObject
        depth_map: Optional[np.ndarray],
    ) -> Tuple[float, int, str]:
        """
        Sample depth for one object's bounding box and return
        (smoothed_depth_value, n_samples, proximity_string).

        Returns (-1.0, 0, Proximity.UNKNOWN) if depth_map is None or
        insufficient valid samples exist.
        """
        if depth_map is None:
            return self._NO_DEPTH, 0, Proximity.UNKNOWN

        # Extract ROI from depth map
        raw_value, n_samples = self._sample_roi(obj.bbox, depth_map)

        if raw_value is None:
            return self._NO_DEPTH, 0, Proximity.UNKNOWN

        # Apply per-track temporal smoothing
        smoothed = self._smooth(obj.track_id, raw_value)

        # Classify proximity
        proximity = self._classify(smoothed)

        return smoothed, n_samples, proximity

    def _sample_roi(
        self,
        bbox: Tuple,
        depth_map: np.ndarray,
    ) -> Tuple[Optional[float], int]:
        """
        Sample the central ROI of a bounding box from the depth map.

        Returns (median_depth_value, n_valid_samples).
        Returns (None, 0) if insufficient valid samples.

        The central ROI uses roi_center_fraction — by default the central 50%
        of the bbox in each dimension. This avoids background pixels at the
        edges of the bounding box which often belong to a different surface.

        The trimmed median (5th–95th percentile) is used instead of a plain
        median to reduce sensitivity to specular highlights or depth artefacts
        at the object boundary.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape

        # Clamp to frame bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None, 0

        # Shrink to central ROI
        bw = x2 - x1
        bh = y2 - y1
        shrink_x = bw * (1.0 - self._roi_fraction) / 2.0
        shrink_y = bh * (1.0 - self._roi_fraction) / 2.0

        cx1 = int(x1 + shrink_x)
        cy1 = int(y1 + shrink_y)
        cx2 = int(x2 - shrink_x)
        cy2 = int(y2 - shrink_y)

        # Clamp again after shrink
        cx1 = max(0, min(cx1, w - 1))
        cy1 = max(0, min(cy1, h - 1))
        cx2 = max(cx1 + 1, min(cx2, w))
        cy2 = max(cy1 + 1, min(cy2, h))

        roi = depth_map[cy1:cy2, cx1:cx2].flatten()

        # Filter invalid values
        valid = roi[
            np.isfinite(roi) &
            (roi > 0.0) &
            (roi <= 1.0)
        ]

        n_valid = len(valid)
        if n_valid < self._min_valid_samples:
            return None, n_valid

        # Trimmed median
        p5  = np.percentile(valid, 5)
        p95 = np.percentile(valid, 95)
        trimmed = valid[(valid >= p5) & (valid <= p95)]

        if len(trimmed) == 0:
            trimmed = valid  # fallback

        return float(np.median(trimmed)), n_valid

    def _smooth(self, track_id: int, raw_value: float) -> float:
        """
        Apply temporal smoothing to the depth value for one track.

        Maintains a sliding window deque of size smoothing_window per
        track_id. Returns the mean of recent values.

        Why smoothing:
            Depth Anything V2 output can vary slightly between frames even
            for a static object, because the normalisation (d - min)/(max - min)
            is scene-relative and changes when other objects enter or leave the
            frame. A short window average (default 5) absorbs this variance
            without introducing noticeable lag.
        """
        if track_id not in self._depth_history:
            self._depth_history[track_id] = deque(maxlen=self._smoothing_window)

        self._depth_history[track_id].append(raw_value)
        return float(np.mean(self._depth_history[track_id]))

    def _classify(self, depth_value: float) -> str:
        """Apply proximity thresholds to a smoothed depth value."""
        if depth_value > self._thresh_very_close:
            return Proximity.VERY_CLOSE
        if depth_value > self._thresh_close:
            return Proximity.CLOSE
        if depth_value > self._thresh_medium:
            return Proximity.MEDIUM
        return Proximity.FAR

    # -----------------------------------------------------------------------
    # Priority lookup
    # -----------------------------------------------------------------------

    def _get_priority(self, class_id: int) -> str:
        """Return the navigation priority for a class_id."""
        return self._priority_by_class.get(class_id, "unknown")

    @staticmethod
    def _build_priority_map(config: dict) -> Dict[int, str]:
        """
        Build a class_id → priority string lookup table from config.yaml.

        Uses the same object_classes structure as DetectionFilter so the
        priority classification is guaranteed consistent.
        """
        obj_cfg = config.get("object_classes", {})
        result: Dict[int, str] = {}

        categories = {
            "navigation_critical": obj_cfg.get("navigation_critical", []),
            "contextual":          obj_cfg.get("contextual", []),
            "low_priority":        obj_cfg.get("low_priority", []),
        }

        for priority_str, class_list in categories.items():
            for item in class_list:
                if isinstance(item, dict):
                    cid = item.get("id")
                    if cid is not None:
                        result[int(cid)] = priority_str

        return result

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def reset_history(self, track_id: Optional[int] = None) -> None:
        """
        Clear the depth smoothing history.

        Args:
            track_id: If provided, clear only this track's history.
                      If None, clear all history.
        """
        if track_id is None:
            self._depth_history.clear()
            logger.debug("DepthFusion: all depth history cleared.")
        elif track_id in self._depth_history:
            del self._depth_history[track_id]
            logger.debug("DepthFusion: history cleared for track %d.", track_id)

    def get_depth_history(self, track_id: int) -> List[float]:
        """Return a copy of the depth smoothing history for a track."""
        h = self._depth_history.get(track_id)
        return list(h) if h else []

    @property
    def active_track_count(self) -> int:
        """Number of tracks currently holding depth history."""
        return len(self._depth_history)

    def __repr__(self) -> str:
        return (
            f"DepthFusion(roi_fraction={self._roi_fraction}, "
            f"smoothing_window={self._smoothing_window}, "
            f"active_tracks={self.active_track_count})"
        )
