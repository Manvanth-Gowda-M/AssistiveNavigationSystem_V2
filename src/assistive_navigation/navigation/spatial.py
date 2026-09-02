"""
Spatial Reasoning Module
========================
Determines the horizontal direction (LEFT / CENTER / RIGHT) for each
fused object using its normalised bounding-box center_x.

Phase: 8
Status: IMPLEMENTED

What this module does:
    - Accepts List[FusedObject] from Phase 7 (DepthFusion)
    - Reads center_x from each FusedObject (already normalised 0.0–1.0)
    - Compares center_x to configurable zone thresholds
    - Produces a DirectedObject for each FusedObject — identical except
      for the added 'direction' field
    - Does NOT recalculate the bounding box (center_x is already correct)

What this module does NOT do:
    - Does NOT re-run detection (Phase 3), tracking (Phase 5),
      depth (Phase 6), or fusion (Phase 7)
    - Does NOT score navigation priority (Phase 9)
    - Does NOT apply temporal confirmation (Phase 10)
    - Does NOT produce audio alerts (Phase 11–12)
    - Does NOT estimate vertical position or 3D angle

Direction zones (from config.yaml spatial section):
    LEFT   :  center_x < left_zone_end         default: < 0.35
    CENTER :  left_zone_end <= center_x
              < right_zone_start               default: [0.35, 0.65)
    RIGHT  :  center_x >= right_zone_start     default: >= 0.65

Boundary rule (deterministic, no ambiguity):
    At exactly center_x = left_zone_end:  → CENTER  (not LEFT)
    At exactly center_x = right_zone_start: → RIGHT (not CENTER)
    This follows a half-open interval convention:
        LEFT   = [0.0, left_zone_end)
        CENTER = [left_zone_end, right_zone_start)
        RIGHT  = [right_zone_start, 1.0]

Invalid center_x handling:
    center_x < 0.0  → treat as LEFT (clamped)
    center_x > 1.0  → treat as RIGHT (clamped)
    center_x is NaN → fallback to CENTER, log warning

Threshold validation:
    left_zone_end must be strictly less than right_zone_start.
    If misconfigured, SpatialReasoner raises ValueError at construction —
    never silently produces wrong results.

Usage:
    from assistive_navigation.navigation.spatial import SpatialReasoner
    from assistive_navigation.utils.config_loader import load_config

    config = load_config()
    spatial = SpatialReasoner(config)

    # Each frame: takes List[FusedObject], returns List[DirectedObject]
    directed = spatial.assign_direction(fused_objects)
    for obj in directed:
        print(obj.track_id, obj.class_name, obj.direction, obj.proximity)
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Direction constants
# ---------------------------------------------------------------------------

class Direction:
    """
    Horizontal direction labels.

    LEFT   : Object's centre is in the left zone of the frame.
    CENTER : Object's centre is in the central zone of the frame.
    RIGHT  : Object's centre is in the right zone of the frame.
    UNKNOWN: center_x was not a valid finite number (edge case).
    """
    LEFT    = "LEFT"
    CENTER  = "CENTER"
    RIGHT   = "RIGHT"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# DirectedObject dataclass
# ---------------------------------------------------------------------------

@dataclass
class DirectedObject:
    """
    A FusedObject with a horizontal direction label added.

    All fields from FusedObject (Phase 7) are preserved unchanged.
    The 'direction' field is the only addition made by Phase 8.

    Fields from FusedObject (passed through):
        track_id     : Stable integer ID from the tracker.
        class_id     : COCO class ID.
        class_name   : Human-readable class name.
        confidence   : YOLO detection confidence (0.0–1.0).
        bbox         : (x1, y1, x2, y2) absolute pixel coordinates.
        bbox_norm    : (x1, y1, x2, y2) normalised 0.0–1.0.
        center_x     : Normalised horizontal centre (0=left, 1=right).
        center_y     : Normalised vertical centre (0=top, 1=bottom).
        age          : Total frames this track has existed.
        last_seen    : Frames since last matched detection.
        stability    : matched_frames / total_age (0.0–1.0).
        area_norm    : Bbox area as fraction of frame area.
        frame_width  : Frame width in pixels.
        frame_height : Frame height in pixels.
        raw_depth    : Smoothed depth value (0.0–1.0 or -1.0 if unknown).
        proximity    : FAR / MEDIUM / CLOSE / VERY_CLOSE / UNKNOWN.
        priority     : navigation_critical / contextual / low_priority / unknown.
        depth_samples: Valid depth samples used.
        depth_age    : Frames since last depth inference.

    Field added by Phase 8:
        direction    : "LEFT" | "CENTER" | "RIGHT" | "UNKNOWN"
                       Horizontal position of this object in the frame.
                       Based on center_x relative to configurable zone thresholds.
    """
    # ── From FusedObject (all passed through) ─────────────────────────────
    track_id:      int
    class_id:      int
    class_name:    str
    confidence:    float
    bbox:          Tuple[float, float, float, float]
    bbox_norm:     Tuple[float, float, float, float]
    center_x:      float
    center_y:      float
    age:           int
    last_seen:     int
    stability:     float
    area_norm:     float
    frame_width:   int
    frame_height:  int
    raw_depth:     float
    proximity:     str
    priority:      str
    depth_samples: int
    depth_age:     int

    # ── Added by Phase 8 ──────────────────────────────────────────────────
    direction:     str     # Direction constant string

    def __repr__(self) -> str:
        return (
            f"DirectedObject(id={self.track_id}, class={self.class_name}, "
            f"conf={self.confidence:.2f}, direction={self.direction}, "
            f"proximity={self.proximity}, priority={self.priority})"
        )


# ---------------------------------------------------------------------------
# SpatialReasoner
# ---------------------------------------------------------------------------

class SpatialReasoner:
    """
    Assigns LEFT / CENTER / RIGHT direction to each FusedObject.

    Uses the object's normalised center_x (0.0 = left edge, 1.0 = right edge)
    and compares it to configurable zone thresholds from config.yaml.

    Construction validates that the thresholds are logically consistent.
    No changes to config.yaml are required — the spatial: section was
    created in Phase 1 and is already correct.

    Args:
        config (dict): Full configuration from config.yaml.
                       Uses the 'spatial' section.
    """

    def __init__(self, config: dict) -> None:
        spatial_cfg = config.get("spatial", {})

        self._left_end:     float = float(spatial_cfg.get("left_zone_end",    0.35))
        self._right_start:  float = float(spatial_cfg.get("right_zone_start", 0.65))

        # Validate thresholds — fail fast rather than silently produce
        # wrong results from a misconfigured config file.
        if not (0.0 < self._left_end < 1.0):
            raise ValueError(
                f"spatial.left_zone_end must be in (0.0, 1.0), "
                f"got {self._left_end}"
            )
        if not (0.0 < self._right_start < 1.0):
            raise ValueError(
                f"spatial.right_zone_start must be in (0.0, 1.0), "
                f"got {self._right_start}"
            )
        if self._left_end >= self._right_start:
            raise ValueError(
                f"spatial.left_zone_end ({self._left_end}) must be strictly "
                f"less than spatial.right_zone_start ({self._right_start}). "
                f"Check config.yaml — the center zone would be empty or negative."
            )

        logger.debug(
            "SpatialReasoner created — LEFT:[0.0, %.2f)  "
            "CENTER:[%.2f, %.2f)  RIGHT:[%.2f, 1.0]",
            self._left_end,
            self._left_end, self._right_start,
            self._right_start,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def assign_direction(self, fused_objects: list) -> List[DirectedObject]:
        """
        Add direction labels to a list of FusedObjects.

        Creates one DirectedObject per FusedObject with all existing fields
        preserved and a 'direction' field added. The original FusedObjects
        are not modified.

        Args:
            fused_objects: List[FusedObject] from DepthFusion.fuse().
                           May be empty.

        Returns:
            List[DirectedObject] in the same order as fused_objects.
            Empty list if fused_objects is empty.
        """
        if not fused_objects:
            return []

        directed: List[DirectedObject] = []

        for obj in fused_objects:
            direction = self.classify_direction(obj.center_x)

            directed.append(DirectedObject(
                # Pass all FusedObject fields through unchanged
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
                raw_depth=obj.raw_depth,
                proximity=obj.proximity,
                priority=obj.priority,
                depth_samples=obj.depth_samples,
                depth_age=obj.depth_age,
                # Phase 8 addition
                direction=direction,
            ))

        logger.debug(
            "SpatialReasoner.assign_direction(): %d objects",
            len(directed)
        )

        return directed

    def classify_direction(self, center_x: float) -> str:
        """
        Classify a normalised horizontal position into a direction zone.

        This is a pure function — same input always produces same output.
        Can be called directly for testing or single-value lookups.

        Zone rules (half-open intervals):
            LEFT   : center_x <  left_zone_end           [0.0, left_end)
            CENTER : left_zone_end <= center_x
                     < right_zone_start                   [left_end, right_start)
            RIGHT  : center_x >= right_zone_start         [right_start, 1.0]

        Boundary values (deterministic):
            center_x = left_zone_end (default 0.35):
                → CENTER (not LEFT)
            center_x = right_zone_start (default 0.65):
                → RIGHT (not CENTER)

        Out-of-range values (clamped before comparison):
            center_x < 0.0  → treated as 0.0 → LEFT
            center_x > 1.0  → treated as 1.0 → RIGHT

        Invalid values:
            center_x is NaN or infinite → UNKNOWN (logged as warning)

        Args:
            center_x: Normalised horizontal position (0.0 = left, 1.0 = right).

        Returns:
            Direction.LEFT, Direction.CENTER, Direction.RIGHT,
            or Direction.UNKNOWN.
        """
        # Guard: non-finite values
        if not math.isfinite(center_x):
            logger.warning(
                "SpatialReasoner.classify_direction(): "
                "center_x=%s is not finite — returning UNKNOWN",
                center_x,
            )
            return Direction.UNKNOWN

        # Clamp to [0.0, 1.0] so out-of-range values fall into a valid zone
        cx = max(0.0, min(1.0, center_x))

        # Half-open interval classification
        if cx < self._left_end:
            return Direction.LEFT

        if cx < self._right_start:
            return Direction.CENTER

        return Direction.RIGHT

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def left_zone_end(self) -> float:
        """Upper boundary of the LEFT zone (exclusive). Default: 0.35."""
        return self._left_end

    @property
    def right_zone_start(self) -> float:
        """Lower boundary of the RIGHT zone (inclusive). Default: 0.65."""
        return self._right_start

    @property
    def center_zone_width(self) -> float:
        """Width of the CENTER zone as a fraction of the frame. Default: 0.30."""
        return self._right_start - self._left_end

    def __repr__(self) -> str:
        return (
            f"SpatialReasoner("
            f"LEFT:[0, {self._left_end:.2f})  "
            f"CENTER:[{self._left_end:.2f}, {self._right_start:.2f})  "
            f"RIGHT:[{self._right_start:.2f}, 1])"
        )
