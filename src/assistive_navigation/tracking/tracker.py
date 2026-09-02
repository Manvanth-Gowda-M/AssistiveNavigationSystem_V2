"""
Object Tracker Module
=====================
Assigns stable track IDs to detected objects across video frames using
ByteTrack (accessed directly via the Ultralytics API).

Phase: 5
Status: IMPLEMENTED

What tracking solves:
    Without a tracker, every frame is independent. Frame 100 and frame 101
    might both detect a chair, but the system has no way of knowing they are
    the SAME chair. The tracker solves this by assigning a persistent integer
    ID (track_id) to each object across frames.

    This makes later phases possible:
    - Temporal confirmation (Phase 10): "this is the 8th consecutive frame
      that Object 42 (chair) has been detected"
    - Depth smoothing (Phase 7): average depth values over time for one
      specific tracked object

How ByteTrack works (beginner summary):
    1. A Kalman filter predicts where each tracked object will be next frame.
    2. New detections are matched to predictions using IoU (box overlap).
    3. High-confidence detections are matched first; low-confidence ones are
       used in a second pass to recover briefly occluded tracks.
    4. Unmatched detections start new tracks.
    5. Tracks with no match for MAX_MISSED_FRAMES are dropped.

Architecture decision — direct ByteTrack API:
    We call BYTETracker.update() directly rather than model.track(), because
    model.track() re-runs YOLO inference internally — we don't want that.
    We already have filtered detections from ObjectDetector + DetectionFilter.
    The direct API accepts a DetectionAdapter wrapping our Detection list.

    BYTETracker.update() output: numpy array (N, 8) with columns:
    [x1, y1, x2, y2, track_id, confidence, class_id, index]

Phase 4 implication:
    The empty-scene test found ~203 FP/min from a static background
    (a room element misclassified as `person`). This static background will
    receive a persistent track_id immediately and maintain it indefinitely.
    The tracker CANNOT distinguish a real moving person from a static
    background pattern. Phase 10 (temporal confirmation + position stability)
    and Phase 15 (FP reduction) are the designated mitigations.

Usage:
    from assistive_navigation.tracking.tracker import ObjectTracker

    config = load_config()
    tracker = ObjectTracker(config)
    # No load() needed — ByteTrack has no model file.

    # Each frame:
    tracked = tracker.update(filtered.accepted, frame_width, frame_height)
    for obj in tracked:
        print(obj.track_id, obj.class_name, obj.age, obj.stability)
"""

import logging
import yaml
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TrackedObject dataclass — the public output of this module
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """
    One tracked object in the current frame.

    Fields:
        track_id    : Unique integer ID, stable across frames as long as the
                      object is continuously tracked. A new ID is assigned
                      when a track is lost and re-detected.
        class_id    : COCO class ID (0=person, 56=chair, etc.)
        class_name  : Human-readable name ("person", "chair", etc.)
        confidence  : YOLO detection confidence at this frame (0.0–1.0).
                      High confidence does NOT guarantee correct detection.
        bbox        : (x1, y1, x2, y2) absolute pixel coordinates.
        bbox_norm   : (x1, y1, x2, y2) normalised to 0.0–1.0.
        center_x    : Normalised horizontal centre (0.0=left, 1.0=right).
        center_y    : Normalised vertical centre (0.0=top, 1.0=bottom).
        age         : Total frames this track has existed. Increments every
                      update(), including frames where no detection matched.
        last_seen   : Frames since last matched detection. 0 = matched this
                      frame. >0 = coasting on Kalman prediction.
        stability   : matched_frames / total_age (0.0–1.0).
                      Higher = more consistently detected.
        area_norm   : Bbox area as fraction of total frame area.
        frame_width : Frame width in pixels (for reference).
        frame_height: Frame height in pixels (for reference).
    """
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

    def __repr__(self) -> str:
        return (
            f"TrackedObject(id={self.track_id}, class={self.class_name}, "
            f"conf={self.confidence:.2f}, age={self.age}, "
            f"last_seen={self.last_seen}, stability={self.stability:.2f})"
        )


# ---------------------------------------------------------------------------
# DetectionAdapter — wraps our Detection list into the format BYTETracker needs
# ---------------------------------------------------------------------------

class _DetectionAdapter:
    """
    Adapts a list of Detection objects into a subscriptable array-like
    object that BYTETracker.update() can consume.

    BYTETracker expects an object with:
      .xywh  : (N, 4) float32 array — [cx, cy, w, h] in pixels
      .conf  : (N,)   float32 array — confidence scores
      .cls   : (N,)   float32 array — class IDs
    And it must support boolean/integer indexing (result[mask]).

    We also need to carry xyxy and class_id for later lookup, since
    BYTETracker returns track IDs but not the original class names.
    """

    def __init__(self, detections: list) -> None:
        """
        Args:
            detections: List of Detection objects from DetectionFilter.accepted
        """
        n = len(detections)
        if n == 0:
            self._xywh  = np.zeros((0, 4), dtype=np.float32)
            self._conf  = np.zeros(0, dtype=np.float32)
            self._cls   = np.zeros(0, dtype=np.float32)
            self._xyxy  = np.zeros((0, 4), dtype=np.float32)
            self._names = []
            return

        xyxy_arr = np.array([list(d.bbox) for d in detections], dtype=np.float32)
        x1, y1, x2, y2 = xyxy_arr[:, 0], xyxy_arr[:, 1], xyxy_arr[:, 2], xyxy_arr[:, 3]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = x2 - x1
        h  = y2 - y1

        self._xywh  = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        self._conf  = np.array([d.confidence for d in detections], dtype=np.float32)
        self._cls   = np.array([float(d.class_id) for d in detections], dtype=np.float32)
        self._xyxy  = xyxy_arr
        self._names = [d.class_name for d in detections]

    # Properties required by BYTETracker
    @property
    def xywh(self) -> np.ndarray:
        return self._xywh

    @property
    def conf(self) -> np.ndarray:
        return self._conf

    @property
    def cls(self) -> np.ndarray:
        return self._cls

    def __len__(self) -> int:
        return len(self._conf)

    def __getitem__(self, idx) -> "_DetectionAdapter":
        """Support boolean and integer indexing (required by BYTETracker)."""
        new = _DetectionAdapter.__new__(_DetectionAdapter)
        new._xywh  = self._xywh[idx]
        new._conf  = self._conf[idx]
        new._cls   = self._cls[idx]
        new._xyxy  = self._xyxy[idx]
        new._names = (
            [self._names[i] for i in np.flatnonzero(idx)]
            if isinstance(idx, np.ndarray) and idx.dtype == bool
            else [self._names[i] for i in (np.array([idx]) if np.isscalar(idx) else idx)]
        )
        # Ensure 2D
        if new._xywh.ndim == 1:
            new._xywh = new._xywh[None]
            new._xyxy = new._xyxy[None]
        if new._conf.ndim == 0:
            new._conf = new._conf[None]
            new._cls  = new._cls[None]
        return new


# ---------------------------------------------------------------------------
# Internal per-track state
# ---------------------------------------------------------------------------

@dataclass
class _TrackState:
    """Mutable per-track history. Not exposed outside this module."""
    track_id:       int
    class_id:       int
    class_name:     str
    total_age:      int = 0    # frames since first seen
    matched_frames: int = 0    # frames where a detection was matched
    last_seen:      int = 0    # frames since last matched detection


# ---------------------------------------------------------------------------
# ObjectTracker — the public class
# ---------------------------------------------------------------------------

class ObjectTracker:
    """
    Wraps ByteTrack to assign stable track IDs to accepted detections.

    Call update() once per frame with the list of accepted Detection objects
    from DetectionFilter. The tracker returns TrackedObject instances — one
    per active track.

    No model file is required. ByteTrack is a pure algorithm (Kalman filter
    + Hungarian algorithm) with no neural network weights.

    Args:
        config (dict): Full configuration dictionary from config.yaml.
                       Uses the 'tracking' section.
    """

    def __init__(self, config: dict) -> None:
        trk_cfg = config.get("tracking", {})

        self._tracker_type: str  = trk_cfg.get("tracker", "bytetrack")
        self._max_missed:   int  = int(trk_cfg.get("max_missed_frames", 30))
        self._min_age:      int  = int(trk_cfg.get("min_track_age", 5))
        self._iou_threshold: float = float(trk_cfg.get("iou_threshold", 0.3))

        # Internal state registry: track_id (int) -> _TrackState
        self._track_states: Dict[int, _TrackState] = {}

        # Frame counter
        self._frame_number: int = 0

        # ByteTrack instance (initialised on first update())
        self._bt = None
        self._bt_args = None
        self._is_ready: bool = False

        # COCO class names — populated from first detection batch
        self._class_names: Dict[int, str] = {}

        logger.debug(
            "ObjectTracker created — type=%s, max_missed=%d, min_age=%d",
            self._tracker_type, self._max_missed, self._min_age
        )

        # Initialise ByteTrack immediately
        self._init_bytetrack()

    def _init_bytetrack(self) -> None:
        """
        Initialise the ByteTrack algorithm with settings from config.

        ByteTrack parameters (loaded from bundled bytetrack.yaml, then
        overridden with our config values):
          track_buffer : mirrors our max_missed_frames
          match_thresh : mirrors our iou_threshold
        """
        try:
            import ultralytics
            from ultralytics.trackers.byte_tracker import BYTETracker

            # Load the bundled YAML defaults
            cfg_path = (
                Path(ultralytics.__file__).parent
                / "cfg" / "trackers" / "bytetrack.yaml"
            )
            cfg = yaml.safe_load(cfg_path.read_text())

            # Override with our config values
            cfg["track_buffer"] = self._max_missed
            cfg["match_thresh"] = self._iou_threshold

            self._bt_args = SimpleNamespace(**cfg)
            self._bt = BYTETracker(self._bt_args)
            self._is_ready = True

            logger.info(
                "ByteTrack initialised — track_buffer=%d, match_thresh=%.2f",
                self._max_missed, self._iou_threshold
            )

        except Exception as e:
            logger.error("Failed to initialise ByteTrack: %s", e)
            raise RuntimeError(
                f"ByteTrack initialisation failed: {e}\n"
                f"Ensure ultralytics is installed: pip install ultralytics"
            ) from e

    # -----------------------------------------------------------------------
    # Main update method
    # -----------------------------------------------------------------------

    def update(
        self,
        accepted_detections: list,   # List[Detection] from FilterResult.accepted
        frame_width: int,
        frame_height: int,
    ) -> List[TrackedObject]:
        """
        Process one frame's accepted detections and return tracked objects.

        Call this once per frame AFTER getting accepted detections from
        DetectionFilter.

        Args:
            accepted_detections: List[Detection] — FilterResult.accepted.
                                  These are already filtered by confidence
                                  and class. May be empty.
            frame_width:         Frame width in pixels.
            frame_height:        Frame height in pixels.

        Returns:
            List[TrackedObject]: Currently active tracks, sorted by track_id.
            Returns [] if there are no active tracks.

        Raises:
            RuntimeError: If ByteTrack was not initialised successfully.
        """
        if not self._is_ready or self._bt is None:
            raise RuntimeError(
                "ObjectTracker is not ready. ByteTrack failed to initialise."
            )

        self._frame_number += 1

        # Build class name cache from incoming detections
        for det in accepted_detections:
            self._class_names[det.class_id] = det.class_name

        # Handle empty detection frame
        if len(accepted_detections) == 0:
            self._age_all_tracks()
            self._drop_stale_tracks()
            return []

        # Wrap detections in adapter
        adapter = _DetectionAdapter(accepted_detections)

        # Run ByteTrack
        try:
            bt_result = self._bt.update(adapter, img=None)
        except Exception as e:
            logger.warning(
                "ByteTrack update error on frame %d: %s — returning []",
                self._frame_number, e
            )
            return []

        # bt_result: numpy (N, 8) — [x1, y1, x2, y2, track_id, conf, cls_id, idx]
        tracked_objects: List[TrackedObject] = []
        active_ids_this_frame: set = set()

        if bt_result is not None and len(bt_result) > 0:
            for row in bt_result:
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                tid    = int(row[4])
                conf   = float(row[5])
                cls_id = int(row[6])

                # Clamp to frame bounds
                x1 = max(0.0, min(x1, frame_width))
                y1 = max(0.0, min(y1, frame_height))
                x2 = max(0.0, min(x2, frame_width))
                y2 = max(0.0, min(y2, frame_height))

                cls_name = self._class_names.get(cls_id, f"class_{cls_id}")

                # Normalised coordinates
                x1n = x1 / frame_width
                y1n = y1 / frame_height
                x2n = x2 / frame_width
                y2n = y2 / frame_height
                cx  = (x1n + x2n) / 2.0
                cy  = (y1n + y2n) / 2.0
                area_norm = ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)

                # Update internal state
                if tid not in self._track_states:
                    self._track_states[tid] = _TrackState(
                        track_id=tid,
                        class_id=cls_id,
                        class_name=cls_name,
                    )

                state = self._track_states[tid]
                state.total_age      += 1
                state.matched_frames += 1
                state.last_seen       = 0
                state.class_id        = cls_id
                state.class_name      = cls_name

                stability = (
                    state.matched_frames / state.total_age
                    if state.total_age > 0 else 1.0
                )

                tracked_objects.append(TrackedObject(
                    track_id=tid,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    bbox_norm=(x1n, y1n, x2n, y2n),
                    center_x=cx,
                    center_y=cy,
                    age=state.total_age,
                    last_seen=0,
                    stability=stability,
                    area_norm=area_norm,
                    frame_width=frame_width,
                    frame_height=frame_height,
                ))
                active_ids_this_frame.add(tid)

        # Age out tracks that were NOT matched this frame
        for tid, state in self._track_states.items():
            if tid not in active_ids_this_frame:
                state.total_age  += 1
                state.last_seen  += 1

        self._drop_stale_tracks()

        logger.debug(
            "Tracker frame %d: %d tracks active",
            self._frame_number, len(tracked_objects)
        )

        return sorted(tracked_objects, key=lambda t: t.track_id)

    # -----------------------------------------------------------------------
    # State helpers
    # -----------------------------------------------------------------------

    def _age_all_tracks(self) -> None:
        """Increment age and last_seen for all tracks (no detections this frame)."""
        for state in self._track_states.values():
            state.total_age += 1
            state.last_seen += 1

    def _drop_stale_tracks(self) -> None:
        """Remove tracks lost for more than max_missed_frames frames."""
        to_drop = [
            tid for tid, st in self._track_states.items()
            if st.last_seen > self._max_missed
        ]
        for tid in to_drop:
            logger.debug(
                "Dropping stale track %d (last_seen=%d > max=%d)",
                tid, self._track_states[tid].last_seen, self._max_missed
            )
            del self._track_states[tid]

    def reset(self) -> None:
        """
        Clear all track state and restart the ByteTrack algorithm.

        Use this if you want to re-run the tracker from scratch without
        creating a new ObjectTracker instance (e.g. after pausing the camera).
        """
        self._track_states.clear()
        self._frame_number = 0
        self._class_names.clear()
        # Re-initialise ByteTrack to clear its internal state
        self._bt = None
        self._is_ready = False
        self._init_bytetrack()
        logger.info("ObjectTracker reset.")

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if ByteTrack initialised successfully."""
        return self._is_ready

    @property
    def active_track_count(self) -> int:
        """Number of tracks currently in the state registry."""
        return len(self._track_states)

    @property
    def frame_number(self) -> int:
        """Total number of update() calls made."""
        return self._frame_number

    @property
    def tracker_type(self) -> str:
        """Configured tracker name."""
        return self._tracker_type

    def get_track_age(self, track_id: int) -> Optional[int]:
        """Return the total_age of a specific track, or None if not found."""
        st = self._track_states.get(track_id)
        return st.total_age if st else None

    def get_track_stability(self, track_id: int) -> Optional[float]:
        """Return the stability (0–1) of a specific track, or None if not found."""
        st = self._track_states.get(track_id)
        if st is None or st.total_age == 0:
            return None
        return st.matched_frames / st.total_age

    def __repr__(self) -> str:
        return (
            f"ObjectTracker(type={self._tracker_type}, "
            f"frame={self._frame_number}, "
            f"active_tracks={self.active_track_count}, "
            f"ready={self._is_ready})"
        )
