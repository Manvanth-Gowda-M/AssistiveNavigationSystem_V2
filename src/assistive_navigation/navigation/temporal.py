"""
Temporal Confirmation Module
==============================
Maintains per-track state across pipeline updates to gate which objects
are temporally reliable enough for downstream alerting.

Phase: 10
Status: IMPLEMENTED

SAFETY NOTE — ALL THRESHOLD VALUES ARE STARTING ESTIMATES:
    confirmation_frames=8, min_area_norm=0.02, miss_tolerance=3,
    position_stability_threshold=0.15, min_nav_score_for_confirmation=0.05
    are documented starting values that have NOT been validated against
    real-world navigation safety requirements.

    In particular: min_area_norm=0.02 is INTENTIONALLY CONSERVATIVE.
    The Phase 4 evaluation identified a persistent background-element FP
    with area_norm≈0.08 that runs for 40-110 consecutive frames.
    With min_area_norm=0.02, this FP WILL be confirmed after 8 frames.
    This is acknowledged. Phase 15 must raise min_area_norm empirically
    after measuring genuine small-object area distributions.

    Do NOT treat confirmed=True as a guaranteed real obstacle.
    This system is a research prototype.

What this module does:
    - Accepts List[ScoredObject] from Phase 9 (NavigationPriorityEngine)
    - Maintains per-track_id temporal state across calls
    - Applies five independent confirmation gates each update
    - Returns List[ConfirmedObject] with temporal metadata attached
    - Expires state for tracks that have disappeared too long

What this module does NOT do:
    - Does NOT produce audio or alerts (Phase 12)
    - Does NOT modify detection, tracking, depth, fusion, or scoring
    - Does NOT guarantee absence of false positives
    - Does NOT handle speech cooldowns

Five confirmation gates (must ALL pass for count to progress):
    Gate 1: min_area_norm      — bbox must exceed minimum size
    Gate 2: confirmation_frames — consecutive passing count threshold
    Gate 3: class_stability    — class label must be consistent
    Gate 4: position_stability — bbox center must not jump wildly
    Gate 5: min_nav_score      — nav_score must exceed permissive floor

State machine per track_id:
    NEW        : Just appeared, no consecutive count yet.
    CANDIDATE  : Passing gates, count > 0 but < confirmation_frames.
    CONFIRMED  : Reached confirmation_frames consecutive passing frames.
    LOST       : Not seen for 1..miss_tolerance frames (count paused).

    Transitions:
      NEW/CANDIDATE/CONFIRMED  --[all gates pass]-→  count++
      count reaches confirmation_frames             → CONFIRMED
      any soft gate fails                           → count resets to 0, state→CANDIDATE
      Gate 1 fails (area too small)                 → hard fail, state→NEW, count=0
      object missing 1..miss_tolerance updates      → LOST  (count frozen)
      object missing > miss_tolerance updates       → state deleted (fresh start next)
      LOST object reappears within miss_tolerance   → resumes previous state/count

Data flow:
    ScoredObject (from Phase 9)
        ↓ TemporalConfirmationFilter.update()
    ConfirmedObject  (ScoredObject + confirmation_state, consecutive_count,
                      is_confirmed, gate_failures)

Usage:
    from assistive_navigation.navigation.temporal import TemporalConfirmationFilter
    from assistive_navigation.utils.config_loader import load_config

    config = load_config()
    tcf = TemporalConfirmationFilter(config)

    # Each frame — pass List[ScoredObject] from Phase 9:
    confirmed_objects = tcf.update(scored_objects)
    for obj in confirmed_objects:
        if obj.is_confirmed:
            # eligible for alerting (Phase 12 decides whether to speak)
            pass
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confirmation state constants
# ---------------------------------------------------------------------------

class ConfirmationState:
    """
    Temporal confirmation states for a tracked object.

    NEW       : Track just appeared. Not yet accumulating toward confirmation.
    CANDIDATE : Accumulating consecutive passing observations.
                count > 0 but has not yet reached confirmation_frames.
    CONFIRMED : Has reached confirmation_frames consecutive passing observations.
                Eligible for alerting (Phase 12 decides when to speak).
    LOST      : Object disappeared for 1..miss_tolerance frames.
                Count is frozen. If it reappears soon, count resumes.
    """
    NEW       = "NEW"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    LOST      = "LOST"


# ---------------------------------------------------------------------------
# Gate name constants — used in gate_failures list
# ---------------------------------------------------------------------------

class GateName:
    AREA        = "area_too_small"
    POSITION    = "position_jump"
    CLASS       = "class_instability"
    NAV_SCORE   = "nav_score_too_low"


# ---------------------------------------------------------------------------
# Internal per-track state record
# ---------------------------------------------------------------------------

@dataclass
class _TrackTemporalState:
    """
    Internal state record for one track_id.
    Not exposed in the public API.

    Fields:
        track_id          : The ByteTrack track ID this state belongs to.
        state             : Current ConfirmationState string.
        consecutive_count : Consecutive frames that passed all gates.
                            Resets to 0 on any soft gate failure.
                            Frozen (not reset) when state is LOST.
        missed_count      : Consecutive updates where this track_id was absent.
                            Reset to 0 when track reappears.
        class_history     : Rolling deque of recent class labels.
        last_center_x     : center_x from previous passing update (or None).
        last_center_y     : center_y from previous passing update (or None).
        last_seen_state   : State just before going LOST (to restore on return).
        last_seen_count   : consecutive_count just before going LOST.
    """
    track_id:          int
    state:             str   = ConfirmationState.NEW
    consecutive_count: int   = 0
    missed_count:      int   = 0
    class_history:     deque = field(default_factory=deque)
    last_center_x:     Optional[float] = None
    last_center_y:     Optional[float] = None
    last_seen_state:   str   = ConfirmationState.NEW
    last_seen_count:   int   = 0


# ---------------------------------------------------------------------------
# ConfirmedObject dataclass — public output of this module
# ---------------------------------------------------------------------------

@dataclass
class ConfirmedObject:
    """
    A ScoredObject with temporal confirmation metadata added.

    All fields from ScoredObject (Phase 9) are preserved unchanged.
    Four fields are added by Phase 10.

    Fields from ScoredObject (passed through):
        track_id, class_id, class_name, confidence,
        bbox, bbox_norm, center_x, center_y,
        age, last_seen, stability, area_norm,
        frame_width, frame_height,
        raw_depth, proximity, priority, depth_samples, depth_age,
        direction, nav_score

    Fields added by Phase 10:
        confirmation_state : "NEW" | "CANDIDATE" | "CONFIRMED" | "LOST"
                             Current state in the temporal state machine.
        consecutive_count  : Number of consecutive updates that passed all gates.
                             Resets on gate failure; frozen when LOST.
        is_confirmed       : True iff confirmation_state == CONFIRMED.
                             This is the primary flag for Phase 12 to check.
                             True does NOT guarantee the detection is correct.
        gate_failures      : List of gate name strings that failed this update.
                             Empty if all gates passed.
                             Useful for debugging and Phase 15 tuning.
    """
    # ── From ScoredObject (unchanged) ─────────────────────────────────────
    track_id:      int
    class_id:      int
    class_name:    str
    confidence:    float
    bbox:          tuple
    bbox_norm:     tuple
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
    direction:     str
    nav_score:     float

    # ── Added by Phase 10 ─────────────────────────────────────────────────
    confirmation_state: str         # ConfirmationState constant
    consecutive_count:  int         # consecutive passing observations
    is_confirmed:       bool        # True iff state == CONFIRMED
    gate_failures:      List[str]   # which gates failed this update

    def __repr__(self) -> str:
        return (
            f"ConfirmedObject(id={self.track_id}, class={self.class_name}, "
            f"state={self.confirmation_state}, count={self.consecutive_count}, "
            f"confirmed={self.is_confirmed}, score={self.nav_score:.3f})"
        )


# ---------------------------------------------------------------------------
# TemporalConfirmationFilter
# ---------------------------------------------------------------------------

class TemporalConfirmationFilter:
    """
    Maintains per-track temporal state and gates objects toward confirmation.

    Call update() once per pipeline cycle (after Phase 9 scoring).
    Returns ConfirmedObject for every input ScoredObject. The filter never
    drops objects — it attaches state metadata so Phase 12 can decide
    whether to act.

    IMPORTANT: All threshold parameters are starting estimates requiring
    Phase 15 empirical calibration. See module docstring for details.

    Args:
        config (dict): Full configuration from config.yaml.
                       Uses the 'temporal' section.
    """

    def __init__(self, config: dict) -> None:
        t = config.get("temporal", {})

        self._confirmation_frames:       int   = int(t.get("confirmation_frames",       8))
        self._miss_tolerance:            int   = int(t.get("miss_tolerance",            3))
        self._class_stability_threshold: float = float(t.get("class_stability_threshold", 0.8))
        self._class_history_window:      int   = int(t.get("class_history_window",      10))
        self._position_stability_thresh: float = float(t.get("position_stability_threshold", 0.15))
        self._min_area_norm:             float = float(t.get("min_area_norm",            0.02))
        self._min_nav_score:             float = float(t.get("min_nav_score_for_confirmation", 0.05))

        # Per-track state registry: track_id (int) → _TrackTemporalState
        self._states: Dict[int, _TrackTemporalState] = {}

        # Update cycle counter
        self._update_count: int = 0

        logger.debug(
            "TemporalConfirmationFilter created — "
            "confirmation_frames=%d, miss_tolerance=%d, "
            "min_area_norm=%.3f (STARTING ESTIMATE), "
            "position_thresh=%.2f, class_thresh=%.2f, min_score=%.2f",
            self._confirmation_frames, self._miss_tolerance,
            self._min_area_norm, self._position_stability_thresh,
            self._class_stability_threshold, self._min_nav_score,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(self, scored_objects: list) -> List[ConfirmedObject]:
        """
        Process one pipeline cycle's ScoredObjects and return ConfirmedObjects.

        For every input ScoredObject:
          1. Evaluate all five gates.
          2. Update the per-track state based on gate results.
          3. Wrap in a ConfirmedObject with the state metadata attached.

        Objects NOT present in this update (tracks that disappeared):
          - Their state transitions toward LOST then EXPIRED.
          - They do NOT appear in the output — only present objects are returned.
          - If a track reappears within miss_tolerance updates, its state resumes.

        Args:
            scored_objects: List[ScoredObject] from NavigationPriorityEngine.score().
                            May be empty.

        Returns:
            List[ConfirmedObject] in the same order as scored_objects.
            Empty list if scored_objects is empty.
        """
        self._update_count += 1

        # Build set of active track IDs this update
        active_ids = {obj.track_id for obj in scored_objects}

        # Age out tracks that are not present this update
        self._age_absent_tracks(active_ids)

        results: List[ConfirmedObject] = []

        for obj in scored_objects:
            confirmed = self._process_object(obj)
            results.append(confirmed)

        return results

    # -----------------------------------------------------------------------
    # Per-object processing
    # -----------------------------------------------------------------------

    def _process_object(self, obj) -> "ConfirmedObject":
        """
        Apply the five gates to one ScoredObject and update its track state.

        Returns a ConfirmedObject with updated temporal metadata.
        """
        tid = obj.track_id

        # Get or create state for this track
        if tid not in self._states:
            state = _TrackTemporalState(
                track_id=tid,
                class_history=deque(maxlen=self._class_history_window),
            )
            self._states[tid] = state
        else:
            state = self._states[tid]

        # If state was LOST and track just reappeared, restore saved state
        if state.state == ConfirmationState.LOST:
            state.state             = state.last_seen_state
            state.consecutive_count = state.last_seen_count
            state.missed_count      = 0
            logger.debug(
                "Track %d reappeared after LOST — resuming %s count=%d",
                tid, state.state, state.consecutive_count
            )

        # Evaluate all five gates
        failures = self._evaluate_gates(obj, state)

        # Update class history regardless of gate results
        state.class_history.append(obj.class_name)

        # Apply gate results to state
        if failures:
            if GateName.AREA in failures:
                # Gate 1 is a hard fail: reset to NEW regardless of history
                state.state             = ConfirmationState.NEW
                state.consecutive_count = 0
                state.last_center_x     = None
                state.last_center_y     = None
                logger.debug(
                    "Track %d: area gate HARD FAIL (area_norm=%.4f < min=%.4f) → NEW",
                    tid, obj.area_norm, self._min_area_norm
                )
            else:
                # Soft gate failure: reset count but stay CANDIDATE (not NEW)
                state.consecutive_count = 0
                if state.state != ConfirmationState.CONFIRMED:
                    state.state = ConfirmationState.CANDIDATE
                # On soft failure: do NOT update last position
                # (position is now the jump-destination, don't anchor there)
                logger.debug(
                    "Track %d: soft gate(s) failed %s → count reset to 0",
                    tid, failures
                )
        else:
            # All gates passed — increment count
            state.consecutive_count += 1
            state.missed_count       = 0

            # Update position anchor
            state.last_center_x = obj.center_x
            state.last_center_y = obj.center_y

            # Transition to appropriate state
            if state.consecutive_count >= self._confirmation_frames:
                state.state = ConfirmationState.CONFIRMED
            elif state.consecutive_count > 0:
                state.state = ConfirmationState.CANDIDATE

            logger.debug(
                "Track %d: all gates pass → %s (count=%d/%d)",
                tid, state.state,
                state.consecutive_count, self._confirmation_frames,
            )

        return ConfirmedObject(
            # Pass through all ScoredObject fields
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
            direction=obj.direction,
            nav_score=obj.nav_score,
            # Phase 10 additions
            confirmation_state=state.state,
            consecutive_count=state.consecutive_count,
            is_confirmed=(state.state == ConfirmationState.CONFIRMED),
            gate_failures=failures,
        )

    # -----------------------------------------------------------------------
    # Five gates
    # -----------------------------------------------------------------------

    def _evaluate_gates(
        self, obj, state: _TrackTemporalState
    ) -> List[str]:
        """
        Evaluate all five gates for one object update.

        Returns a list of gate name strings for each gate that failed.
        An empty list means all gates passed.

        Gate evaluation order matters: Gate 1 (area) is evaluated first
        because it is a hard fail. The others are soft fails.
        """
        failures: List[str] = []

        # ── Gate 1: Minimum bounding-box area ─────────────────────────────
        # Rejects detections that are too small to be reliable navigation objects.
        # STARTING ESTIMATE: min_area_norm=0.02. Phase 15 must calibrate.
        # NOTE: The Phase 4 background-element FP (area_norm≈0.08) PASSES
        # this gate with the current default. This is acknowledged and expected.
        if obj.area_norm < self._min_area_norm:
            failures.append(GateName.AREA)
            # Return immediately — area failure is hard, no need to check others
            return failures

        # ── Gate 5: Minimum nav_score ──────────────────────────────────────
        # Very permissive floor. Objects scoring below 0.05 are unusual
        # (requires all components to be minimal simultaneously).
        # STARTING ESTIMATE: 0.05.
        if obj.nav_score < self._min_nav_score:
            failures.append(GateName.NAV_SCORE)

        # ── Gate 3: Class label stability ─────────────────────────────────
        # Check whether the class label has been consistent in recent history.
        # On the very first observation (empty history), this gate passes.
        if len(state.class_history) >= 2:
            window = list(state.class_history)
            recent_class = obj.class_name
            matching = sum(1 for c in window if c == recent_class)
            frac = matching / len(window)
            if frac < self._class_stability_threshold:
                failures.append(GateName.CLASS)

        # ── Gate 4: Position stability ─────────────────────────────────────
        # Compare current bbox centre to the last PASSING update's centre.
        # If there was no previous passing update, this gate passes.
        if state.last_center_x is not None and state.last_center_y is not None:
            dx = obj.center_x - state.last_center_x
            dy = obj.center_y - state.last_center_y
            displacement = math.sqrt(dx * dx + dy * dy)
            if displacement > self._position_stability_thresh:
                failures.append(GateName.POSITION)

        return failures

    # -----------------------------------------------------------------------
    # Absent track aging
    # -----------------------------------------------------------------------

    def _age_absent_tracks(self, active_ids: set) -> None:
        """
        For tracks not present in active_ids, increment missed_count.
        Tracks exceeding miss_tolerance are removed (expired).
        """
        to_expire = []
        for tid, state in self._states.items():
            if tid in active_ids:
                continue  # present this update, handled in _process_object

            if state.state == ConfirmationState.LOST:
                state.missed_count += 1
                if state.missed_count > self._miss_tolerance:
                    to_expire.append(tid)
                    logger.debug(
                        "Track %d EXPIRED (missed=%d > tolerance=%d)",
                        tid, state.missed_count, self._miss_tolerance
                    )
            else:
                # First absence: save state and go LOST
                state.last_seen_state = state.state
                state.last_seen_count = state.consecutive_count
                state.state           = ConfirmationState.LOST
                state.missed_count    = 1
                logger.debug(
                    "Track %d → LOST (first miss, was %s count=%d)",
                    tid, state.last_seen_state, state.last_seen_count
                )

        for tid in to_expire:
            del self._states[tid]

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def reset(self, track_id: Optional[int] = None) -> None:
        """
        Clear temporal state.

        Args:
            track_id: If provided, clear only this track's state.
                      If None, clear all state (full reset).
        """
        if track_id is None:
            self._states.clear()
            self._update_count = 0
            logger.info("TemporalConfirmationFilter: full reset.")
        elif track_id in self._states:
            del self._states[track_id]
            logger.debug("TemporalConfirmationFilter: cleared track %d.", track_id)

    def get_state(self, track_id: int) -> Optional[str]:
        """Return the current ConfirmationState for a track, or None if unknown."""
        s = self._states.get(track_id)
        return s.state if s else None

    def get_consecutive_count(self, track_id: int) -> int:
        """Return the current consecutive count for a track (0 if not tracked)."""
        s = self._states.get(track_id)
        return s.consecutive_count if s else 0

    def get_confirmed_track_ids(self) -> List[int]:
        """Return all track IDs currently in CONFIRMED state."""
        return [
            tid for tid, s in self._states.items()
            if s.state == ConfirmationState.CONFIRMED
        ]

    @property
    def active_track_count(self) -> int:
        """Number of tracks currently in the state registry."""
        return len(self._states)

    @property
    def update_count(self) -> int:
        """Total number of update() calls made."""
        return self._update_count

    @property
    def confirmation_frames(self) -> int:
        """Configured confirmation_frames threshold."""
        return self._confirmation_frames

    @property
    def miss_tolerance(self) -> int:
        """Configured miss_tolerance threshold."""
        return self._miss_tolerance

    @property
    def min_area_norm(self) -> float:
        """Configured minimum bbox area threshold (starting estimate)."""
        return self._min_area_norm

    def __repr__(self) -> str:
        return (
            f"TemporalConfirmationFilter("
            f"conf_frames={self._confirmation_frames}, "
            f"miss_tol={self._miss_tolerance}, "
            f"min_area={self._min_area_norm:.3f}, "
            f"active_tracks={self.active_track_count}, "
            f"updates={self._update_count})"
        )
