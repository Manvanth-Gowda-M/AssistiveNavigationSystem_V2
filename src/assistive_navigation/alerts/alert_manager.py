"""
Alert Manager
=============
Converts confirmed navigation objects into spoken alerts.

Phase: 12
Status: IMPLEMENTED

What this module does:
  - Accepts List[ConfirmedObject] from Phase 10 (TemporalConfirmationFilter)
  - Filters to objects where is_confirmed=True only
  - Selects the most urgent eligible object (highest nav_score)
  - Decides whether to speak based on cooldown, escalation, and change rules
  - Formats a concise navigation message
  - Calls queue.put(text) — never calls TTS directly
  - Returns AlertResult with metadata for logging and testing

What this module does NOT do:
  - Does not detect objects           (Phase 3)
  - Does not track objects            (Phase 5)
  - Does not estimate depth           (Phase 6–7)
  - Does not assign direction         (Phase 8)
  - Does not score priority           (Phase 9)
  - Does not apply temporal gates     (Phase 10)
  - Does not speak directly           (Phase 11 queue does that)
  - Does not tune Phase 4 FP issues   (Phase 15)

Alert eligibility:
  An object is eligible for alerting if and only if:
    1. obj.is_confirmed == True
    2. obj.direction != "UNKNOWN"
    3. obj.nav_score > 0.0

Alert triggers (any one is sufficient):
  - New confirmed object  : track_id has no prior state, or prior state expired
  - Proximity escalation  : proximity moved closer (MEDIUM→CLOSE, CLOSE→VERY_CLOSE)
  - Direction change      : direction changed since last alert
  - Cooldown expired      : cooldown_seconds elapsed since last alert

Suppression (all must hold for silence):
  - Same track, same proximity (not escalated), same direction, within cooldown

Selection:
  - Only one alert per process() call
  - Of all eligible candidates needing an alert, choose highest nav_score
  - This prevents a far low-priority object from overriding a close person

Message formats (from config.yaml alerts.templates):
  new_object : "person ahead"
  standard   : "person ahead, very close"
  escalation : "person ahead, now very close"

SAFETY NOTE:
  All timing values (cooldown_seconds=5.0, reappearance_window_seconds=3.0)
  are starting estimates. They have NOT been validated against navigation
  safety requirements. Phase 15 must empirically evaluate alert latency,
  repeat frequency, and whether important obstacles are missed due to
  suppression. Do not treat these values as safety-optimal.

Usage:
    from assistive_navigation.alerts.alert_manager import AlertManager
    from assistive_navigation.audio.queue import SimpleAudioQueue
    from assistive_navigation.audio.tts import TextToSpeech
    from assistive_navigation.utils.config_loader import load_config

    config = load_config()

    tts = TextToSpeech(config)
    tts.initialise()

    q = SimpleAudioQueue(tts)
    q.start()

    manager = AlertManager(config, q)

    # Each pipeline cycle:
    result = manager.process(confirmed_objects)
    if result.alert_issued:
        print(f"Alert: {result.alert_text}")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proximity ordering for escalation detection
# ---------------------------------------------------------------------------

_PROXIMITY_ORDER: Dict[str, int] = {
    "UNKNOWN":    -1,   # unknown = cannot determine escalation
    "FAR":         0,
    "MEDIUM":      1,
    "CLOSE":       2,
    "VERY_CLOSE":  3,
}


# ---------------------------------------------------------------------------
# AlertResult — returned by process() for logging and testing
# ---------------------------------------------------------------------------

@dataclass
class AlertResult:
    """
    Metadata about what the AlertManager decided this update.

    Returned by AlertManager.process() on every call.
    Does not affect the pipeline — purely for logging and testing.

    Fields:
        alert_issued    : True if an alert was enqueued this update.
        alert_text      : The text enqueued ("" if no alert).
        selected_track  : track_id that triggered the alert (-1 if none).
        trigger_reason  : Why the alert was issued (or "none").
        eligible_count  : Number of confirmed eligible objects this update.
        suppressed_count: Number of eligible objects that were suppressed.
    """
    alert_issued:     bool
    alert_text:       str
    selected_track:   int
    trigger_reason:   str    # "new_object" | "proximity_escalation" |
                              # "direction_change" | "cooldown_expired" | "none"
    eligible_count:   int
    suppressed_count: int

    def __repr__(self) -> str:
        if self.alert_issued:
            return (
                f"AlertResult(issued=True, text={self.alert_text!r}, "
                f"track={self.selected_track}, reason={self.trigger_reason})"
            )
        return (
            f"AlertResult(issued=False, "
            f"eligible={self.eligible_count}, suppressed={self.suppressed_count})"
        )


# ---------------------------------------------------------------------------
# Internal per-track alert state
# ---------------------------------------------------------------------------

@dataclass
class _TrackAlertState:
    """Per-track state maintained by the AlertManager."""
    track_id:           int
    last_proximity:     str
    last_direction:     str
    last_alert_time:    float    # time.monotonic() of last alert
    first_confirmed_at: float    # time.monotonic() when first confirmed
    alert_count:        int = 0


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Converts confirmed navigation objects into spoken navigation alerts.

    Receives List[ConfirmedObject] each pipeline cycle.
    Issues at most one alert per cycle via SimpleAudioQueue.
    Maintains per-track state to suppress redundant speech and detect
    meaningful changes (proximity escalation, direction change).

    IMPORTANT: All timing thresholds are STARTING ESTIMATES.
    Phase 15 must empirically validate these values.

    Args:
        config (dict): Full configuration from config.yaml.
                       Uses the 'alerts', 'spatial', and 'proximity' sections.
        audio_queue:   A SimpleAudioQueue instance that has been started.
                       If None, alerts are logged but not spoken.
    """

    def __init__(self, config: dict, audio_queue=None) -> None:
        alerts_cfg  = config.get("alerts",    {})
        spatial_cfg = config.get("spatial",   {})
        prox_cfg    = config.get("proximity",  {})

        # Timing parameters (starting estimates — require Phase 15 validation)
        self._cooldown_seconds:       float = float(alerts_cfg.get("cooldown_seconds",            5.0))
        self._reappearance_window_s:  float = float(alerts_cfg.get("reappearance_window_seconds", 3.0))

        # Message templates
        templates = alerts_cfg.get("templates", {})
        self._tmpl_standard   = templates.get("standard",   "{object} {direction}, {proximity}")
        self._tmpl_new_object = templates.get("new_object", "{object} {direction}")
        self._tmpl_escalation = templates.get("escalation", "{object} {direction}, now {proximity}")

        # Direction → spoken phrase mapping
        direction_labels = spatial_cfg.get("labels", {})
        self._dir_labels: Dict[str, str] = {
            "CENTER":  direction_labels.get("center",  "ahead"),
            "LEFT":    direction_labels.get("left",    "on your left"),
            "RIGHT":   direction_labels.get("right",   "on your right"),
            "UNKNOWN": "nearby",
        }

        # Proximity → spoken phrase mapping
        prox_labels = prox_cfg.get("labels", {})
        self._prox_labels: Dict[str, str] = {
            "VERY_CLOSE": prox_labels.get("very_close", "very close"),
            "CLOSE":      prox_labels.get("close",      "close"),
            "MEDIUM":     prox_labels.get("medium",     "nearby"),
            "FAR":        "",        # not spoken — direction-only message used
            "UNKNOWN":    "",        # not spoken
        }

        # SimpleAudioQueue (may be None for testing without audio)
        self._queue = audio_queue

        # Per-track alert state: track_id → _TrackAlertState
        self._track_states: Dict[int, _TrackAlertState] = {}

        # Total alerts issued
        self._total_alerts: int = 0

        logger.debug(
            "AlertManager created — cooldown=%.1fs, reappearance_window=%.1fs",
            self._cooldown_seconds, self._reappearance_window_s,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def process(self, confirmed_objects: list) -> AlertResult:
        """
        Evaluate confirmed objects and issue at most one alert.

        Steps:
          1. Age out expired track states.
          2. Filter to eligible confirmed objects.
          3. For each eligible object, determine if an alert is warranted.
          4. Select the highest nav_score candidate.
          5. Format and enqueue the alert text.
          6. Update per-track state.
          7. Return AlertResult.

        Args:
            confirmed_objects: List[ConfirmedObject] from
                               TemporalConfirmationFilter.update().
                               May be empty.

        Returns:
            AlertResult describing what was decided this update.
        """
        now = time.monotonic()

        # Step 1: expire stale track states
        self._expire_stale_states(now)

        # Step 2: filter to eligible confirmed objects
        eligible = [
            obj for obj in confirmed_objects
            if obj.is_confirmed
            and obj.direction != "UNKNOWN"
            and obj.nav_score > 0.0
        ]

        if not eligible:
            return AlertResult(
                alert_issued=False, alert_text="", selected_track=-1,
                trigger_reason="none",
                eligible_count=0, suppressed_count=0,
            )

        # Step 3: determine which eligible objects need an alert
        candidates: List[tuple] = []   # (obj, reason)
        suppressed_count = 0

        for obj in eligible:
            reason = self._should_alert(obj, now)
            if reason:
                candidates.append((obj, reason))
            else:
                suppressed_count += 1

        if not candidates:
            return AlertResult(
                alert_issued=False, alert_text="", selected_track=-1,
                trigger_reason="none",
                eligible_count=len(eligible),
                suppressed_count=suppressed_count,
            )

        # Step 4: select highest nav_score candidate
        selected_obj, reason = max(candidates, key=lambda c: c[0].nav_score)

        # Step 5: format alert text
        alert_text = self._format_alert(selected_obj, reason)

        # Step 6: enqueue
        if alert_text:
            self._enqueue(alert_text)
            self._total_alerts += 1

            # Update track state
            self._update_state(selected_obj, now)

            logger.info(
                "Alert issued: %r (track=%d, reason=%s, score=%.3f)",
                alert_text, selected_obj.track_id, reason, selected_obj.nav_score,
            )

        return AlertResult(
            alert_issued=bool(alert_text),
            alert_text=alert_text,
            selected_track=selected_obj.track_id if alert_text else -1,
            trigger_reason=reason if alert_text else "none",
            eligible_count=len(eligible),
            suppressed_count=suppressed_count,
        )

    # -----------------------------------------------------------------------
    # Alert decision logic
    # -----------------------------------------------------------------------

    def _should_alert(self, obj, now: float) -> Optional[str]:
        """
        Determine whether obj warrants an alert this update.

        Returns a trigger reason string if yes, None if suppressed.

        Trigger reasons (in priority order — first matching wins):
          "new_object"           : track has no prior alert state
          "proximity_escalation" : proximity moved to a closer level
          "direction_change"     : direction changed since last alert
          "cooldown_expired"     : cooldown_seconds elapsed since last alert
          None                   : suppress
        """
        tid = obj.track_id
        state = self._track_states.get(tid)

        # No prior state → new object
        if state is None:
            return "new_object"

        # Check proximity escalation (must be strictly closer)
        new_order = _PROXIMITY_ORDER.get(obj.proximity, -1)
        old_order = _PROXIMITY_ORDER.get(state.last_proximity, -1)
        if new_order > 0 and new_order > old_order:
            return "proximity_escalation"

        # Check direction change
        if obj.direction != state.last_direction:
            return "direction_change"

        # Check cooldown expiry
        if (now - state.last_alert_time) >= self._cooldown_seconds:
            return "cooldown_expired"

        # All suppression conditions hold
        return None

    # -----------------------------------------------------------------------
    # Message formatting
    # -----------------------------------------------------------------------

    def _format_alert(self, obj, reason: str) -> str:
        """
        Format a spoken navigation message for the given object and reason.

        Uses templates from config.yaml where possible.
        Returns "" if the message would be meaningless (e.g. proximity=FAR
        and no proximity label — uses new_object template instead).
        """
        cls_name   = obj.class_name
        dir_label  = self._dir_labels.get(obj.direction, "nearby")
        prox_label = self._prox_labels.get(obj.proximity, "")

        # If no meaningful proximity label, fall back to direction-only
        if not prox_label:
            text = self._tmpl_new_object.format(
                object=cls_name,
                direction=dir_label,
            )
        elif reason == "proximity_escalation":
            text = self._tmpl_escalation.format(
                object=cls_name,
                direction=dir_label,
                proximity=prox_label,
            )
        else:
            text = self._tmpl_standard.format(
                object=cls_name,
                direction=dir_label,
                proximity=prox_label,
            )

        return text.strip()

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    def _update_state(self, obj, now: float) -> None:
        """Record that we just alerted for this track."""
        tid = obj.track_id
        existing = self._track_states.get(tid)
        if existing:
            existing.last_proximity  = obj.proximity
            existing.last_direction  = obj.direction
            existing.last_alert_time = now
            existing.alert_count    += 1
        else:
            self._track_states[tid] = _TrackAlertState(
                track_id=tid,
                last_proximity=obj.proximity,
                last_direction=obj.direction,
                last_alert_time=now,
                first_confirmed_at=now,
                alert_count=1,
            )

    def _expire_stale_states(self, now: float) -> None:
        """
        Remove per-track state for tracks that have been absent longer
        than reappearance_window_seconds.

        We cannot know exactly when a track disappeared (that happens in
        Phase 10's state machine), so we use the time since last alert
        as a proxy. If a track has not been alerted in
        reappearance_window_seconds × 2, it is safe to expire.
        """
        expiry_threshold = self._reappearance_window_s * 2.0
        to_expire = [
            tid for tid, state in self._track_states.items()
            if (now - state.last_alert_time) > expiry_threshold
        ]
        for tid in to_expire:
            del self._track_states[tid]
            logger.debug("AlertManager: expired state for track %d.", tid)

    def _enqueue(self, text: str) -> None:
        """Enqueue text to the audio queue. Handles None queue gracefully."""
        if self._queue is None:
            logger.debug("AlertManager: no queue — would have spoken: %r", text)
            return
        try:
            self._queue.put(text)
        except Exception as e:
            logger.warning(
                "AlertManager: failed to enqueue %r: %s", text, e
            )

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all per-track alert state. Next update treated as first."""
        self._track_states.clear()
        logger.info("AlertManager: state reset.")

    def get_track_state(self, track_id: int) -> Optional[_TrackAlertState]:
        """Return a copy of the alert state for a specific track, or None."""
        return self._track_states.get(track_id)

    def set_queue(self, audio_queue) -> None:
        """Replace or set the audio queue at runtime."""
        self._queue = audio_queue

    @property
    def active_track_count(self) -> int:
        """Number of tracks currently holding alert state."""
        return len(self._track_states)

    @property
    def total_alerts(self) -> int:
        """Total number of alerts issued since creation or last reset."""
        return self._total_alerts

    @property
    def cooldown_seconds(self) -> float:
        """Configured cooldown interval (starting estimate)."""
        return self._cooldown_seconds

    def __repr__(self) -> str:
        return (
            f"AlertManager(cooldown={self._cooldown_seconds}s, "
            f"active_tracks={self.active_track_count}, "
            f"total_alerts={self._total_alerts})"
        )
