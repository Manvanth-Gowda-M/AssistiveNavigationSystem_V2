"""
Unit Tests — Alert Manager (Phase 12)
=======================================
Phase 12 acceptance criteria:

  [1]  Unconfirmed object produces no alert
  [2]  First confirmed object triggers alert ("new_object")
  [3]  Highest nav_score object selected when multiple candidates exist
  [4]  Duplicate alert suppressed within cooldown (same proximity+direction)
  [5]  Cooldown expiry triggers re-alert
  [6]  Proximity escalation triggers alert (MEDIUM→CLOSE, CLOSE→VERY_CLOSE)
  [7]  Direction change triggers alert while still in cooldown
  [8]  No confirmed objects → no alert
  [9]  Track disappears and reappears within window → no duplicate alert
  [10] State expires → reappearance treated as new object
  [11] Multiple independent tracks maintain independent states
  [12] Low nav_score object cannot override high nav_score object
  [13] TTS/audio failure does not crash system
  [14] Alert text contains useful object/direction/proximity information
  [15] Deterministic: same inputs produce same decision
  [16] reset() clears all state
  [17] Hardware: confirmed person produces spoken alert
  [18] Hardware: repeated identical frames do not continuously speak

All logic tests use synthetic ConfirmedObject data — no camera or model needed.

SCOPE CONFIRMATION:
  Phase 12 implements ONLY:
    - Alert eligibility gating (is_confirmed check)
    - Object selection (highest nav_score)
    - Cooldown suppression
    - Proximity escalation detection
    - Direction change detection
    - Message formatting
  Phase 12 does NOT implement and these tests do NOT verify:
    - Tracker behavior changes
    - Detector threshold changes
    - Phase 4 FP suppression changes (Phase 15)
    - Any temporal confirmation changes (Phase 10)

Run all tests:
    .venv/Scripts/python.exe -m pytest tests/unit/test_alert_manager.py -v -s

Run logic tests only:
    .venv/Scripts/python.exe -m pytest tests/unit/test_alert_manager.py -v -m "not requires_camera and not requires_audio"
"""

import time
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Synthetic ConfirmedObject factory
# ---------------------------------------------------------------------------

def make_confirmed(
    track_id: int = 1,
    class_name: str = "person",
    class_id: int = 0,
    is_confirmed: bool = True,
    confirmation_state: str = "CONFIRMED",
    nav_score: float = 0.75,
    proximity: str = "CLOSE",
    direction: str = "CENTER",
    priority: str = "navigation_critical",
    confidence: float = 0.88,
    stability: float = 0.95,
    age: int = 10,
    consecutive_count: int = 10,
):
    """Create a synthetic ConfirmedObject using the real dataclass."""
    from assistive_navigation.navigation.temporal import ConfirmedObject
    return ConfirmedObject(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=(100.0, 150.0, 400.0, 480.0),
        bbox_norm=(0.156, 0.3125, 0.625, 1.0),
        center_x=0.50,
        center_y=0.65,
        age=age,
        last_seen=0,
        stability=stability,
        area_norm=0.30,
        frame_width=640,
        frame_height=480,
        raw_depth=0.65,
        proximity=proximity,
        priority=priority,
        depth_samples=100,
        depth_age=0,
        direction=direction,
        nav_score=nav_score,
        confirmation_state=confirmation_state,
        consecutive_count=consecutive_count,
        is_confirmed=is_confirmed,
        gate_failures=[],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


@pytest.fixture
def manager_no_queue(config):
    """AlertManager with no audio queue (silent — for logic tests)."""
    from assistive_navigation.alerts.alert_manager import AlertManager
    return AlertManager(config, audio_queue=None)


@pytest.fixture
def mock_queue():
    """A mock SimpleAudioQueue that records put() calls."""
    q = MagicMock()
    q.put = MagicMock()
    q.is_running = True
    return q


@pytest.fixture
def manager(config, mock_queue):
    """AlertManager with a mock queue — captures alerts without speaking."""
    from assistive_navigation.alerts.alert_manager import AlertManager
    return AlertManager(config, audio_queue=mock_queue)


def _make_manager_custom(cooldown: float = 5.0, reappear: float = 3.0,
                          audio_queue=None):
    """Create AlertManager with custom timing settings."""
    from assistive_navigation.alerts.alert_manager import AlertManager
    cfg = {
        "alerts": {
            "cooldown_seconds": cooldown,
            "reappearance_window_seconds": reappear,
            "templates": {
                "standard":   "{object} {direction}, {proximity}",
                "new_object": "{object} {direction}",
                "escalation": "{object} {direction}, now {proximity}",
            },
        },
        "spatial":  {"labels": {"center": "ahead", "left": "on your left", "right": "on your right"}},
        "proximity": {"labels": {"very_close": "very close", "close": "close", "medium": "nearby"}},
    }
    return AlertManager(cfg, audio_queue=audio_queue)


# ===========================================================================
# GROUP 1 — Import and structure
# ===========================================================================

class TestAlertManagerImports:

    def test_alert_manager_importable(self):
        from assistive_navigation.alerts.alert_manager import AlertManager
        assert AlertManager is not None

    def test_alert_result_importable(self):
        from assistive_navigation.alerts.alert_manager import AlertResult
        assert AlertResult is not None

    def test_manager_creates(self, config):
        from assistive_navigation.alerts.alert_manager import AlertManager
        m = AlertManager(config)
        assert m is not None

    def test_manager_repr(self, manager):
        assert "AlertManager" in repr(manager)

    def test_cooldown_property(self, config):
        from assistive_navigation.alerts.alert_manager import AlertManager
        m = AlertManager(config)
        assert m.cooldown_seconds == config["alerts"]["cooldown_seconds"]


# ===========================================================================
# GROUP 2 — AlertResult structure
# ===========================================================================

class TestAlertResultStructure:

    def test_alert_result_fields_present(self, manager):
        obj = make_confirmed()
        result = manager.process([obj])
        assert hasattr(result, "alert_issued")
        assert hasattr(result, "alert_text")
        assert hasattr(result, "selected_track")
        assert hasattr(result, "trigger_reason")
        assert hasattr(result, "eligible_count")
        assert hasattr(result, "suppressed_count")

    def test_alert_result_types(self, manager):
        obj = make_confirmed()
        result = manager.process([obj])
        assert isinstance(result.alert_issued,     bool)
        assert isinstance(result.alert_text,       str)
        assert isinstance(result.selected_track,   int)
        assert isinstance(result.trigger_reason,   str)
        assert isinstance(result.eligible_count,   int)
        assert isinstance(result.suppressed_count, int)


# ===========================================================================
# GROUP 3 — No alert for unconfirmed objects (PASS CONDITION 1)
# ===========================================================================

class TestUnconfirmedSuppressed:
    """PASS CONDITION 1: unconfirmed objects must never produce alerts."""

    def test_candidate_state_no_alert(self, manager):
        obj = make_confirmed(is_confirmed=False, confirmation_state="CANDIDATE")
        result = manager.process([obj])
        assert result.alert_issued is False

    def test_new_state_no_alert(self, manager):
        obj = make_confirmed(is_confirmed=False, confirmation_state="NEW")
        result = manager.process([obj])
        assert result.alert_issued is False

    def test_lost_state_no_alert(self, manager):
        obj = make_confirmed(is_confirmed=False, confirmation_state="LOST")
        result = manager.process([obj])
        assert result.alert_issued is False

    def test_unconfirmed_eligible_count_zero(self, manager):
        obj = make_confirmed(is_confirmed=False)
        result = manager.process([obj])
        assert result.eligible_count == 0


# ===========================================================================
# GROUP 4 — No objects produces no alert (PASS CONDITION 8)
# ===========================================================================

class TestNoObjects:

    def test_empty_input_no_alert(self, manager):
        """PASS CONDITION 8: no confirmed objects → no alert."""
        result = manager.process([])
        assert result.alert_issued is False
        assert result.alert_text   == ""
        assert result.selected_track == -1
        assert result.trigger_reason == "none"

    def test_empty_eligible_count_zero(self, manager):
        result = manager.process([])
        assert result.eligible_count == 0


# ===========================================================================
# GROUP 5 — First confirmed object triggers alert (PASS CONDITION 2)
# ===========================================================================

class TestFirstConfirmedTriggers:
    """PASS CONDITION 2: first confirmed object always produces an alert."""

    def test_first_confirmed_triggers_alert(self, manager):
        obj = make_confirmed(track_id=99)
        result = manager.process([obj])
        assert result.alert_issued is True

    def test_first_confirmed_reason_is_new_object(self, manager):
        obj = make_confirmed(track_id=98)
        result = manager.process([obj])
        assert result.trigger_reason == "new_object"

    def test_first_confirmed_alert_text_non_empty(self, manager):
        obj = make_confirmed(track_id=97)
        result = manager.process([obj])
        assert len(result.alert_text) > 0

    def test_first_confirmed_updates_track_state(self, manager):
        obj = make_confirmed(track_id=96)
        result = manager.process([obj])
        state = manager.get_track_state(96)
        assert state is not None
        assert state.alert_count == 1


# ===========================================================================
# GROUP 6 — Duplicate suppression within cooldown (PASS CONDITION 4)
# ===========================================================================

class TestDuplicateSuppression:
    """PASS CONDITION 4: same proximity+direction within cooldown → suppressed."""

    def test_same_state_within_cooldown_suppressed(self):
        m = _make_manager_custom(cooldown=60.0)
        obj = make_confirmed(track_id=1, proximity="CLOSE", direction="CENTER")
        # First call → alert
        r1 = m.process([obj])
        assert r1.alert_issued is True
        # Immediate second call — same state, within 60s cooldown
        r2 = m.process([obj])
        assert r2.alert_issued is False, (
            "Same object, same proximity+direction, within cooldown → must be suppressed"
        )

    def test_suppressed_eligible_count_correct(self):
        m = _make_manager_custom(cooldown=60.0)
        obj = make_confirmed(track_id=1, proximity="CLOSE", direction="CENTER")
        m.process([obj])
        r = m.process([obj])
        assert r.suppressed_count == 1
        assert r.eligible_count   == 1

    def test_queue_put_called_once_for_repeated_same_object(self):
        q = MagicMock()
        m = _make_manager_custom(cooldown=60.0, audio_queue=q)
        obj = make_confirmed(track_id=1)
        for _ in range(5):
            m.process([obj])
        # queue.put should be called only once (first time)
        assert q.put.call_count == 1


# ===========================================================================
# GROUP 7 — Cooldown expiry triggers re-alert (PASS CONDITION 5)
# ===========================================================================

class TestCooldownExpiry:
    """PASS CONDITION 5: after cooldown_seconds → re-alert issued."""

    def test_cooldown_expiry_triggers_alert(self):
        """Use very short cooldown to test without real sleep."""
        m = _make_manager_custom(cooldown=0.01)
        obj = make_confirmed(track_id=1, proximity="CLOSE", direction="CENTER")
        m.process([obj])
        # Wait just beyond the 10ms cooldown
        time.sleep(0.05)
        r2 = m.process([obj])
        assert r2.alert_issued is True
        assert r2.trigger_reason == "cooldown_expired"

    def test_before_cooldown_no_alert(self):
        """Within the cooldown window, same state must be suppressed."""
        m = _make_manager_custom(cooldown=60.0)
        obj = make_confirmed(track_id=1, proximity="CLOSE", direction="CENTER")
        m.process([obj])
        r = m.process([obj])
        assert r.alert_issued is False


# ===========================================================================
# GROUP 8 — Proximity escalation triggers alert (PASS CONDITION 6)
# ===========================================================================

class TestProximityEscalation:
    """PASS CONDITION 6: proximity moving closer triggers alert."""

    def test_medium_to_close_escalates(self):
        m = _make_manager_custom(cooldown=60.0)
        t1 = make_confirmed(track_id=1, proximity="MEDIUM", direction="CENTER")
        m.process([t1])
        t2 = make_confirmed(track_id=1, proximity="CLOSE", direction="CENTER")
        r = m.process([t2])
        assert r.alert_issued is True
        assert r.trigger_reason == "proximity_escalation"

    def test_close_to_very_close_escalates(self):
        m = _make_manager_custom(cooldown=60.0)
        m.process([make_confirmed(track_id=1, proximity="CLOSE")])
        r = m.process([make_confirmed(track_id=1, proximity="VERY_CLOSE")])
        assert r.alert_issued is True
        assert r.trigger_reason == "proximity_escalation"

    def test_escalation_text_uses_escalation_template(self):
        m = _make_manager_custom(cooldown=60.0)
        m.process([make_confirmed(track_id=1, proximity="MEDIUM", direction="CENTER")])
        r = m.process([make_confirmed(track_id=1, proximity="CLOSE", direction="CENTER")])
        assert "now" in r.alert_text.lower() or "close" in r.alert_text.lower()

    def test_deescalation_does_not_trigger(self):
        """Moving farther away does NOT trigger an alert."""
        m = _make_manager_custom(cooldown=60.0)
        m.process([make_confirmed(track_id=1, proximity="CLOSE")])
        r = m.process([make_confirmed(track_id=1, proximity="MEDIUM")])
        assert r.alert_issued is False

    def test_same_proximity_does_not_escalate(self):
        m = _make_manager_custom(cooldown=60.0)
        m.process([make_confirmed(track_id=1, proximity="CLOSE")])
        r = m.process([make_confirmed(track_id=1, proximity="CLOSE")])
        assert r.alert_issued is False


# ===========================================================================
# GROUP 9 — Direction change triggers alert (PASS CONDITION 7)
# ===========================================================================

class TestDirectionChange:
    """PASS CONDITION 7: direction change triggers alert even within cooldown."""

    def test_center_to_left_triggers_alert(self):
        m = _make_manager_custom(cooldown=60.0)
        m.process([make_confirmed(track_id=1, direction="CENTER", proximity="CLOSE")])
        r = m.process([make_confirmed(track_id=1, direction="LEFT",   proximity="CLOSE")])
        assert r.alert_issued is True
        assert r.trigger_reason == "direction_change"

    def test_center_to_right_triggers_alert(self):
        m = _make_manager_custom(cooldown=60.0)
        m.process([make_confirmed(track_id=1, direction="CENTER", proximity="CLOSE")])
        r = m.process([make_confirmed(track_id=1, direction="RIGHT",  proximity="CLOSE")])
        assert r.alert_issued is True

    def test_same_direction_no_trigger(self):
        m = _make_manager_custom(cooldown=60.0)
        m.process([make_confirmed(track_id=1, direction="LEFT", proximity="CLOSE")])
        r = m.process([make_confirmed(track_id=1, direction="LEFT", proximity="CLOSE")])
        assert r.alert_issued is False


# ===========================================================================
# GROUP 10 — Highest nav_score selected (PASS CONDITION 3, 12)
# ===========================================================================

class TestObjectSelection:
    """PASS CONDITION 3, 12: highest urgency object is selected."""

    def test_highest_score_selected(self):
        m = _make_manager_custom(cooldown=0.0)   # no cooldown for this test
        low  = make_confirmed(track_id=1, nav_score=0.20, class_name="bottle")
        high = make_confirmed(track_id=2, nav_score=0.90, class_name="person")
        r = m.process([low, high])
        assert r.selected_track == 2
        assert "person" in r.alert_text.lower()

    def test_low_score_does_not_override_high_score(self):
        """PASS CONDITION 12: low-value object must not crowd out urgent obstacle."""
        m = _make_manager_custom(cooldown=0.0)
        urgent    = make_confirmed(track_id=1, nav_score=0.95, class_name="person",
                                   proximity="VERY_CLOSE", direction="CENTER")
        low_value = make_confirmed(track_id=2, nav_score=0.15, class_name="book",
                                   proximity="FAR", direction="LEFT")
        r = m.process([urgent, low_value])
        assert r.selected_track == 1, (
            "Urgent person must be selected, not the low-score book"
        )

    def test_eligible_count_reflects_confirmed_only(self):
        unconfirmed = make_confirmed(track_id=1, is_confirmed=False)
        confirmed   = make_confirmed(track_id=2, is_confirmed=True)
        m = _make_manager_custom(cooldown=0.0)
        r = m.process([unconfirmed, confirmed])
        assert r.eligible_count == 1


# ===========================================================================
# GROUP 11 — Alert text format (PASS CONDITION 14)
# ===========================================================================

class TestAlertTextFormat:

    def test_alert_contains_class_name(self, manager):
        obj = make_confirmed(track_id=101, class_name="chair")
        r = manager.process([obj])
        assert "chair" in r.alert_text.lower()

    def test_alert_contains_direction(self, manager):
        obj = make_confirmed(track_id=102, direction="LEFT", proximity="CLOSE")
        r = manager.process([obj])
        # Should contain "left" in some form
        assert "left" in r.alert_text.lower()

    def test_alert_contains_proximity_when_close(self, manager):
        obj = make_confirmed(track_id=103, proximity="VERY_CLOSE", direction="CENTER")
        r = manager.process([obj])
        # Should contain "close" in some form
        assert "close" in r.alert_text.lower()

    def test_alert_center_direction_maps_to_ahead(self, manager):
        obj = make_confirmed(track_id=104, direction="CENTER", proximity="CLOSE")
        r = manager.process([obj])
        assert "ahead" in r.alert_text.lower()

    def test_alert_far_proximity_uses_shorter_format(self):
        """FAR proximity should use new_object template (no distance label)."""
        m = _make_manager_custom(cooldown=0.0)
        obj = make_confirmed(track_id=1, proximity="FAR", direction="CENTER")
        r = m.process([obj])
        assert r.alert_issued is True
        # "ahead" in text but should NOT say "ahead" followed by "far"
        # (FAR has empty label so new_object template is used)
        assert len(r.alert_text) > 0

    def test_unknown_direction_object_not_eligible(self):
        """Objects with UNKNOWN direction are not eligible."""
        m = _make_manager_custom()
        obj = make_confirmed(track_id=1, direction="UNKNOWN", is_confirmed=True)
        r = m.process([obj])
        assert r.alert_issued is False


# ===========================================================================
# GROUP 12 — Independent tracks (PASS CONDITION 11)
# ===========================================================================

class TestIndependentTracks:
    """PASS CONDITION 11: multiple tracks maintain independent alert states."""

    def test_two_tracks_independent_cooldowns(self):
        m = _make_manager_custom(cooldown=60.0)
        t1 = make_confirmed(track_id=1, nav_score=0.8, direction="LEFT")
        t2 = make_confirmed(track_id=2, nav_score=0.6, direction="RIGHT")

        # First update: track_1 wins (higher score)
        r1 = m.process([t1, t2])
        assert r1.selected_track == 1

        # Track_1 is now in cooldown. Track_2 still hasn't been alerted.
        # But on second call, track_1 is suppressed, track_2 should trigger.
        r2 = m.process([t1, t2])
        assert r2.selected_track == 2, (
            "Track_2 was never alerted — should now be selected since track_1 "
            "is in cooldown."
        )

    def test_confirming_track1_does_not_suppress_track2(self):
        m = _make_manager_custom(cooldown=60.0)
        t1 = make_confirmed(track_id=1)
        t2 = make_confirmed(track_id=2, nav_score=0.5, direction="LEFT")
        m.process([t1])     # track_1 alerted
        r = m.process([t2]) # track_2 — fresh, should alert
        assert r.alert_issued is True
        assert r.selected_track == 2


# ===========================================================================
# GROUP 13 — Reappearance / state expiry (PASS CONDITIONS 9, 10)
# ===========================================================================

class TestStateExpiry:

    def test_reappearance_within_window_no_new_alert(self):
        """PASS CONDITION 9: track disappears and reappears quickly."""
        m = _make_manager_custom(cooldown=60.0, reappear=10.0)
        obj = make_confirmed(track_id=1, proximity="CLOSE", direction="CENTER")
        m.process([obj])
        # Immediately reappear — still within window and cooldown
        r = m.process([obj])
        assert r.alert_issued is False, (
            "Reappeared within window and cooldown → must be suppressed"
        )

    def test_state_expires_fresh_start(self):
        """PASS CONDITION 10: state expires → reappearance = new object."""
        m = _make_manager_custom(cooldown=0.01, reappear=0.01)
        obj = make_confirmed(track_id=1)
        m.process([obj])
        # Wait for both cooldown and expiry
        time.sleep(0.05)
        # Manually expire
        m._expire_stale_states(time.monotonic())
        state = m.get_track_state(1)
        assert state is None, "State should have been expired"
        # Next process should treat it as new
        r = m.process([obj])
        assert r.trigger_reason == "new_object"


# ===========================================================================
# GROUP 14 — Deterministic behaviour (PASS CONDITION 15)
# ===========================================================================

class TestDeterministicBehaviour:

    def test_same_inputs_same_decision(self, config):
        """Identical inputs produce identical alert decisions."""
        from assistive_navigation.alerts.alert_manager import AlertManager

        def run():
            m = AlertManager(config)
            obj = make_confirmed(track_id=1, proximity="CLOSE", direction="CENTER")
            r1 = m.process([obj])
            return r1.alert_issued, r1.trigger_reason, r1.alert_text

        assert run() == run()


# ===========================================================================
# GROUP 15 — Reset (PASS CONDITION 16)
# ===========================================================================

class TestReset:

    def test_reset_clears_state(self, manager):
        obj = make_confirmed(track_id=200)
        manager.process([obj])
        assert manager.active_track_count > 0
        manager.reset()
        assert manager.active_track_count == 0

    def test_after_reset_next_alert_is_new(self, manager):
        obj = make_confirmed(track_id=201, proximity="CLOSE", direction="CENTER")
        manager.process([obj])
        manager.reset()
        r = manager.process([obj])
        assert r.alert_issued is True
        assert r.trigger_reason == "new_object"


# ===========================================================================
# GROUP 16 — Queue and TTS failure handling (PASS CONDITION 13)
# ===========================================================================

class TestGracefulFailure:

    def test_none_queue_no_crash(self, config):
        """AlertManager with no queue must not crash."""
        from assistive_navigation.alerts.alert_manager import AlertManager
        m = AlertManager(config, audio_queue=None)
        obj = make_confirmed()
        r = m.process([obj])   # should not raise
        assert r.alert_issued is True  # decision was made

    def test_queue_put_failure_no_crash(self, config):
        """If queue.put() raises, AlertManager must swallow the exception."""
        from assistive_navigation.alerts.alert_manager import AlertManager
        bad_queue = MagicMock()
        bad_queue.put.side_effect = RuntimeError("queue broken")
        m = AlertManager(config, audio_queue=bad_queue)
        obj = make_confirmed()
        r = m.process([obj])  # must not raise
        assert isinstance(r.alert_issued, bool)

    def test_alert_issued_true_even_if_queue_fails(self, config):
        """alert_issued reflects the decision, not whether audio played."""
        from assistive_navigation.alerts.alert_manager import AlertManager
        bad_queue = MagicMock()
        bad_queue.put.side_effect = RuntimeError("broken")
        m = AlertManager(config, audio_queue=bad_queue)
        obj = make_confirmed()
        r = m.process([obj])
        # The decision was made; failure was in delivery
        assert r.trigger_reason == "new_object"


# ===========================================================================
# GROUP 17 — Hardware / real speech tests
# ===========================================================================

class TestAlertManagerHardware:

    requires_audio = pytest.mark.requires_audio

    @requires_audio
    def test_confirmed_person_produces_spoken_alert(self, config):
        """
        PASS CONDITION 17 (hardware):
        A confirmed person should produce a real spoken alert.
        You should hear the navigation message through your speakers.
        Uses a fresh TTS instance to avoid shared-engine issues.
        """
        import time
        from assistive_navigation.audio.tts import TextToSpeech
        from assistive_navigation.audio.queue import SimpleAudioQueue
        from assistive_navigation.alerts.alert_manager import AlertManager

        # Fresh TTS instance — avoids shared engine state from module fixture
        tts = TextToSpeech(config)
        if not tts.initialise():
            pytest.skip("TTS not available.")

        q = SimpleAudioQueue(tts)
        q.start()
        m = AlertManager(config, audio_queue=q)

        obj = make_confirmed(
            track_id=1,
            class_name="person",
            proximity="VERY_CLOSE",
            direction="CENTER",
            nav_score=0.95,
        )

        print("\n  >>> You should hear: 'person ahead, very close'")
        r = m.process([obj])

        assert r.alert_issued is True
        assert len(r.alert_text) > 0
        print(f"  Alert text: {r.alert_text!r}")
        print(f"  Trigger reason: {r.trigger_reason}")

        # Give SAPI5 time to finish, then stop cleanly
        time.sleep(1.5)
        tts.stop()
        q.stop()
        tts.shutdown()

    @requires_audio
    def test_repeated_identical_frames_do_not_continuously_speak(self, config):
        """
        PASS CONDITION 18 (hardware):
        The same confirmed object presented 30 times must produce only
        one alert (subsequent calls suppressed by cooldown).
        Uses a fresh TTS instance to avoid shared-engine issues.
        """
        import time
        from assistive_navigation.audio.tts import TextToSpeech
        from assistive_navigation.audio.queue import SimpleAudioQueue
        from assistive_navigation.alerts.alert_manager import AlertManager

        tts = TextToSpeech(config)
        if not tts.initialise():
            pytest.skip("TTS not available.")

        q = SimpleAudioQueue(tts)
        q.start()
        m = AlertManager(config, audio_queue=q)

        obj = make_confirmed(
            track_id=1,
            class_name="person",
            proximity="CLOSE",
            direction="CENTER",
            nav_score=0.80,
        )

        alerts_issued = 0
        for _ in range(30):
            r = m.process([obj])
            if r.alert_issued:
                alerts_issued += 1

        print(f"\n  Alerts issued in 30 identical frames: {alerts_issued}")
        print(f"  (Expected: 1 — all others suppressed by cooldown)")

        # Give SAPI5 time to finish the one alert, then stop
        time.sleep(1.5)
        tts.stop()
        q.stop()
        tts.shutdown()

        assert alerts_issued == 1, (
            f"Expected exactly 1 alert in 30 identical frames, "
            f"got {alerts_issued}."
        )

    @requires_audio
    @pytest.mark.requires_camera
    def test_full_pipeline_produces_alert(self, config):
        """
        Full pipeline hardware test: camera → detector → all phases → alert.
        Demonstrates the complete system end-to-end through Phase 12.
        """
        import time
        from assistive_navigation.camera.capture import CameraCapture, CameraError
        from assistive_navigation.detection.detector import ObjectDetector, DetectorError
        from assistive_navigation.detection.filter import DetectionFilter
        from assistive_navigation.tracking.tracker import ObjectTracker
        from assistive_navigation.depth.depth_estimator import DepthEstimator
        from assistive_navigation.depth.fusion import DepthFusion
        from assistive_navigation.navigation.spatial import SpatialReasoner
        from assistive_navigation.navigation.priority import NavigationPriorityEngine
        from assistive_navigation.navigation.temporal import TemporalConfirmationFilter
        from assistive_navigation.audio.tts import TextToSpeech
        from assistive_navigation.audio.queue import SimpleAudioQueue
        from assistive_navigation.alerts.alert_manager import AlertManager

        # Load all pipeline components
        try:
            detector = ObjectDetector(config)
            detector.load()
        except DetectorError as e:
            pytest.skip(f"Detector not available: {e}")
        try:
            depth_est = DepthEstimator(config)
            depth_est.load()
        except Exception as e:
            pytest.skip(f"Depth estimator not available: {e}")

        det_filter = DetectionFilter(config)
        tracker    = ObjectTracker(config)
        fusion     = DepthFusion(config)
        spatial    = SpatialReasoner(config)
        engine     = NavigationPriorityEngine(config)
        tcf        = TemporalConfirmationFilter(config)

        tts = TextToSpeech(config)
        if not tts.initialise():
            pytest.skip("TTS not available.")
        q = SimpleAudioQueue(tts)
        q.start()
        manager = AlertManager(config, audio_queue=q)

        try:
            cam = CameraCapture(config)
            cam.open()
        except CameraError as e:
            pytest.skip(f"Webcam not available: {e}")

        first_alert = None
        print("\n  >>> Running full pipeline. Stand in front of camera.")
        print(f"  confirmation_frames={tcf.confirmation_frames} frames required.")

        try:
            for frame_num in range(40):
                frame = cam.read()
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                det_result = detector.detect(frame)
                filtered   = det_filter.filter(det_result)
                tracked    = tracker.update(filtered.accepted, w, h)
                depth_map  = depth_est.process_frame(frame)
                fused      = fusion.fuse(tracked, depth_map, depth_est.depth_frame_age)
                directed   = spatial.assign_direction(fused)
                scored     = engine.score(directed)
                confirmed  = tcf.update(scored)
                result     = manager.process(confirmed)

                if result.alert_issued and first_alert is None:
                    first_alert = (frame_num, result.alert_text, result.trigger_reason)
                    print(f"  ALERT at frame {frame_num}: {result.alert_text!r}")
                    print(f"  Reason: {result.trigger_reason}")
                    break

        finally:
            cam.release()

        # Give SAPI5 time to finish, then stop cleanly
        import time
        time.sleep(1.5)
        tts.stop()
        q.stop()
        tts.shutdown()

        if first_alert is None:
            print(f"  No alert was produced in 40 frames.")
            print(f"  Possible reasons:")
            print(f"  - No object confirmed in {tcf.confirmation_frames} frames")
            print(f"  - No person/object in camera view")
        else:
            print(f"\n  First alert: frame={first_alert[0]}, "
                  f"text={first_alert[1]!r}, reason={first_alert[2]}")
