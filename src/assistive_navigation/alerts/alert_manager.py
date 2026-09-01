"""
Alert Manager
=============
Decides WHEN to speak and WHAT to say. Does NOT do detection.

Phase: 12 (not yet implemented)
Status: STUB — placeholder only

Receives: confirmed, prioritized tracked objects
Outputs: alert text strings → audio queue

Handles:
  - Per-object cooldown (configurable, default 5 seconds)
  - Proximity escalation: MEDIUM→CLOSE→VERY CLOSE triggers new alert
  - Direction change: object moves CENTER→LEFT triggers new alert
  - New object: first confirmed detection always triggers alert
  - Duplicate suppression: same object/proximity/direction → suppressed
  - Priority ordering: highest priority alert spoken first

Does NOT:
  - Detect objects (that is the detector's job)
  - Make tracking decisions (that is the tracker's job)
  - Control TTS directly (that is the audio module's job)
"""

# Implementation begins in Phase 12.
