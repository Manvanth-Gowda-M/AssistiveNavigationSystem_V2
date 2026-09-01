"""
Temporal Confirmation Module
=============================
Prevents single-frame phantom detections from triggering audio alerts.

Phase: 10 (not yet implemented)
Status: STUB — placeholder only

An object must pass ALL of these checks before becoming an alert candidate:
  1. Detected for at least CONFIRMATION_FRAMES consecutive frames
  2. Class label consistent across frames (CLASS_STABILITY_THRESHOLD)
  3. Bounding box position reasonably stable (POSITION_STABILITY_THRESHOLD)
  4. Track age >= MIN_TRACK_AGE

State machine per track:
  NEW → CANDIDATE → CONFIRMED → ALERTING → (COOLDOWN) → ALERTING

Example: A single-frame "toothbrush" detection → state stays NEW → no alert.
A person detected for 5+ stable frames → CONFIRMED → eligible for alert.
"""

# Implementation begins in Phase 10.
