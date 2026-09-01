"""
Object Tracker Module
=====================
Assigns stable track IDs to detected objects across video frames.

Phase: 5 (not yet implemented)
Status: STUB — placeholder only

Uses: ByteTrack (via Ultralytics) — CPU-compatible, no ReID model needed.

Each tracked object carries:
  track_id, class_id, class_name, confidence, bounding_box,
  center, age (frames seen), last_seen (frames ago), stability

Responsibilities:
- Associate detections to existing tracks using IoU + Kalman filter
- Handle missed detections (track survives up to MAX_MISSED_FRAMES)
- Assign new track IDs for new objects
- Remove stale tracks
- Allow switching to BoT-SORT if ByteTrack proves insufficient
"""

# Implementation begins in Phase 5.
