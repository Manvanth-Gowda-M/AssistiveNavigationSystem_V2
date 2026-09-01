"""
Audio Queue Module
==================
Thread-safe priority queue for audio alerts.

Phase: 11 (not yet implemented)
Status: STUB — placeholder only

Prevents:
  - Overlapping speech (one alert at a time)
  - Alert spam ("person ahead" repeated every frame)
  - Lower-priority alerts blocking critical ones

Rules:
  - Critical alerts (VERY CLOSE, navigation-critical objects) have highest priority
  - Duplicate alerts within cooldown window are silently dropped
  - If queue is full (max_queue_size), lowest-priority pending alert is dropped
  - Critical alert can interrupt a currently-playing lower-priority alert
"""

# Implementation begins in Phase 11.
