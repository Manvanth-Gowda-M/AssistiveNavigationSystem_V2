"""
Performance Monitor
===================
Measures and records FPS, CPU usage, RAM usage, and per-stage latency.

Phase: 1 (skeleton — full implementation in Phase 14)
Status: STUB — placeholder only

Metrics tracked:
  - FPS (frames per second — detection loop)
  - CPU usage percent (psutil)
  - RAM usage MB (psutil)
  - Per-stage latency (ms):
      frame_capture, detection, tracking, depth, fusion,
      spatial, priority, temporal, alert_decision, tts_queue
  - End-to-end latency (frame arrival → alert queued)

Results logged to: logs/performance.csv
"""

# Implementation begins in Phase 14.
# Skeleton only — imported by other modules for type hints.
