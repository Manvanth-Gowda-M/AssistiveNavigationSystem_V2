"""
Assistive Navigation System V2

A software-only real-time assistive navigation prototype.
Uses a standard webcam to detect nearby objects and speak audio alerts.

SAFETY NOTICE: This is a research prototype. It must not be used as a
substitute for a mobility aid, white cane, guide dog, or human assistance.

Architecture:
  Camera → Detection → Filtering → Tracking → Depth → Spatial →
  Priority → Temporal Confirmation → Alert Manager → TTS → Audio Output
"""

__version__ = "0.2.0"
__status__ = "Development — Phase 1"
