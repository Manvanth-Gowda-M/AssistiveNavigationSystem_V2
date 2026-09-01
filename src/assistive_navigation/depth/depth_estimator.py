"""
Depth Estimator Module
======================
Generates a relative depth map from a single camera frame.

Phase: 6 (not yet implemented)
Status: STUB — placeholder only

Model: Depth Anything V2 Small
License: Apache-2.0 (SMALL VARIANT ONLY — do not use Base/Large/Giant)
Source: https://github.com/DepthAnything/Depth-Anything-V2

IMPORTANT CONSTRAINTS:
- Output is RELATIVE depth, NOT metric distance.
- Do not report values as meters unless a proper calibration is performed.
- Reports proximity as: FAR / MEDIUM / CLOSE / VERY CLOSE
- Runs every N frames only (too slow for every frame on CPU).
  The last valid depth map is cached between updates.

CPU inference: ~400–1200ms per frame (varies with input size).
Strategy: run every 5 detection frames (configurable via depth_update_interval).
"""

# Implementation begins in Phase 6.
