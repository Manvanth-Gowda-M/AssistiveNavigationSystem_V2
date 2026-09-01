"""
Object Detector Module
======================
Runs YOLO11n inference via ONNX Runtime to detect objects in a frame.

Phase: 3 (not yet implemented)
Status: STUB — placeholder only

Responsibilities:
- Load YOLO11n ONNX model
- Run inference on a BGR frame (from OpenCV)
- Return list of raw detections: class_id, class_name, confidence, bounding_box
- Measure and log inference latency per frame
- Handle missing model file gracefully
- Log ALL detections (including low-confidence) for analysis

IMPORTANT: High confidence does NOT mean correct detection.
Do not rely on confidence alone — temporal filtering handles false positives.
"""

# Implementation begins in Phase 3.
