"""
FastAPI Server for Assistive Navigation System V2 Testing Web Interface
========================================================================
"""

import asyncio
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from assistive_navigation.web.pipeline_runner import WebPipelineManager
from assistive_navigation.web.test_runner_api import (
    run_hardware_latency_benchmark,
    run_phase15_replay_summary,
    run_phase16_synthetic_eval,
    run_unit_tests,
)

logger = logging.getLogger(__name__)

# Paths
_WEB_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _WEB_DIR / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
(_STATIC_DIR / "css").mkdir(exist_ok=True)
(_STATIC_DIR / "js").mkdir(exist_ok=True)

app = FastAPI(
    title="Assistive Navigation System V2 — Testing Studio",
    description="Interactive Testing, Diagnostics, and Evaluation Web Interface",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline manager instance
pipeline_manager = WebPipelineManager()

# Active websocket clients
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Assistive Navigation Web Studio...")
    # Start streaming loop in background
    pipeline_manager.start_streaming()


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Stopping Assistive Navigation Web Studio...")
    pipeline_manager.stop_streaming()


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_system_status():
    """System health and component statuses."""
    det_loaded = bool(pipeline_manager._detector and pipeline_manager._detector.is_loaded)
    depth_loaded = bool(pipeline_manager._depth and pipeline_manager._depth.is_loaded)
    cam_open = bool(pipeline_manager._cam and pipeline_manager._cam.is_open)

    return {
        "status": "ready" if (det_loaded and depth_loaded) else "degraded",
        "detector_loaded": det_loaded,
        "detector_model": pipeline_manager._detector.model_name if det_loaded else None,
        "depth_loaded": depth_loaded,
        "camera_open": cam_open,
        "source_mode": pipeline_manager._source_mode,
        "streaming_active": pipeline_manager._is_streaming,
    }


@app.get("/api/config")
def get_config():
    """Get active configuration parameters."""
    return pipeline_manager.config


@app.post("/api/config")
def update_config(config_data: Dict[str, Any]):
    """Update active configuration parameters."""
    try:
        updated = pipeline_manager.update_config(config_data)
        return {"success": True, "config": updated}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/stream/start")
def start_stream():
    pipeline_manager.start_streaming()
    return {"success": True, "streaming": True}


@app.post("/api/stream/stop")
def stop_stream():
    pipeline_manager.stop_streaming()
    return {"success": True, "streaming": False}


@app.post("/api/stream/source")
def set_stream_source(data: Dict[str, str]):
    mode = data.get("mode", "camera")
    pipeline_manager.set_source_mode(mode)
    return {"success": True, "mode": mode}


@app.get("/api/telemetry")
def get_telemetry():
    return pipeline_manager.get_latest_telemetry()


@app.get("/api/stream/video")
async def video_stream_feed():
    """MJPEG live video stream generator."""
    async def frame_generator():
        while True:
            jpeg = pipeline_manager.get_latest_frame_jpeg()
            if jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(0.033)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/inspect")
async def inspect_uploaded_file(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns full 11-stage visual and data breakdown.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    # Resize to standard 640x480 for consistent pipeline evaluation
    img = cv2.resize(img, (640, 480))

    try:
        stages = pipeline_manager.inspect_full_stages(img)
        return {"success": True, "filename": file.filename, "stages": stages}
    except Exception as e:
        logger.error("Inspection error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inspection failed: {e}")


@app.post("/api/scenario/run")
def run_scenario(scenario_data: Dict[str, Any]):
    """
    Runs synthetic test scenario across N frames and returns frame telemetry progression.
    """
    scenario_id = scenario_data.get("scenario_id", "A")
    num_frames = int(scenario_data.get("num_frames", 15))

    # Build synthetic sequence according to scenario
    frames_telemetry = []

    for f_idx in range(num_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Background gradient
        for y in range(480):
            val = int(30 + 15 * (y / 480))
            frame[y, :] = (val, val + 5, val + 10)

        # Scenarios definitions:
        if scenario_id == "A":  # Approaching Person
            # Grows larger and stays centered
            scale = 0.5 + 0.5 * (f_idx / num_frames)
            w = int(120 * scale)
            h = int(240 * scale)
            x1, y1 = 320 - w // 2, 240 - h // 2
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (200, 200, 200), -1)
            cv2.putText(frame, "PERSON (APPROACHING)", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        elif scenario_id == "B":  # Stationary Chair Obstacle
            x1, y1, w, h = 270, 220, 140, 160
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (160, 120, 90), -1)
            cv2.putText(frame, "CHAIR (AHEAD)", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        elif scenario_id == "C":  # Fast Crossing Person (Left to Right)
            cx = int(80 + (500 * (f_idx / num_frames)))
            w, h = 100, 200
            x1, y1 = cx - w // 2, 240 - h // 2
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (200, 150, 150), -1)
            cv2.putText(frame, "CROSSING", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)

        elif scenario_id == "D":  # Small False-Positive Wall Artifact (Area < 0.15)
            x1, y1, w, h = 300, 200, 20, 30
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (100, 100, 100), -1)
            cv2.putText(frame, "BACKGROUND NOISE", (x1 - 20, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        else:
            # Default clutter
            cv2.rectangle(frame, (200, 180), (320, 340), (150, 150, 150), -1)

        telemetry, annotated = pipeline_manager.process_single_frame(frame)
        frames_telemetry.append({
            "frame_index": f_idx,
            "telemetry": telemetry,
        })

    return {
        "success": True,
        "scenario_id": scenario_id,
        "frames_processed": len(frames_telemetry),
        "results": frames_telemetry,
    }


# ---------------------------------------------------------------------------
# Automated Test Suite API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/tests/unit")
def execute_unit_tests(data: Optional[Dict[str, str]] = None):
    filter_pattern = data.get("filter") if data else None
    res = run_unit_tests(filter_pattern)
    return res


@app.post("/api/tests/phase16")
def execute_phase16_eval():
    res = run_phase16_synthetic_eval()
    return res


@app.post("/api/tests/phase15")
def execute_phase15_replay():
    res = run_phase15_replay_summary()
    return res


@app.post("/api/tests/benchmark")
def execute_benchmark():
    res = run_hardware_latency_benchmark()
    return res


# ---------------------------------------------------------------------------
# WebSocket Telemetry Stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected.")
    try:
        while True:
            telemetry = pipeline_manager.get_latest_telemetry()
            if telemetry:
                await websocket.send_json(telemetry)
            await asyncio.sleep(0.05)
    except (WebSocketDisconnect, Exception) as e:
        logger.debug("WebSocket client disconnected: %s", e)


# ---------------------------------------------------------------------------
# Static Files & Root
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Assistive Navigation Studio — Loading Static Assets...</h1>"
