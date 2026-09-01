# Assistive Navigation System V2

A software-only, real-time assistive navigation prototype that uses a standard
webcam to detect nearby objects and speak short audio alerts.

---

## ⚠️ IMPORTANT SAFETY DISCLAIMER

**This is a research and educational prototype — NOT a certified assistive device.**

- This system **can and will** miss obstacles.
- This system **can and will** produce false detections (reporting objects that are not there).
- Object detection models are imperfect. Depth estimation on a standard camera is approximate.
- **Do not use this system as a substitute for a mobility aid, white cane, guide dog, or human assistance.**
- **Do not rely on this system in traffic, near stairs, on roads, or in any environment where a missed detection could cause injury.**
- This system is intended for **controlled, indoor testing only** during development.
- Always test in a **safe environment** with a sighted person present.

---

## What This System Does

1. Captures video from a standard laptop webcam — no additional hardware required.
2. Detects objects in the camera view using a local AI model (YOLO11n).
3. Classifies each detected object.
4. Determines the object's position: LEFT, CENTER (AHEAD), or RIGHT.
5. Estimates relative proximity: VERY CLOSE, CLOSE, MEDIUM, or FAR.
6. Tracks objects across frames so each obstacle has a stable identity.
7. Filters unstable/false detections using temporal confirmation.
8. Prioritizes the most navigation-relevant obstacles.
9. Speaks short, clear audio alerts — e.g., *"Person ahead, very close."*
10. Runs fully offline after initial model download — no cloud required.

---

## What This System Does NOT Do

- Does not use any custom hardware — only a normal laptop webcam and speaker.
- Does not connect to the internet during normal operation.
- Does not store video recordings or upload camera data anywhere.
- Does not provide exact metric distances (proximity is relative, not calibrated by default).
- Does not detect all possible obstacles (coverage is limited to COCO object classes + extensions).
- Is not suitable for outdoor navigation, road crossing, or stair navigation.

---

## Hardware Requirements (to run the software)

This is a **software-only project**. No hardware purchases required.

| Requirement | Minimum | Recommended |
|---|---|---|
| CPU | 4-core, 2 GHz | Intel i5/i7 8th gen+ |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB free | 10 GB free |
| Webcam | Any USB or integrated | 720p or better |
| Speaker | Any output | Headphones for clarity |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ | Windows 11 |
| Python | 3.9 – 3.11 | 3.10.x |
| GPU | Not required | NVIDIA GPU for faster depth |

---

## Quick Start (after setup)

```bash
# 1. Activate the virtual environment (Windows)
.venv\Scripts\activate

# 2. Run the system
python -m assistive_navigation

# 3. Exit
Press Ctrl+C
```

---

## Project Status

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for current phase and known issues.

---

## Development Phases

| Phase | Focus | Status |
|---|---|---|
| 0 | Discovery & Planning | ✅ Complete |
| 1 | Environment & Repository Setup | 🔄 In Progress |
| 2 | Camera Capture Module | ⏳ Pending |
| 3 | Object Detector Benchmark | ⏳ Pending |
| 4 | Detection Evaluation | ⏳ Pending |
| 5 | Object Tracking | ⏳ Pending |
| 6 | Depth Estimation | ⏳ Pending |
| 7 | Detection + Depth Fusion | ⏳ Pending |
| 8 | Direction Estimation | ⏳ Pending |
| 9 | Navigation Priority | ⏳ Pending |
| 10 | Temporal Filtering | ⏳ Pending |
| 11 | Audio / TTS | ⏳ Pending |
| 12 | Alert Manager | ⏳ Pending |
| 13 | Full Integration | ⏳ Pending |
| 14 | Performance Optimization | ⏳ Pending |
| 15 | False-Positive Reduction | ⏳ Pending |
| 16 | Evaluation | ⏳ Pending |
| 17 | Documentation | ⏳ Pending |
| 18 | Final Demonstration | ⏳ Pending |

---

## Technology Stack

| Component | Technology | License |
|---|---|---|
| Object detection | YOLO11n (Ultralytics) | AGPL-3.0 |
| Inference backend | ONNX Runtime | MIT |
| Object tracking | ByteTrack (via Ultralytics) | AGPL-3.0 |
| Depth estimation | Depth Anything V2 Small | Apache-2.0 |
| Camera capture | OpenCV | Apache-2.0 |
| Text-to-speech | pyttsx3 + Windows SAPI5 | MIT |
| Configuration | PyYAML | MIT |
| Language | Python 3.10 | PSF |

Full license details: [`docs/licenses.md`](docs/licenses.md)

---

## License

The source code in this repository is released under the **MIT License** — see [`LICENSE`](LICENSE).

Note: This project uses third-party libraries and models with their own licenses.
See [`docs/licenses.md`](docs/licenses.md) for the complete license inventory.

---

## Privacy

All camera processing happens **locally on your device**.
No video frames are uploaded to any server or cloud service.
No data is stored unless explicitly enabled in `config/config.yaml`.
