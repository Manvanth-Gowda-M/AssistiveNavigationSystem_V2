# Assistive Navigation System V2

A software-only, CPU-based, real-time **research prototype** that uses a
standard webcam to detect nearby objects and speak short navigation alerts
(e.g. *"person ahead, close"*). It runs fully offline after an initial
model download. No custom hardware, no cloud, no GPU required.

> **Status:** Phases 0–17 complete. Phase 18 (final demonstration & validation)
> complete for all headless items; live webcam/audio/performance items are
> **[NOT RUN]** pending physical execution (see
> [`demo/demo_script.md`](demo/demo_script.md)). See
> [`PROJECT_STATUS.md`](PROJECT_STATUS.md).
>
> **Final report:** [`docs/final_report.md`](docs/final_report.md) ·
> **Presentation:** [`docs/presentation.md`](docs/presentation.md) ·
> **Demo evidence:** [`demo/`](demo/)

---

## ⚠️ IMPORTANT SAFETY DISCLAIMER

**This is a research and educational prototype — NOT a certified assistive device.**

- This system **can and will** miss obstacles.
- This system **can and will** produce false detections (reporting objects that are not there).
- Object detection models are imperfect. Depth on a standard camera is **relative, not metric**.
- **Do not use this system as a substitute for a mobility aid, white cane, guide dog, or human assistance.**
- **Do not rely on this system in traffic, near stairs, on roads, or anywhere a missed detection could cause injury.** Stairs and doors are **not reliably detectable** (they are outside the model's object classes — see limitations).
- Intended for **controlled, indoor testing only**, always with a sighted person present.

---

## 1. Project overview

The system captures webcam frames and runs an 11-stage pipeline —
detection → filtering → tracking → depth → fusion → direction → priority →
temporal confirmation → alerting → speech — to announce the most
navigation-relevant nearby object as a short spoken phrase. Everything runs
locally on the CPU.

## 2. Problem statement

People with limited vision benefit from lightweight, low-cost awareness of
nearby obstacles. Commercial aids are expensive and often require special
hardware. This project explores **how far a purely software, CPU-only,
single-webcam prototype can go** using free/open-source components — and,
just as importantly, documents **where it falls short**.

## 3. System objectives

- Real-time-ish operation on a CPU-only laptop (no GPU/CUDA).
- Free/open-source components only.
- Detect common indoor navigation objects and speak concise alerts.
- Suppress obvious false positives without hiding genuine objects.
- Be **honest and auditable**: every threshold is documented, and every
  quantitative claim is labelled by evidence type (see below).

---

## What this system does

1. Captures video from a standard laptop webcam — no extra hardware.
2. Detects objects using a local model (YOLO11n via Ultralytics/PyTorch).
3. Filters detections by confidence + a navigation-relevant class list.
4. Tracks objects across frames (ByteTrack) for stable identities.
5. Estimates **relative** depth (Depth Anything V2 Small, ONNX Runtime).
6. Assigns direction: LEFT / CENTER (ahead) / RIGHT.
7. Classifies **relative** proximity: VERY CLOSE / CLOSE / MEDIUM / FAR.
8. Scores navigation priority and confirms objects temporally.
9. Speaks a short alert for the highest-priority confirmed object.
10. Runs fully offline after the initial model download.

## What this system does NOT do

- No custom hardware — a normal webcam and speaker only.
- No internet during normal operation; no video upload; no recording by default.
- No exact metric distances (proximity is **relative**, uncalibrated).
- No coverage beyond the model's object classes (COCO 80 + configured subset).
- **No reliable door or stair detection** (outside COCO classes).
- Not suitable for outdoor navigation, road crossing, or stairs.

---

## How results are labelled (read this before trusting any number)

Every quantitative statement in this project's docs is tagged with one of:

| Label | Meaning |
|---|---|
| **[MEASURED]** | From real captured data (Phase 4 physical evaluation; Phase 15 replay of that data). |
| **[SYNTHETIC-DETERMINISTIC]** | Phase 16 Layer 2 — correct by construction; validates pipeline **logic**, not real-world accuracy. |
| **[OPERATOR-LIVE / NOT RUN]** | Requires a live camera session; not executed unless an operator runs it. |
| **[DESIGN TARGET]** | An aspiration or starting estimate, not a validated result. |
| **[UNVALIDATED ASSUMPTION]** | A modelling assumption (e.g. relative-depth thresholds). |

This project makes **no** claims of precision, recall, false-negative rate,
mAP, accuracy, or generalization — no labelled dataset exists to support them.

---

## Hardware requirements (to run the software)

Software-only project. No hardware purchases required.

| Requirement | Minimum | Recommended |
|---|---|---|
| CPU | 4-core, 2 GHz | Intel i5/i7 8th gen+ |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB free | 10 GB free |
| Webcam | Any USB/integrated | 720p or better |
| Speaker | Any output | Headphones for clarity |
| OS | Windows 10/11 (primary); Linux/macOS possible | Windows 11 |
| Python | 3.9 – 3.11 | 3.10.x (tested: 3.10.11) |
| GPU | Not required (CPU-only) | Not used |

Reference machine used during development: Intel i5-12450H, 16 GB RAM,
Windows 11, Python 3.10.11, CPU-only (no CUDA).

---

## Quick start (after setup)

```bash
# 1. Create + activate the virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the depth model (YOLO auto-downloads on first run)
python scripts/download_models.py

# 4. Run the system (debug window)
python -m assistive_navigation
#    or headless / audio-only:
python -m assistive_navigation --headless

# 5. Exit: press Q in the debug window, or Ctrl+C
```

Full clean-machine steps: [`docs/installation.md`](docs/installation.md).
Running, testing, and evaluation: [`docs/usage.md`](docs/usage.md).
Reproducing results: [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Documentation map

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Architecture, data flow, per-stage explanations (items 4–18) |
| [`docs/installation.md`](docs/installation.md) | Clean-machine setup, requirements, model acquisition, configuration |
| [`docs/usage.md`](docs/usage.md) | Running the app, debug/headless, tests, evaluation, hardware eval |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Reproduce Phase 4/15/16 + all checklists |
| [`docs/models.md`](docs/models.md) | Model sources, sizes, licenses, INT8-rejection rationale |
| [`docs/testing.md`](docs/testing.md) | Test layout, markers, counts, how to run subsets |
| [`docs/evaluation.md`](docs/evaluation.md) | Phase 4 physical eval + Phase 16 three-layer methodology (authoritative) |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Troubleshooting table |
| [`docs/licenses.md`](docs/licenses.md) | Full license inventory |
| [`docs/final_report.md`](docs/final_report.md) | Phase 18 final report (all phases, labelled evidence) |
| [`docs/presentation.md`](docs/presentation.md) | ~14-slide presentation outline |
| [`demo/`](demo/) | Phase 18 evidence package + operator demo runbook |

---

## Development phases

| Phase | Focus | Status |
|---|---|---|
| 0 | Discovery & Planning | ✅ Complete |
| 1 | Environment & Repository Setup | ✅ Complete |
| 2 | Camera Capture Module | ✅ Complete |
| 3 | Object Detection Module | ✅ Complete |
| 4 | Detection Evaluation (physical) | ✅ Complete |
| 5 | Object Tracking (ByteTrack) | ✅ Complete |
| 6 | Depth Estimation (Depth Anything V2 Small) | ✅ Complete |
| 7 | Detection + Depth Fusion | ✅ Complete |
| 8 | Spatial Direction | ✅ Complete |
| 9 | Navigation Priority | ✅ Complete |
| 10 | Temporal Confirmation | ✅ Complete |
| 11 | Audio / TTS | ✅ Complete |
| 12 | Alert Manager | ✅ Complete |
| 13 | Full Pipeline Integration | ✅ Complete |
| 14 | Performance Optimization & Measurement | ✅ Complete |
| 15 | False-Positive Reduction | ✅ Complete |
| 16 | Final System Evaluation (3-layer) | ✅ Complete |
| 17 | Documentation & Reproducibility | ✅ Complete |
| 18 | Final Demonstration & Validation | ✅ Complete (headless); live items [NOT RUN] |

---

## Technology stack

| Component | Technology (as actually used) | License |
|---|---|---|
| Object detection | YOLO11n via **Ultralytics** on **PyTorch** (`.pt` weights, CPU) | AGPL-3.0 (Ultralytics) |
| Object tracking | ByteTrack (Ultralytics built-in) | AGPL-3.0 |
| Depth estimation | Depth Anything V2 **Small** (float32 ONNX) via **ONNX Runtime** | Apache-2.0 (model) / MIT (ORT) |
| PyTorch backend | torch, torchvision | BSD-3-Clause |
| Camera capture / imaging | OpenCV (`opencv-python`) | Apache-2.0 |
| Text-to-speech | pyttsx3 + Windows SAPI5 | MIT (pyttsx3) |
| Configuration | PyYAML | MIT |
| Performance monitoring | psutil | BSD-3-Clause |
| Language | Python 3.10 | PSF |

> Note: detection runs through the **Ultralytics/PyTorch** API (`.pt` weights),
> while the **depth** model runs through **ONNX Runtime**. Full inventory and
> exact versions: [`docs/licenses.md`](docs/licenses.md) and
> [`requirements.txt`](requirements.txt).

---

## Known limitations

- **Relative depth only** — proximity buckets are ordinal, not metric distances. [UNVALIDATED ASSUMPTION]
- **Class-confusion mislabels** occur (e.g. a table detected as a bottle). [MEASURED, Phase 4]
- **Doors and stairs are not COCO classes** — not reliably detected. Safety-critical gap. [MEASURED, Phase 4]
- **Remaining large-area false positives** overlap the genuine-object size range and are not fully suppressed. [MEASURED, Phase 15]
- **`couch` was NOT TESTED** (no real couch available during physical evaluation).
- **Single-environment evaluation** — one room, camera, and lighting; results do not generalize.
- Thresholds (proximity, cooldown, confirmation frames) are documented **starting estimates**. [DESIGN TARGET / UNVALIDATED ASSUMPTION]

## Future improvements

- Calibrated/metric depth (e.g. stereo or a depth sensor) for true distances.
- A door/stair detector to close the COCO capability gap.
- A labelled dataset to enable formal precision/recall/mAP evaluation.
- Multi-environment testing for generalization.
- Optional neural TTS (e.g. Piper) for clearer speech.

---

## Citation / attribution

If you reference this project, please cite it as a research prototype and
credit the upstream components:

```
Assistive Navigation System V2 (research prototype), 2026.
Uses: Ultralytics YOLO11 (AGPL-3.0); Depth Anything V2 Small (Apache-2.0);
ByteTrack (via Ultralytics); ONNX Runtime (MIT); OpenCV (Apache-2.0);
pyttsx3 (MIT). Pretrained detection weights trained on COCO.
```

Upstream references:
- Ultralytics YOLO — https://github.com/ultralytics/ultralytics (AGPL-3.0)
- Depth Anything V2 — https://github.com/DepthAnything/Depth-Anything-V2 (Small = Apache-2.0)
- ByteTrack — Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022
- COCO — Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014 (dataset CC BY 4.0)

See [`docs/licenses.md`](docs/licenses.md) for the complete inventory.

---

## License

The source code in this repository is released under the **MIT License** — see
[`LICENSE`](LICENSE). This project depends on third-party libraries and models
under their own licenses, including **AGPL-3.0** (Ultralytics/YOLO11) and
**Apache-2.0** (Depth Anything V2 Small). As an open-source research prototype,
the AGPL-3.0 terms are satisfied. Full details: [`docs/licenses.md`](docs/licenses.md).

---

## Privacy / local processing

All camera processing happens **locally on your device**. No video frames are
uploaded to any server or cloud service. Nothing is recorded or stored unless
you explicitly enable logging in [`config/config.yaml`](config/config.yaml)
(e.g. detection or performance CSV logs, which contain metrics — not video).
After the initial model download, the system runs fully offline.
