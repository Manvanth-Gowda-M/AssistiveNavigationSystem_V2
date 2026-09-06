# Installation — Clean Machine Setup

Covers README items 19–22: clean-machine setup, runtime requirements, model
acquisition, and configuration overview.

---

## 20. Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10/11 (primary, tested). Linux/macOS possible but the SAPI5 voice is Windows-only; other platforms fall back to their default pyttsx3 driver. |
| Python | 3.9 – 3.11 (tested on **3.10.11**). Not yet validated on 3.12+. |
| CPU | 4-core minimum; CPU-only (no GPU/CUDA required or used). |
| RAM | 8 GB minimum, 16 GB recommended. |
| Disk | ~5 GB free (dependencies + models). |
| Webcam | Any USB/integrated camera. |
| Speaker | Any audio output. |

Reference environment: Intel i5-12450H, 16 GB RAM, Windows 11, Python 3.10.11,
CPU-only.

---

## 19. Clean-machine setup (Windows PowerShell)

```powershell
# 1. Get the project
#    (clone your repository, or copy the project folder)
cd AssistiveNavigationSystem_V2

# 2. Create an isolated virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\activate

# 4. Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# 5. Install the pinned dependencies
pip install -r requirements.txt

# 6. Install this project into the venv (editable) so `python -m ...` works
pip install -e .
```

The pinned versions in [`requirements.txt`](../requirements.txt) match the
tested environment exactly. Installing `ultralytics` automatically pulls in
`torch` and `torchvision` (CPU builds).

Step 6 (`pip install -e .`) installs the package in editable mode. This is what
makes `python -m assistive_navigation` resolve the package. Without it, the
`-m` command fails with `No module named assistive_navigation`. (If you skip the
install, you can still run from the project root via
`$env:PYTHONPATH="src"; python -m assistive_navigation`.)

> **Do not upgrade/downgrade** the pinned versions casually — the detection,
> tracking, and depth behaviour (and the Phase 4/15/16 evaluation baselines)
> were validated against these exact versions.

### Linux / macOS notes
- Create the venv the same way; activate with `source .venv/bin/activate`.
- On Linux, `pyttsx3` uses `espeak`/`espeak-ng` (install via your package
  manager). On macOS it uses `nsss`. The `Zira`/`David` SAPI5 voices are
  Windows-only; set `audio.sapi5_voice` accordingly or expect a default voice.

---

## 21. Model acquisition

Two models are used:

1. **YOLO11n (detection)** — `models/yolo11n.pt` (~5.4 MB).
   Downloaded **automatically by Ultralytics** on the first run; no manual
   step required. License: AGPL-3.0 (inherits the Ultralytics framework).

2. **Depth Anything V2 Small (float32 ONNX)** — `models/depth_anything_v2_small.onnx`
   (~94.5 MB). Downloaded via the helper script:

```powershell
# Check what already exists:
python scripts/download_models.py --check

# Download the depth model:
python scripts/download_models.py
```

License: **Apache-2.0 (Small variant ONLY)**. See [`models.md`](models.md) for
sources, the INT8-rejection rationale, and license constraints.

After both models are present, the system runs **fully offline**.

---

## 22. Configuration overview

All tunable parameters live in [`config/config.yaml`](../config/config.yaml)
(the single source of truth). You normally do not need to edit it to run the
system. Key sections:

| Section | Purpose | Notable defaults |
|---|---|---|
| `camera` | Device id, resolution, FPS cap | `device_id: 0`, 640×480 |
| `detection` | Model, confidence, input size, threads | `confidence_threshold: 0.35`, `input_size: 640` |
| `object_classes` | Navigation-relevant classes + ignored list | — |
| `tracking` | ByteTrack parameters | `max_missed_frames: 30` |
| `depth` | Depth model, input size, update interval, threads | `input_size: 252`, `depth_update_interval: 5` |
| `proximity` | Relative depth → band thresholds | very_close 0.75 / close 0.55 / medium 0.35 |
| `spatial` | Direction zone boundaries | left 0.35 / right 0.65 |
| `temporal` | Confirmation gates | `confirmation_frames: 8`, `min_area_norm: 0.15` (Phase 15) |
| `priority` | nav_score weights | — |
| `alerts` | Cooldown, templates, queue size | `cooldown_seconds: 5.0` |
| `audio` | TTS engine, voice, rate, volume | `engine: sapi5`, `sapi5_voice: Zira` |
| `debug` / `pipeline` | Debug window, logging paths | `show_debug_window: true` |

> Proximity, cooldown, and confirmation thresholds are documented **starting
> estimates / relative assumptions**, not calibrated physical values. Changing
> them changes behaviour; re-run the tests and evaluation if you do.

Next: [`usage.md`](usage.md) for running, testing, and evaluation.
