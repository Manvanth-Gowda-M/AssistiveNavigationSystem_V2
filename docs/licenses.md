# License Inventory — Assistive Navigation System V2

All third-party software and models used in this project are listed here
with their licenses, sources, and implications.

Last updated: Phase 17 (verified against the installed `.venv` via `pip freeze`).

---

## Our Code

| Item | License | Notes |
|---|---|---|
| This project's source code | **MIT** | See `LICENSE` file |

---

## Core Libraries

| Library | Version | License | Source | Commercial Use | Notes |
|---|---|---|---|---|---|
| Python | 3.10.11 | PSF License | python.org | ✅ Yes | |
| OpenCV (opencv-python) | ≥4.8.0 | Apache-2.0 | github.com/opencv/opencv | ✅ Yes | |
| ONNX Runtime | ≥1.16.0 | MIT | github.com/microsoft/onnxruntime | ✅ Yes | |
| NumPy | ≥1.24.0 | BSD-3-Clause | numpy.org | ✅ Yes | |
| ~~SciPy~~ | — | BSD-3-Clause | scipy.org | ✅ Yes | **NOT installed / NOT used.** Considered earlier but never adopted; listed for transparency only. |
| PyYAML | ≥6.0 | MIT | pyyaml.org | ✅ Yes | |
| psutil | ≥5.9.0 | BSD-3-Clause | github.com/giampaolo/psutil | ✅ Yes | |
| pyttsx3 | ≥2.90 | MIT | github.com/nateshmbhat/pyttsx3 | ✅ Yes | |
| pytest | ≥7.0.0 | MIT | pytest.org | ✅ Yes | Dev only |

---

## AI Models and Frameworks

| Component | License | Source | Commercial Use | Critical Notes |
|---|---|---|---|---|
| **Ultralytics (YOLO11, ByteTrack, BoT-SORT)** | **AGPL-3.0** | github.com/ultralytics/ultralytics | ⚠️ **Conditional** | Free for open-source/research. If this project is ever distributed as a closed commercial product, an Ultralytics Enterprise license is required. Current use (open-source research prototype) is fully compliant. **Installed version: 8.4.137.** |
| **PyTorch (torch)** | **BSD-3-Clause** | github.com/pytorch/pytorch | ✅ Yes | Backend used by Ultralytics for YOLO11n inference (CPU). Pulled in automatically as an Ultralytics dependency. **Installed version: 2.13.0.** |
| **torchvision** | **BSD-3-Clause** | github.com/pytorch/vision | ✅ Yes | Vision ops used by Ultralytics. Pulled in automatically. **Installed version: 0.28.0.** |
| **Depth Anything V2 Small** | **Apache-2.0** | github.com/DepthAnything/Depth-Anything-V2 | ✅ Yes | **Small variant ONLY.** Base/Large/Giant use CC-BY-NC-4.0 (non-commercial). We use Small exclusively. |
| **Depth Anything V2 Base/Large/Giant** | **CC-BY-NC-4.0** | Same repo | ❌ Non-commercial only | **DO NOT USE.** NC restriction. |
| YOLO11n pretrained weights | AGPL-3.0 | Ultralytics HuggingFace | ⚠️ Same as Ultralytics | Weights inherit the framework license. |

---

## Audio

| Component | License | Notes |
|---|---|---|
| Windows SAPI5 TTS (David, Zira voices) | Microsoft Windows License | Built into Windows 11 — already installed, no separate download |
| pyttsx3 | MIT | Python wrapper for SAPI5 |
| ~~piper-tts-plus~~ (considered, NOT adopted) | MIT | Considered as a Phase 11 neural-TTS upgrade but **NOT installed / NOT used**. The system uses pyttsx3 + Windows SAPI5. |

---

## COCO Dataset (for pretrained weights)

The YOLO11n model was trained on COCO (Common Objects in Context).
- Dataset license: CC BY 4.0
- We do not distribute the dataset.
- We use pretrained weights only.
- Attribution: Lin, T.-Y., et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.

---

## Key Legal Notes

1. **AGPL-3.0 trigger:** If this project is ever packaged and distributed as a
   closed-source commercial application (not open-source), a paid Ultralytics
   Enterprise license is required. For a research prototype with open-source
   code, AGPL-3.0 is fully satisfied.

2. **Depth model size restriction:** ONLY Depth Anything V2 Small (Apache-2.0)
   is approved for use. Never substitute Base, Large, or Giant variants without
   re-evaluating the license change to CC-BY-NC-4.0.

3. **No cloud services:** All inference is local. No third-party API licenses
   apply to runtime operation.
