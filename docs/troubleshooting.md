# Troubleshooting

Covers README item 29. Common issues and how to resolve them. This is a
research prototype — some behaviour (missed objects, false positives) is
expected and documented in [Known limitations](#known-limitations-not-bugs).

---

## Troubleshooting table

| Symptom | Likely cause | What to try |
|---|---|---|
| `ModuleNotFoundError` on run | venv not activated / deps not installed | `.venv\Scripts\activate`, then `pip install -r requirements.txt` |
| Detector fails to load / downloads on first run | YOLO11n auto-downloads via Ultralytics on first use | Ensure internet for the first run only; afterwards it runs offline from `models/yolo11n.pt` |
| Depth model missing / depth disabled | `models/depth_anything_v2_small.onnx` not downloaded | `python scripts/download_models.py`; verify with `--check` |
| Depth INT8 file present but unused | INT8 variant was rejected (unsupported `ConvInteger` on onnxruntime 1.23.2) | Expected — the float32 model is used. See `models.md` |
| Camera won't open / `CameraError` | Wrong `camera.device_id`, camera in use, or no webcam | Close other apps using the camera; try `device_id: 1` in `config/config.yaml` |
| Black/blank frames | Camera covered or driver returning blanks | Uncover the lens; check the OS camera privacy settings |
| Low FPS | CPU load, other apps, or a slow camera driver | Close background apps; use `--headless`; FPS is measured, not guaranteed |
| No speech output | TTS unavailable, muted output, or non-Windows without a driver | Check volume; on Windows confirm SAPI5 voices; on Linux install `espeak-ng` |
| Wrong/absent voice name | `audio.sapi5_voice` not available on this machine | Set to an installed voice (e.g. `David` or `Zira` on Windows) |
| Too many alerts | Cooldown too short for your scene, or moving objects | `alerts.cooldown_seconds` is a starting estimate; adjust and re-test |
| Spurious "person" in an empty scene | Background misclassification (documented) | Phase 15 area gate reduces this; some large-area FP remain by design |
| Door/stairs not detected | Not COCO classes (architectural gap) | Expected and documented; do not rely on the system near stairs |
| A `requires_camera` test fails | Hardware/timing-dependent (e.g. raw FPS) | Not a logic failure; run `-m "not requires_camera"` for the logic suite |
| Debug window won't close | Focus not on the window | Click the window then press `Q`, or `Ctrl+C` in the terminal |
| Headless run needed (no display) | Server/CI or audio-only use | Run `python -m assistive_navigation --headless` |

---

## Known limitations (NOT bugs)

These are expected behaviours of a research prototype, documented across
`README.md`, `models.md`, and `evaluation.md`:

- **Relative (not metric) depth** — proximity buckets are ordinal. [UNVALIDATED ASSUMPTION]
- **Class-confusion mislabels** (e.g. table → bottle). [MEASURED — Phase 4]
- **Doors/stairs not detectable** (outside COCO classes). [MEASURED — Phase 4]
- **Remaining large-area false positives** that overlap the genuine size range. [MEASURED — Phase 15]
- **`couch` NOT TESTED** (no real couch was available).
- **Single-environment evaluation** — results do not generalize.

If you see one of these, it is a documented limitation, not a defect. Report
anything that contradicts the documented behaviour with the exact command,
`config/config.yaml` values, and console output.
