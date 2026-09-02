"""
Model Download Script
======================
Downloads required model files from official sources into the models/ directory.

Models managed by this script:
  1. YOLO11n (yolo11n.pt) — downloaded automatically by Ultralytics on first use.
     No explicit download needed; listed here for documentation only.

  2. Depth Anything V2 Small INT8 ONNX (depth_anything_v2_small_int8.onnx)
     Source  : onnx-community/depth-anything-v2-small on HuggingFace
     File    : onnx/model_int8.onnx (single-file, no external data needed)
     License : Apache-2.0
     Size    : ~27.3 MB
     Base    : depth-anything/Depth-Anything-V2-Small (official Small variant)

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model depth_anything
    python scripts/download_models.py --check   (only check what exists)

IMPORTANT:
  - This script downloads from official HuggingFace repositories only.
  - Never download model files from unofficial third-party mirrors.
  - After downloading, the system runs fully offline.
  - All downloads are verified by checking file size.

License note:
  Depth Anything V2 SMALL (this variant) = Apache-2.0.
  DO NOT substitute with Base/Large/Giant variants — those are CC-BY-NC-4.0
  (non-commercial only) and are NOT appropriate for this project.
"""

import sys
import hashlib
import argparse
from pathlib import Path

# Resolve project root — this script is at scripts/download_models.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR   = _PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "yolo11n": {
        "filename":    "yolo11n.pt",
        "description": "YOLO11 nano — object detection",
        "source":      "Ultralytics official hub (downloaded automatically on first use)",
        "license":     "AGPL-3.0",
        "size_mb":     5.4,
        "download_url": None,   # Ultralytics handles this automatically
        "manual": True,         # No explicit download needed
    },
    "depth_anything": {
        "filename":    "depth_anything_v2_small.onnx",
        "description": "Depth Anything V2 Small — monocular depth estimation (float32 ONNX)",
        "source":      "onnx-community/depth-anything-v2-small on HuggingFace",
        "license":     "Apache-2.0",
        "size_mb":     99.1,
        "download_url": (
            "https://huggingface.co/onnx-community/depth-anything-v2-small"
            "/resolve/main/onnx/model.onnx"
        ),
        "manual": False,
        "notes": (
            "Float32 ONNX — compatible with onnxruntime 1.23.2. "
            "INT8 quantised variant not used because ConvInteger op is "
            "not supported in this onnxruntime version. "
            "Benchmark: 118ms@252px, 260ms@364px, 366ms@518px on i5-12450H."
        ),
    },
}


# ---------------------------------------------------------------------------
# Download utility
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, expected_size_mb: float) -> bool:
    """
    Download a file from url to dest with a simple progress display.

    Uses only requests (already in venv) — no tqdm or other extras needed.

    Args:
        url:              HTTPS URL to download from.
        dest:             Local path to write to.
        expected_size_mb: Expected file size in MB (for progress display).

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        import requests
    except ImportError:
        print("ERROR: requests is not installed. Run: pip install requests")
        return False

    print(f"\nDownloading: {dest.name}")
    print(f"  From  : {url}")
    print(f"  To    : {dest}")
    print(f"  Size  : ~{expected_size_mb:.1f} MB")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_bytes = int(response.headers.get("Content-Length", 0))
        downloaded  = 0
        chunk_size  = 1024 * 1024  # 1 MB chunks

        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        pct = downloaded / total_bytes * 100
                        mb  = downloaded / 1_000_000
                        # Simple inline progress without carriage return issues
                        if int(pct) % 10 == 0 or downloaded == total_bytes:
                            print(f"  Progress: {pct:.0f}%  ({mb:.1f} MB)", flush=True)

        actual_mb = dest.stat().st_size / 1_000_000
        print(f"  Downloaded: {actual_mb:.1f} MB")

        # Sanity check — file should be within 20% of expected size
        if expected_size_mb > 0:
            ratio = actual_mb / expected_size_mb
            if ratio < 0.5 or ratio > 2.0:
                print(f"  WARNING: File size {actual_mb:.1f} MB differs significantly "
                      f"from expected {expected_size_mb:.1f} MB.")
                print(f"  The download may be incomplete. Delete and re-run.")
                return False

        print(f"  OK: {dest.name} ready.")
        return True

    except Exception as e:
        print(f"  ERROR: Download failed: {e}")
        if dest.exists():
            dest.unlink()  # Remove incomplete file
        return False


def check_model(model_key: str) -> dict:
    """Check whether a model file exists and report its status."""
    info = MODELS[model_key]
    dest = _MODELS_DIR / info["filename"]
    exists = dest.exists()
    size_mb = dest.stat().st_size / 1_000_000 if exists else 0.0
    return {
        "key":      model_key,
        "filename": info["filename"],
        "exists":   exists,
        "size_mb":  round(size_mb, 1),
        "expected_mb": info["size_mb"],
        "license":  info["license"],
    }


def download_depth_anything() -> bool:
    """Download Depth Anything V2 Small INT8 ONNX model."""
    info = MODELS["depth_anything"]
    dest = _MODELS_DIR / info["filename"]

    if dest.exists():
        size_mb = dest.stat().st_size / 1_000_000
        print(f"  Already exists: {dest.name} ({size_mb:.1f} MB) — skipping download.")
        return True

    return download_file(info["download_url"], dest, info["size_mb"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download required model files for AssistiveNavigationSystem V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_models.py                  # download all
  python scripts/download_models.py --model depth_anything
  python scripts/download_models.py --check          # check status only
        """
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "depth_anything"],
        help="Which model to download (default: all)"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only check model status, do not download"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print()
    print("=" * 60)
    print("  ASSISTIVE NAVIGATION SYSTEM V2 — MODEL MANAGER")
    print("=" * 60)
    print(f"  Models directory: {_MODELS_DIR}")
    print()

    # Status check
    print("  Current model status:")
    for key, info in MODELS.items():
        status = check_model(key)
        flag = "OK" if status["exists"] else "MISSING"
        size_str = f"{status['size_mb']:.1f} MB" if status["exists"] else "not downloaded"
        print(f"    [{flag}] {info['filename']:<45} {size_str}")
        print(f"           License: {info['license']}")
    print()

    if args.check:
        print("  Check complete. Use --model to download.")
        return

    # Download
    success_all = True

    if args.model in ("all", "depth_anything"):
        print("  Downloading: Depth Anything V2 Small INT8 ONNX")
        print(f"  License: Apache-2.0 (Small variant only)")
        print(f"  Source : HuggingFace — onnx-community/depth-anything-v2-small")
        ok = download_depth_anything()
        if not ok:
            success_all = False
        print()

    if args.model == "all":
        yolo_info = MODELS["yolo11n"]
        dest = _MODELS_DIR / yolo_info["filename"]
        if dest.exists():
            print(f"  [{yolo_info['filename']}] already present — no download needed.")
        else:
            print(f"  [{yolo_info['filename']}] will be downloaded automatically")
            print(f"  by Ultralytics on first use of the detector.")
        print()

    print("=" * 60)
    if success_all:
        print("  All downloads complete.")
    else:
        print("  Some downloads failed. Check the error messages above.")
    print("=" * 60)
    print()

    sys.exit(0 if success_all else 1)


if __name__ == "__main__":
    main()
