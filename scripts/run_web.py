"""
Launch Assistive Navigation System V2 Testing Web Studio
=========================================================

Usage:
    python scripts/run_web.py
    python scripts/run_web.py --port 8000 --open-browser
    python scripts/run_web.py --host 0.0.0.0 --port 8080
"""

import argparse
import sys
import webbrowser
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Assistive Navigation Testing Studio Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open web browser on start")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")

    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 70)
    print("  ASSISTIVE NAVIGATION SYSTEM V2 — TESTING STUDIO")
    print(f"  Server URL: {url}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 70)

    if args.open_browser:
        import threading
        import time
        threading.Thread(target=lambda: (time.sleep(1.0), webbrowser.open(url)), daemon=True).start()

    uvicorn.run(
        "assistive_navigation.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
