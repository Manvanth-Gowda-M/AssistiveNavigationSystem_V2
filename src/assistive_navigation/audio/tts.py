"""
Text-to-Speech Module
=====================
Converts alert text to spoken audio using a local TTS engine.

Phase: 11 (not yet implemented)
Status: STUB — placeholder only

Phase 11 default: pyttsx3 + Windows SAPI5
  - Zero additional setup — voices David and Zira already installed
  - Lowest latency (~50–150ms)
  - Fully offline

Phase 11 upgrade path: piper-tts-plus
  - Natural neural voice quality
  - MIT license
  - Slightly higher latency (~100–300ms)
  - Install: pip install piper-tts-plus

Speech runs in a background thread so it NEVER blocks the detection loop.
Audio output goes to the default system speaker/headphones.
No audio data is uploaded anywhere.
"""

# Implementation begins in Phase 11.
