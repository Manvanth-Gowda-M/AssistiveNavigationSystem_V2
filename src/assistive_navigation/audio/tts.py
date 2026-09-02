"""
Text-to-Speech Module
=====================
Converts text strings to spoken audio using a local, offline TTS engine.

Phase: 11
Status: IMPLEMENTED

Engine: pyttsx3 (MIT licence) wrapping Windows SAPI5
  - Zero additional setup — David and Zira voices already installed
  - Fully offline — no network access during normal operation
  - Confirmed working on this machine (Phase 11 hardware test)

Upgrade path (documented, NOT implemented in Phase 11):
  - piper-tts-plus (MIT) produces more natural neural speech
  - Requires a one-time ~100-200 MB voice model download
  - Can replace pyttsx3 without changing the calling interface
  - Config key: audio.engine = "piper"  (switch in Phase 14+)

What this module does:
  - Initialises the pyttsx3/SAPI5 engine once at startup
  - Exposes speak() for non-blocking speech (background thread)
  - Exposes speak_sync() for blocking speech (testing / direct use)
  - Handles all failures gracefully — never crashes the pipeline
  - Logs all failures as warnings

What this module does NOT do:
  - Does NOT queue multiple messages (SimpleAudioQueue does that)
  - Does NOT decide WHAT to say (Phase 12 — AlertManager)
  - Does NOT apply cooldown or deduplication (Phase 12)
  - Does NOT upload audio data anywhere
  - Does NOT access the internet

SAFETY NOTE:
  TTS is a research prototype component. Audio output quality is not
  validated for independent navigation guidance. This module must not
  be relied upon as the sole safety mechanism.

Usage:
    from assistive_navigation.audio.tts import TextToSpeech
    from assistive_navigation.utils.config_loader import load_config

    config = load_config()
    tts = TextToSpeech(config)
    tts.initialise()

    tts.speak("Person ahead, very close.")   # non-blocking
    tts.speak_sync("Chair on your left.")    # blocking (waits for finish)
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Local offline text-to-speech using pyttsx3 + Windows SAPI5.

    Designed to be initialised once at startup and reused throughout the
    pipeline lifetime. Speech never blocks the calling thread when
    non_blocking=True (the default from config).

    If the engine fails to initialise (e.g. on a headless server with no
    audio device), is_available returns False and all speak() calls become
    silent no-ops. The pipeline is never interrupted.

    Args:
        config (dict): Full configuration from config.yaml.
                       Uses the 'audio' section.
    """

    def __init__(self, config: dict) -> None:
        audio_cfg = config.get("audio", {})

        self._engine_name:   str   = audio_cfg.get("engine",       "sapi5")
        self._voice_name:    str   = audio_cfg.get("sapi5_voice",  "Zira")
        self._speech_rate:   int   = int(audio_cfg.get("speech_rate", 175))
        self._volume:        float = float(audio_cfg.get("volume",   0.9))
        self._non_blocking:  bool  = bool(audio_cfg.get("non_blocking", True))

        # pyttsx3 engine object — None until initialise() is called
        self._engine = None
        self._is_available: bool = False

        # Lock to prevent concurrent access to the pyttsx3 engine.
        # pyttsx3 is NOT thread-safe — only one thread may call it at a time.
        self._engine_lock = threading.Lock()

        # Background thread handle for non-blocking speech
        self._speech_thread: Optional[threading.Thread] = None

        logger.debug(
            "TextToSpeech created — engine=%s, voice=%s, rate=%d, vol=%.1f",
            self._engine_name, self._voice_name, self._speech_rate, self._volume,
        )

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def initialise(self) -> bool:
        """
        Initialise the pyttsx3 TTS engine.

        Must be called once before speak() or speak_sync().
        Safe to call again if already initialised (returns True immediately).

        Returns:
            True if the engine initialised successfully.
            False if initialisation failed (e.g. no audio device).
            On failure, all speak() calls become silent no-ops.
        """
        if self._is_available:
            return True

        try:
            import pyttsx3
            engine = pyttsx3.init()

            # Configure rate and volume
            engine.setProperty("rate",   self._speech_rate)
            engine.setProperty("volume", self._volume)

            # Select the configured voice by name
            voices = engine.getProperty("voices")
            matched = False
            if voices:
                for v in voices:
                    if self._voice_name.lower() in v.name.lower():
                        engine.setProperty("voice", v.id)
                        matched = True
                        logger.info(
                            "TTS voice selected: %s (id=%s)", v.name, v.id
                        )
                        break
                if not matched:
                    # Voice name not found — use first available voice
                    engine.setProperty("voice", voices[0].id)
                    logger.warning(
                        "TTS voice '%s' not found. Using '%s' instead.",
                        self._voice_name, voices[0].name,
                    )

            self._engine = engine
            self._is_available = True
            logger.info(
                "TextToSpeech initialised — engine=%s, rate=%d, volume=%.1f",
                self._engine_name, self._speech_rate, self._volume,
            )
            return True

        except Exception as e:
            self._engine = None
            self._is_available = False
            logger.warning(
                "TextToSpeech initialisation failed: %s\n"
                "All speak() calls will be silent no-ops. "
                "Check that an audio device is available and pyttsx3 is installed.",
                e,
            )
            return False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """
        Speak the given text.

        If non_blocking=True (config default): speech runs in a background
        thread. The calling thread returns immediately. If the previous
        non-blocking speech is still in progress, this call waits briefly
        for it to finish before starting (to avoid overlapping speech at
        this layer). Full overlap prevention is SimpleAudioQueue's job.

        If non_blocking=False: blocks until speech completes.

        Empty or None text is silently ignored.
        Any TTS failure is logged as a warning and silently swallowed.

        Args:
            text: The text to speak. May be any string.
        """
        if not self._is_available:
            logger.debug("TTS unavailable — skipping: %r", text)
            return

        text = _sanitise(text)
        if not text:
            return

        if self._non_blocking:
            # Wait for previous non-blocking thread to finish if still running.
            # This prevents audio overlap when speak() is called rapidly.
            if self._speech_thread and self._speech_thread.is_alive():
                self._speech_thread.join(timeout=5.0)
                if self._speech_thread.is_alive():
                    logger.warning(
                        "TTS: previous speech thread still running after 5s timeout. "
                        "Skipping: %r", text
                    )
                    return

            self._speech_thread = threading.Thread(
                target=self._speak_in_thread,
                args=(text,),
                daemon=True,
                name=f"tts-{hash(text) & 0xFFFF:04x}",
            )
            self._speech_thread.start()
        else:
            self.speak_sync(text)

    def speak_sync(self, text: str) -> None:
        """
        Speak text and block until speech is complete.

        Intended for testing and for use by SimpleAudioQueue's consumer
        thread (which handles serialisation at a higher level).

        Empty or None text is silently ignored.
        Any failure is logged and swallowed.

        Args:
            text: The text to speak.
        """
        if not self._is_available:
            logger.debug("TTS unavailable (sync) — skipping: %r", text)
            return

        text = _sanitise(text)
        if not text:
            return

        self._speak_in_thread(text)

    def stop(self) -> None:
        """
        Attempt to stop any currently playing speech.

        Not guaranteed to interrupt mid-word on all platforms.
        Waits for the background thread to finish if non-blocking.
        """
        if self._engine is not None:
            try:
                with self._engine_lock:
                    self._engine.stop()
            except Exception as e:
                logger.debug("TTS stop() error (non-critical): %s", e)

        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join(timeout=2.0)

    def shutdown(self) -> None:
        """
        Shut down the TTS engine cleanly.

        Call once when the application exits. After this, is_available
        returns False and speak() is a no-op.
        """
        self.stop()
        if self._engine is not None:
            try:
                with self._engine_lock:
                    self._engine.stop()
            except Exception:
                pass
        self._engine = None
        self._is_available = False
        logger.info("TextToSpeech shut down.")

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _speak_in_thread(self, text: str) -> None:
        """Run pyttsx3 synthesis. Called either directly or from a Thread."""
        try:
            with self._engine_lock:
                self._engine.say(text)
                self._engine.runAndWait()
            logger.debug("TTS spoke: %r", text)
        except Exception as e:
            logger.warning("TTS speech failed for %r: %s", text, e)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if the TTS engine initialised successfully."""
        return self._is_available

    @property
    def is_speaking(self) -> bool:
        """True if a non-blocking speech thread is currently running."""
        return bool(self._speech_thread and self._speech_thread.is_alive())

    @property
    def engine_name(self) -> str:
        """Name of the configured TTS engine."""
        return self._engine_name

    @property
    def voice_name(self) -> str:
        """Configured voice name."""
        return self._voice_name

    @property
    def speech_rate(self) -> int:
        """Configured speech rate (words per minute)."""
        return self._speech_rate

    def __repr__(self) -> str:
        return (
            f"TextToSpeech(engine={self._engine_name}, "
            f"voice={self._voice_name}, "
            f"rate={self._speech_rate}, "
            f"available={self._is_available})"
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sanitise(text) -> str:
    """
    Sanitise input text before speaking.
    Returns empty string for None, non-string, or whitespace-only input.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    return text.strip()
