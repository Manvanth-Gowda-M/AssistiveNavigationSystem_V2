"""
Audio Queue Module — Phase 11 Minimal Implementation
======================================================
Thread-safe FIFO queue that serialises text messages to the TTS engine,
ensuring spoken messages do not overlap.

Phase: 11
Status: IMPLEMENTED (minimal FIFO only)

SCOPE RESTRICTION — Phase 11 only implements FIFO serialisation:
  This module does NOT implement:
    - Priority ordering                        (Phase 12)
    - Deduplication / duplicate suppression   (Phase 12)
    - Per-object cooldown timers              (Phase 12)
    - Speech interruption                      (Phase 12)
    - Proximity-based speech decisions         (Phase 12)
    - Alert decision logic                     (Phase 12)

Windows/pyttsx3 threading note:
    pyttsx3's runAndWait() can hang indefinitely when called from a
    secondary thread on Windows because SAPI5's COM event loop is
    not re-entrant from non-main threads.

    This implementation uses a dedicated speech worker thread with its
    own pyttsx3 engine instance (separate from TextToSpeech) to avoid
    this limitation. The consumer queues text to that dedicated worker.

What this module does:
  - Provides a thread-safe queue (stdlib queue.Queue)
  - Runs a dedicated speech-worker thread with its own pyttsx3 engine
  - Consumer reads from queue and forwards to the speech worker
  - Guarantees that messages are spoken in FIFO order without overlap
  - Provides start() and stop() for lifecycle management

Usage:
    from assistive_navigation.audio.tts import TextToSpeech
    from assistive_navigation.audio.queue import SimpleAudioQueue
    from assistive_navigation.utils.config_loader import load_config

    config = load_config()
    tts = TextToSpeech(config)
    tts.initialise()

    q = SimpleAudioQueue(tts)
    q.start()

    q.put("Person ahead, very close.")
    q.put("Chair on your left.")

    q.stop()
"""

import logging
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_STOP_SENTINEL = None


class SimpleAudioQueue:
    """
    Thread-safe FIFO queue that serialises speech messages.

    Uses a dedicated speech worker thread with its own pyttsx3 engine
    instance to avoid the SAPI5/pyttsx3 cross-thread limitation on Windows.

    Phase 11 provides only FIFO ordering. Phase 12 will introduce
    priority, cooldown, and deduplication on top of this queue.

    Args:
        tts: A TextToSpeech instance. Used only to read config settings
             (voice, rate, volume). The queue creates its own internal
             engine for background speech.
    """

    def __init__(self, tts) -> None:
        self._tts = tts
        self._queue: queue.Queue = queue.Queue(maxsize=0)
        self._worker_thread: Optional[threading.Thread] = None
        self._running: bool = False
        logger.debug("SimpleAudioQueue created.")

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the background speech worker thread. Idempotent."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="audio-queue-worker",
        )
        self._worker_thread.start()
        logger.info("SimpleAudioQueue worker thread started.")

    def stop(self) -> None:
        """
        Stop the speech worker thread.

        Sends a sentinel, then waits up to 3 seconds for the thread to exit.
        The worker exits cleanly after finishing any message in progress.
        """
        if not self._running:
            return
        self._running = False
        self._queue.put(_STOP_SENTINEL)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        self._worker_thread = None
        logger.info("SimpleAudioQueue stopped.")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def put(self, text: str) -> None:
        """
        Add a text message to the speech queue.
        Thread-safe. Empty or None text is silently discarded.
        """
        if text is None or (isinstance(text, str) and not text.strip()):
            return
        if not isinstance(text, str):
            text = str(text)
        self._queue.put(text)
        logger.debug("SimpleAudioQueue: enqueued %r (qsize=%d)", text, self._queue.qsize())

    def clear(self) -> None:
        """Discard all pending messages. Does not interrupt current speech."""
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            logger.debug("SimpleAudioQueue: cleared %d messages.", cleared)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return (
            self._running and
            self._worker_thread is not None and
            self._worker_thread.is_alive()
        )

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    # -----------------------------------------------------------------------
    # Speech worker
    # -----------------------------------------------------------------------

    def _worker(self) -> None:
        """
        Background worker thread.

        Creates its own pyttsx3 engine so that runAndWait() runs in this
        dedicated thread — avoiding the SAPI5 cross-thread deadlock.
        """
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Mirror config settings from the TextToSpeech instance
            engine.setProperty("rate",   self._tts.speech_rate)
            engine.setProperty("volume", getattr(self._tts, '_volume', 0.9))
            # Mirror voice selection
            voices = engine.getProperty("voices")
            if voices:
                for v in voices:
                    if self._tts.voice_name.lower() in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
            logger.debug("SimpleAudioQueue worker: pyttsx3 engine ready.")
        except Exception as e:
            logger.warning("SimpleAudioQueue worker: could not init pyttsx3: %s", e)
            engine = None

        while self._running:
            try:
                item = self._queue.get(block=True, timeout=0.5)
            except queue.Empty:
                continue

            if item is _STOP_SENTINEL:
                logger.debug("SimpleAudioQueue worker: received sentinel, exiting.")
                break

            if engine is not None:
                try:
                    engine.say(item)
                    engine.runAndWait()
                    logger.debug("SimpleAudioQueue worker: spoke %r", item)
                except Exception as e:
                    logger.warning("SimpleAudioQueue worker: speech failed %r: %s", item, e)
            else:
                # No engine — fall back to TextToSpeech.speak_sync()
                try:
                    self._tts.speak_sync(item)
                except Exception as e:
                    logger.warning("SimpleAudioQueue worker: fallback failed %r: %s", item, e)

            try:
                self._queue.task_done()
            except Exception:
                pass

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        logger.debug("SimpleAudioQueue worker: thread exiting.")

    def __repr__(self) -> str:
        return f"SimpleAudioQueue(running={self.is_running}, qsize={self.qsize})"
