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

  The above are Phase 12 responsibilities. Any attempt to add them here
  would violate the phase boundary.

What this module does:
  - Provides a thread-safe queue (stdlib queue.Queue)
  - Runs a single background consumer thread that reads messages one at
    a time and calls TextToSpeech.speak_sync() for each
  - Guarantees that messages are spoken in FIFO order without overlap
  - Provides start() and stop() for lifecycle management
  - stop() waits for the current message to finish before returning
  - Handles TTS failures gracefully — a failed message is discarded
    and the next message is processed

What this module does NOT do:
  - Does not prioritise any message over another
  - Does not suppress repeated identical messages
  - Does not apply any time-based cooldowns
  - Does not upload or transmit audio data anywhere
  - Does not require network access

Design note on overlapping speech:
  The consumer thread processes one message at a time via speak_sync().
  Callers that put() messages faster than speech can deliver them will
  accumulate a backlog in the queue. Phase 12 will decide whether old
  messages should be dropped or deprioritised. In Phase 11, all messages
  are kept and spoken in order.

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

    # ... later at shutdown ...
    q.stop()
"""

import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel value placed in the queue to signal the consumer to stop
_STOP_SENTINEL = None


class SimpleAudioQueue:
    """
    Thread-safe FIFO queue that serialises speech through TextToSpeech.

    One background consumer thread continuously reads text from the queue
    and calls tts.speak_sync() for each message. This ensures:
      1. Messages are spoken in the order they were put()
      2. No two messages overlap (speak_sync blocks until complete)
      3. The calling thread is never blocked by speech

    Phase 11 provides only FIFO ordering. Phase 12 will introduce
    priority, cooldown, and deduplication on top of this queue.

    Args:
        tts: A TextToSpeech instance that has been initialised.
             If tts.is_available is False, put() is still safe — messages
             are enqueued but silently dropped when the consumer processes them.
    """

    def __init__(self, tts) -> None:
        # TextToSpeech instance
        self._tts = tts

        # Thread-safe FIFO queue (maxsize=0 means unlimited)
        self._queue: queue.Queue = queue.Queue(maxsize=0)

        # Consumer thread
        self._consumer_thread: Optional[threading.Thread] = None
        self._running: bool = False

        logger.debug("SimpleAudioQueue created.")

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the background consumer thread.

        Safe to call multiple times — if already running, does nothing.
        """
        if self._running:
            logger.debug("SimpleAudioQueue already running.")
            return

        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._consume,
            daemon=True,
            name="audio-queue-consumer",
        )
        self._consumer_thread.start()
        logger.info("SimpleAudioQueue consumer thread started.")

    def stop(self) -> None:
        """
        Stop the consumer thread cleanly.

        Sends a stop sentinel to the queue, then waits for the consumer
        thread to finish the current message and exit.

        Blocks until the consumer thread exits (maximum ~10 seconds to
        allow any currently-playing speech to finish).

        After stop(), the queue can be restarted with start().
        """
        if not self._running:
            return

        self._running = False
        # Send sentinel to unblock the consumer if it is waiting
        self._queue.put(_STOP_SENTINEL)

        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=10.0)
            if self._consumer_thread.is_alive():
                logger.warning(
                    "SimpleAudioQueue: consumer thread did not stop within "
                    "10 seconds. It may be blocked in speak_sync()."
                )

        self._consumer_thread = None
        logger.info("SimpleAudioQueue stopped.")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def put(self, text: str) -> None:
        """
        Add a text message to the speech queue.

        Thread-safe. Returns immediately without blocking.
        Empty or None text is silently discarded.

        Args:
            text: The text message to speak.
        """
        if text is None or (isinstance(text, str) and not text.strip()):
            logger.debug("SimpleAudioQueue: discarding empty/None text.")
            return

        if not isinstance(text, str):
            text = str(text)

        self._queue.put(text)
        logger.debug("SimpleAudioQueue: enqueued %r (qsize=%d)", text, self._queue.qsize())

    def clear(self) -> None:
        """
        Discard all pending messages in the queue.

        Does NOT stop currently-playing speech.
        Useful if the navigation context has changed (e.g. object disappeared).

        NOTE: Phase 12 will add smarter logic for deciding when to clear.
        """
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            logger.debug("SimpleAudioQueue: cleared %d pending messages.", cleared)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True if the consumer thread is running."""
        return (
            self._running and
            self._consumer_thread is not None and
            self._consumer_thread.is_alive()
        )

    @property
    def qsize(self) -> int:
        """Approximate number of messages waiting in the queue."""
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        """True if no messages are waiting."""
        return self._queue.empty()

    # -----------------------------------------------------------------------
    # Consumer thread
    # -----------------------------------------------------------------------

    def _consume(self) -> None:
        """
        Consumer thread body.

        Continuously reads from the queue and calls speak_sync().
        Exits when _STOP_SENTINEL is received or _running is False.
        """
        logger.debug("SimpleAudioQueue: consumer thread running.")

        while self._running:
            try:
                # Block until a message is available (or stop sentinel arrives)
                item = self._queue.get(block=True, timeout=1.0)
            except queue.Empty:
                # No message in the last second — loop and check _running again
                continue

            # Stop sentinel
            if item is _STOP_SENTINEL:
                logger.debug("SimpleAudioQueue: consumer received stop sentinel.")
                break

            # Speak the message — speak_sync() handles failures internally
            try:
                self._tts.speak_sync(item)
            except Exception as e:
                logger.warning(
                    "SimpleAudioQueue: speak_sync failed for %r: %s", item, e
                )
            finally:
                self._queue.task_done()

        logger.debug("SimpleAudioQueue: consumer thread exiting.")

    def __repr__(self) -> str:
        return (
            f"SimpleAudioQueue(running={self.is_running}, "
            f"qsize={self.qsize})"
        )
