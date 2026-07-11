"""Network delay + jitter simulation — the offline counterpart of
``OpenFAIR/network_delay.py`` (and the companion of this package's
``packet_loss.py``).

Neither Kafka nor :class:`~offline_simulation.message_bus.MessageBus` adds any
transit latency to an accepted write, so a slow / jittery uplink is simulated
on the sending side: instead of handing a message straight to the bus, the
caller hands us a ``deliver`` callable which we run after a sampled latency of
``max(0, delay_mean_ms + N(0, jitter_std_ms))`` milliseconds, on a private
daemon worker thread so the sender's generation rate is not throttled. Kept in
lockstep with the Dockerised platform's ``OpenFAIR/network_delay.py`` (same
behaviour, same ``stats()`` shape) so ``delay_mean_ms`` / ``jitter_std_ms`` set
in ``config/default.yaml`` produce the same added-latency behaviour in both
modalities.

At the default (0, 0) setting the send runs inline on the caller's thread and
no worker thread is started — a zero-overhead no-op that never perturbs a
baseline run.
"""

import heapq
import random
import threading
import time


class NetworkDelaySimulator:
    """Client-side latency + jitter injector for one outbound message stream."""

    def __init__(self, delay_mean_ms=0.0, jitter_std_ms=0.0):
        self.delay_mean_ms = max(0.0, float(delay_mean_ms or 0.0))
        self.jitter_std_ms = max(0.0, float(jitter_std_ms or 0.0))
        self.sent = 0
        self.total_delay_ms = 0.0
        self.max_delay_ms = 0.0

        self._heap = []
        self._counter = 0
        self._cv = threading.Condition()
        self._worker = None
        self._closed = False

    @property
    def enabled(self):
        return self.delay_mean_ms > 0.0 or self.jitter_std_ms > 0.0

    def sample_delay_ms(self):
        if not self.enabled:
            return 0.0
        d = self.delay_mean_ms
        if self.jitter_std_ms > 0.0:
            d += random.gauss(0.0, self.jitter_std_ms)
        return d if d > 0.0 else 0.0

    def send(self, deliver):
        """Push ``deliver`` (a zero-arg callable performing the real
        ``bus.produce``) through the simulated link — after the sampled delay on
        the worker thread, or inline when no delay is configured."""
        delay_ms = self.sample_delay_ms()
        self.sent += 1
        self.total_delay_ms += delay_ms
        if delay_ms > self.max_delay_ms:
            self.max_delay_ms = delay_ms

        if delay_ms <= 0.0:
            deliver()
            return

        self._ensure_worker()
        deliver_at = time.monotonic() + delay_ms / 1000.0
        with self._cv:
            heapq.heappush(self._heap, (deliver_at, self._counter, deliver))
            self._counter += 1
            self._cv.notify()

    def _ensure_worker(self):
        if self._worker is not None:
            return
        with self._cv:
            if self._worker is None and not self._closed:
                self._worker = threading.Thread(
                    target=self._run, name="network-delay", daemon=True)
                self._worker.start()

    def _run(self):
        while True:
            with self._cv:
                while not self._closed and not self._heap:
                    self._cv.wait()
                if self._closed and not self._heap:
                    return
                deliver_at, _, deliver = self._heap[0]
                wait = deliver_at - time.monotonic()
                if wait > 0:
                    self._cv.wait(timeout=wait)
                    continue
                heapq.heappop(self._heap)
            try:
                deliver()
            except Exception:
                pass

    def close(self, drain=True):
        """Stop the worker; deliver still-pending messages first when ``drain``."""
        with self._cv:
            self._closed = True
            pending = [item[2] for item in self._heap] if drain else []
            self._heap = []
            self._cv.notify_all()
        for deliver in pending:
            try:
                deliver()
            except Exception:
                pass

    @property
    def observed_avg_delay_ms(self):
        return (self.total_delay_ms / self.sent) if self.sent else 0.0

    def stats(self):
        return {
            'delay_mean_ms': self.delay_mean_ms,
            'jitter_std_ms': self.jitter_std_ms,
            'messages_sent': self.sent,
            'avg_applied_delay_ms': self.observed_avg_delay_ms,
            'max_applied_delay_ms': self.max_delay_ms,
        }
