"""In-process message bus — a drop-in replacement for Kafka.

The Dockerised SereBench platform wires every microservice together through
Kafka topics (``{vehicle}_anomalies``, ``{vehicle}_weights``,
``global_weights``, ``{vehicle}_statistics`` ...). The offline simulation runs
every one of those services as a thread inside a single Python process, so
there is no broker to talk to. This module provides the tiny slice of Kafka
semantics the platform actually relies on:

* topic-addressed publish (:meth:`MessageBus.produce`),
* per-consumer subscriptions with the *same* rule librdkafka uses — a
  subscription string starting with ``^`` is a regular expression, anything
  else is an exact topic name,
* a blocking :meth:`BusConsumer.poll` returning ``(topic, value)`` or ``None``.

Values are passed by reference (they are ordinary Python objects, not
serialized bytes), which is exactly what we want for speed; the weight
reporters clone tensors before publishing so no producer and consumer ever
share a mutable ``state_dict``.
"""

import re
import queue
import threading


class BusConsumer:
    """The consumer side of :class:`MessageBus`, mirroring confluent-kafka.

    Each instance owns its own unbounded queue. Only messages published to a
    subscribed topic *after* the subscription is registered are delivered —
    the streaming services in SereBench never depend on historical replay, they
    just fill their buffers from the live stream.
    """

    def __init__(self, bus, group_id=None, maxsize=0):
        self._bus = bus
        self.group_id = group_id
        # A bounded queue with drop-oldest overflow gives the same practical
        # guarantee Kafka does: a slow consumer cannot make the broker grow
        # without bound. In the default rate-limited mode (time_emulation=True)
        # the queue never fills; only a deliberate firehose (time_emulation
        # =false) ever trips the cap, and then dropping the *oldest* telemetry
        # is exactly right — the buffers are reservoirs of recent data anyway.
        self._queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._exact = set()
        self._regex = []
        self._closed = False

    def subscribe(self, topics):
        """(Re)subscribe to a list of topic names / regexes.

        Same semantics as ``confluent_kafka.Consumer.subscribe``: a leading
        ``^`` marks a regular expression, otherwise the string is an exact
        topic name.
        """
        exact = set()
        regex = []
        for t in topics:
            if t.startswith("^"):
                regex.append(re.compile(t))
            else:
                exact.add(t)
        self._exact = exact
        self._regex = regex
        # Registering (again) is idempotent; the bus keeps a single reference.
        self._bus._register(self)

    def matches(self, topic):
        if topic in self._exact:
            return True
        return any(r.search(topic) for r in self._regex)

    def _deliver(self, topic, value):
        if self._closed:
            return
        try:
            self._queue.put_nowait((topic, value))
        except queue.Full:
            # Drop the oldest message to make room for the newest (Kafka-style
            # retention under lag), then account for it.
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((topic, value))
            except queue.Full:
                self.dropped += 1

    def poll(self, timeout=1.0):
        """Block up to ``timeout`` seconds for the next ``(topic, value)``.

        Returns ``None`` on timeout, exactly like ``Consumer.poll``.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self._closed = True
        self._bus._unregister(self)


class MessageBus:
    """A minimal, thread-safe, topic-addressed pub/sub broker."""

    def __init__(self):
        self._consumers = []
        self._topics = set()
        self._lock = threading.Lock()

    # -- producer side -----------------------------------------------------
    def produce(self, topic, value):
        self._topics.add(topic)
        with self._lock:
            targets = [c for c in self._consumers if c.matches(topic)]
        for c in targets:
            c._deliver(topic, value)

    def create_topic(self, topic):
        """Make a topic known to :meth:`list_topics` before anything is sent.

        The FL manager enumerates existing ``*_weights`` topics at startup to
        decide which vehicles to aggregate, so the orchestrator pre-declares
        them just like ``check_and_create_topics`` does on Kafka.
        """
        self._topics.add(topic)

    def list_topics(self):
        return set(self._topics)

    # -- consumer side -----------------------------------------------------
    def consumer(self, group_id=None, maxsize=0):
        """Create a consumer. ``maxsize`` > 0 bounds its queue (drop-oldest on
        overflow); 0 (default) leaves it unbounded, which is right for the
        low-volume control topics (weights / statistics)."""
        return BusConsumer(self, group_id=group_id, maxsize=maxsize)

    def _register(self, consumer):
        with self._lock:
            if consumer not in self._consumers:
                self._consumers.append(consumer)

    def _unregister(self, consumer):
        with self._lock:
            if consumer in self._consumers:
                self._consumers.remove(consumer)
