"""Packet-loss simulation — the offline counterpart of OpenFAIR/packet_loss.py.

Neither Kafka nor :class:`~offline_simulation.message_bus.MessageBus` ever
drops an accepted write, so a lossy uplink is simulated on the sending side:
before a message would be handed to the bus, a Bernoulli(packet_loss_rate)
draw decides whether it is silently discarded instead. Kept in lockstep with
the Dockerised platform's ``OpenFAIR/packet_loss.py`` (same behaviour, same
stats() shape) so a ``packet_loss_rate`` set in ``config/default.yaml``
produces the same drop behaviour in both modalities.
"""

import random


class PacketLossSimulator:
    """Client-side packet-loss injector for one outbound message stream."""

    def __init__(self, packet_loss_rate=0.0):
        self.packet_loss_rate = max(0.0, min(1.0, float(packet_loss_rate or 0.0)))
        self.sent = 0
        self.dropped = 0

    def should_drop(self):
        """Return True if the caller should discard the message instead of sending it."""
        if self.packet_loss_rate <= 0.0:
            self.sent += 1
            return False
        if random.random() < self.packet_loss_rate:
            self.dropped += 1
            return True
        self.sent += 1
        return False

    @property
    def total(self):
        return self.sent + self.dropped

    @property
    def observed_loss_rate(self):
        return (self.dropped / self.total) if self.total else 0.0

    def stats(self):
        return {
            'packet_loss_rate': self.packet_loss_rate,
            'packets_sent': self.sent,
            'packets_dropped': self.dropped,
            'observed_loss_rate': self.observed_loss_rate,
        }
