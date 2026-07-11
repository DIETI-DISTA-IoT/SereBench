"""FLManagerNode — the in-process equivalent of of-fedlearningmanager.

A faithful port of ``manager_server.py``'s ``FederatedLearningManager``:

* builds a global model of the *same* architecture as the vehicles (via the
  vendored :func:`build_model`) and seeds it with the configured init strategy;
* subscribes to every ``{vehicle}_weights`` topic, buffering the latest
  ``state_dict`` per vehicle in a small FIFO;
* every ``aggregation_interval_secs`` (or on each message when the interval is
  0), once *all* vehicle buffers hold at least one update, aggregates them with
  the selected pluggable strategy (FedAvg / FedYogi / FedMedian / FedProx),
  guards against NaNs exactly as upstream does, loads the result into the
  global model and republishes it on ``global_weights`` for the consumers to
  pull.

Kafka is swapped for :class:`MessageBus`; the aggregation maths, the
stateful-vs-function strategy dispatch, the NaN guards and the buffer
pop-after-aggregate semantics are unchanged.
"""

import time
import logging
import threading

import torch

from .models import build_model
from .buffers import GenericBuffer
from .aggregation import federated_averaging, FedYogi, fed_median, fed_prox
from .packet_loss import PacketLossSimulator
from .network_delay import NetworkDelaySimulator

FEDERATED_LEARNING = "FEDERATED_LEARNING"

aggregation_functions = {
    "fedavg": federated_averaging,
    "fedyogi": FedYogi,
    "fedmedian": fed_median,
    "fedprox": fed_prox,
}


class FLManagerNode:
    def __init__(self, config, bus):
        self.config = dict(config)
        self.bus = bus
        self.logger = logging.getLogger(FEDERATED_LEARNING)

        self.aggregation_interval_secs = self.config.get('aggregation_interval_secs', 25)

        self.global_model = build_model(**self.config)
        self.global_model.initialize_weights(self.config['initialization_strategy'])
        self.logger.info(
            f"Global model ({str(self.config.get('model_type', 'mlp')).lower()}) initialized "
            f"using {self.config['initialization_strategy']} initialization.")

        strategy = self.config.get('aggregation_strategy')
        try:
            aggregator = aggregation_functions[strategy]
        except KeyError:
            raise ValueError(
                f"Unknown aggregation strategy '{strategy}'. Available: {sorted(aggregation_functions)}")
        # Stateful strategies (FedYogi) are classes: instantiate for a fresh
        # optimizer state per FL session. Functions are used directly.
        self.aggregation_function = aggregator() if isinstance(aggregator, type) else aggregator

        self.vehicle_weights_topics = self._discover_weight_topics()
        self.weights_buffer = {
            topic: GenericBuffer(size=self.config.get('weights_buffer_size', 5), label=topic)
            for topic in self.vehicle_weights_topics
        }
        self.logger.info(
            f"Starting FL for {len(self.vehicle_weights_topics)} vehicles: "
            f"{self.vehicle_weights_topics}")

        self.packet_loss = PacketLossSimulator(self.config.get('packet_loss_rate', 0.1))
        self.network_delay = NetworkDelaySimulator(
            self.config.get('delay_mean_ms', 0.0), self.config.get('jitter_std_ms', 0.0))

        self._agg_round = 0
        self._stop = False
        self._stop_event = threading.Event()
        self._consuming_thread = None
        self._aggregation_thread = None
        self._weights_consumer = None

    def _discover_weight_topics(self):
        return sorted(t for t in self.bus.list_topics()
                      if t.endswith("_weights") and t != "global_weights")

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._stop = False
        self._stop_event.clear()
        self.bus.create_topic("global_weights")
        self.bus.create_topic("global_metrics")

        self._weights_consumer = self.bus.consumer(group_id=FEDERATED_LEARNING)
        self._weights_consumer.subscribe(list(self.vehicle_weights_topics))

        self._consuming_thread = threading.Thread(target=self._consume_weights_data, daemon=True,
                                                  name="fl_consume")
        self._consuming_thread.start()

        if self.aggregation_interval_secs > 0:
            self._aggregation_thread = threading.Thread(target=self._aggregate_periodically,
                                                       daemon=True, name="fl_aggregate")
            self._aggregation_thread.start()

    def stop(self):
        self._stop = True
        self._stop_event.set()
        if self._consuming_thread:
            self._consuming_thread.join(timeout=self.aggregation_interval_secs + 5)
        if self._aggregation_thread:
            self._aggregation_thread.join(timeout=self.aggregation_interval_secs + 5)
        if self._weights_consumer is not None:
            self._weights_consumer.close()
        # Flush any global-weights update still held by the simulated link.
        self.network_delay.close()
        self.logger.info("Federated learning manager stopped.")

    # -- threads (mirror manager_server.py) --------------------------------
    def _consume_weights_data(self):
        try:
            while not self._stop:
                msg = self._weights_consumer.poll(1.0)
                if msg is None:
                    continue
                topic, value = msg
                self._process_message(topic, value)
        finally:
            pass

    def _process_message(self, topic, msg):
        if topic in self.weights_buffer:
            vehicle_name = topic.replace("_weights", "")
            self.logger.info(f"Received local weights from {vehicle_name}.")
            self.weights_buffer[topic].add(msg)
        if self.aggregation_interval_secs == 0:
            self._aggregate_weights()

    def _aggregate_periodically(self):
        while not self._stop:
            self._stop_event.wait(timeout=self.aggregation_interval_secs)
            if self._stop:
                break
            try:
                self._aggregate_weights()
            except Exception:
                self.logger.exception("Aggregation round failed; will retry on next interval.")

    def _aggregate_weights(self):
        if not self.weights_buffer:
            return
        if not all(len(buf) > 0 for buf in self.weights_buffer.values()):
            self.logger.info("Waiting for more data to aggregate the weights.")
            return

        self._agg_round += 1
        self.logger.info(f"===== FL AGGREGATION ROUND {self._agg_round} "
                         f"({len(self.weights_buffer)} vehicles) =====")

        # NaN guard on the candidate weights.
        for buffer in self.weights_buffer.values():
            candidate = buffer.get()
            if any(torch.isnan(p).any() for p in candidate.values()):
                self.logger.error(f"Candidate weights from {buffer.label} contain NaNs. Skipping.")
                return

        participants = [buf.get() for buf in self.weights_buffer.values()]
        if self.aggregation_function is federated_averaging:
            aggregated = self.aggregation_function(self.global_model.state_dict(), participants)
        else:
            aggregated = self.aggregation_function(self.global_model.state_dict(), participants,
                                                   **self.config)

        if any(torch.isnan(p).any() for p in aggregated.values()):
            self.logger.error("Aggregated state dict contains NaNs. Skipping update.")
            return

        for buffer in self.weights_buffer.values():
            buffer.pop()
        self.global_model.load_state_dict(aggregated)

        if self.packet_loss.should_drop():
            self.logger.debug(f"[packet-loss] dropped global weights update "
                              f"(rate={self.packet_loss.packet_loss_rate})")
            return

        # Publish the fresh global model (clone to avoid cross-node aliasing).
        payload = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
        self.network_delay.send(lambda: self.bus.produce("global_weights", payload))
        self.logger.info(
            f"Aggregated weights and sent global weights (round {self._agg_round}) to the nodes via '{self.config.get('aggregation_strategy')}'.")

    def status(self):
        return {
            'aggregation_rounds': self._agg_round,
            'vehicles': list(self.vehicle_weights_topics),
            'strategy': self.config.get('aggregation_strategy'),
            'packet_loss': self.packet_loss.stats(),
            'network_delay': self.network_delay.stats(),
        }
