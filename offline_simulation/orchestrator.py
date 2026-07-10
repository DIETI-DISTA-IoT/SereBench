"""Orchestrator — the in-process control plane (replaces the dashboard + Docker).

On the platform the dashboard's ``ContainerManager`` spawns one producer and one
consumer container per vehicle, a single FL-manager container and a single
Wandber container, and drives their lifecycle over HTTP while owning the
attack/vehicle-status bookkeeping. This class does all of that in one process:
each "container" is a node object (:class:`ProducerNode`, :class:`ConsumerNode`,
:class:`FLManagerNode`, :class:`WandbNode`) running its own threads, wired
together by a single :class:`MessageBus`.

The control-plane methods mirror the dashboard routes one-to-one
(``produce_all``, ``consume_all``, ``start_federated_learning``, ``start_wandb``,
``start_automatic_attacks``, ``start_experiment``, ``shutdown`` ...) and the
attack helpers reproduce ``start_attack_from_vehicle`` /
``stop_attack_from_vehicle`` (including the reactive mitigation-time
measurement) — but as direct method calls instead of Docker + HTTP, so pressing
a button starts a Python thread and nothing ever touches a container runtime.
"""

import time
import random
import logging
import threading

from .message_bus import MessageBus
from .config_loader import vehicle_names
from .config_builders import (build_producer_config, build_consumer_config,
                              build_fl_config, build_wandb_args)
from .producer import ProducerNode
from .consumer import ConsumerNode
from .fl_manager import FLManagerNode
from .wandb_logger import WandbNode

HEALTHY = "HEALTHY"
INFECTED = "INFECTED"

logger = logging.getLogger("ORCHESTRATOR")


class AttackAgent:
    """Port of of-core/attack_agent.py — periodically infects a random healthy vehicle."""

    def __init__(self, orchestrator, interval):
        self.orchestrator = orchestrator
        self.alive = False
        self.interval = interval
        self.thread = None

    def start(self):
        self.alive = True
        self.thread = threading.Thread(target=self._attacking_thread, daemon=True, name="attack_agent")
        self.thread.start()

    def _attacking_thread(self):
        while self.alive:
            healthy = [v for v, s in self.orchestrator.vehicle_status_dict.items() if s == HEALTHY]
            if healthy:
                self.orchestrator.start_attack_from_vehicle(random.choice(healthy), origin="AI")
            time.sleep(self.interval)

    def stop_all_attacks(self):
        for v, s in list(self.orchestrator.vehicle_status_dict.items()):
            if s == INFECTED:
                self.orchestrator.stop_attack_from_vehicle(v, origin="AI")


class Orchestrator:
    def __init__(self, cfg, wandb_mode=None):
        self.cfg = cfg
        self.wandb_mode = wandb_mode
        self.bus = MessageBus()
        self.vehicle_names = vehicle_names(cfg)

        self.vehicle_status_dict = {v: HEALTHY for v in self.vehicle_names}
        self.last_attack_started_at = {v: time.time() for v in self.vehicle_names}
        self._status_lock = threading.Lock()

        self.producers = {}   # vehicle_name -> ProducerNode
        self.consumers = {}   # vehicle_name -> ConsumerNode
        self.fl_manager = None
        self.wandber = None

        self.attack_agent = AttackAgent(self, cfg.attack.automatic_attack_interval_secs)

    # -- in-process control plane (replaces dashboard HTTP) ----------------
    def get_vehicle_status(self, vehicle_name):
        return self.vehicle_status_dict[vehicle_name]

    def start_attack_from_vehicle(self, vehicle_name, origin="MANUAL"):
        with self._status_lock:
            self.vehicle_status_dict[vehicle_name] = INFECTED
            self.last_attack_started_at[vehicle_name] = time.time()
        preamble = "Automatic" if origin == "AI" else "Manual"
        logger.info(f"[BOTMASTER] {preamble} attack at {vehicle_name} started.")
        return True

    def stop_attack_from_vehicle(self, vehicle_name, origin="MANUAL"):
        """Heal a vehicle and return the reactive mitigation time (seconds)."""
        with self._status_lock:
            if self.vehicle_status_dict.get(vehicle_name) != INFECTED:
                return None
            self.vehicle_status_dict[vehicle_name] = HEALTHY
            mitigation_time = time.time() - self.last_attack_started_at[vehicle_name]
        adverb = "automatically" if origin == "AI" else "manually"
        logger.info(f"[ATO] Attack at {vehicle_name} stopped {adverb} "
                    f"(mitigation_time={mitigation_time:.3f}s).")
        return mitigation_time

    def start_preconf_attack(self):
        for v in self.cfg.attack.preconf_attacking_vehicles:
            if v in self.vehicle_status_dict:
                self.start_attack_from_vehicle(v, origin="MANUAL")
        return "Preconfigured attack started!"

    def stop_preconf_attack(self):
        for v in self.cfg.attack.preconf_attacking_vehicles:
            if v in self.vehicle_status_dict:
                self.stop_attack_from_vehicle(v, origin="MANUAL")
        return "Preconfigured attack stopped!"

    def start_automatic_attacks(self):
        if not self.attack_agent.alive:
            self.attack_agent.start()
        logger.info("Automatic Attack Agent started!")
        return "Automatic Attack Agent started!"

    def stop_automatic_attacks(self):
        self.attack_agent.alive = False
        if self.attack_agent.thread is not None and self.attack_agent.thread.is_alive():
            self.attack_agent.thread.join(1)
        self.attack_agent.stop_all_attacks()
        logger.info("Automatic Attack Agent stopped!")
        return "Automatic Attack Agent stopped!"

    # -- vehicle lifecycle -------------------------------------------------
    def create_vehicles(self):
        for v in self.vehicle_names:
            if v not in self.producers:
                self.producers[v] = ProducerNode(
                    v, build_producer_config(self.cfg, v), self.bus, self.get_vehicle_status)
            if v not in self.consumers:
                self.consumers[v] = ConsumerNode(
                    v, build_consumer_config(self.cfg, v), self.bus,
                    self.get_vehicle_status, self._mitigate)
        logger.info(f"Vehicles created: {self.vehicle_names}")
        return "Vehicles created!"

    def _mitigate(self, vehicle_name):
        return self.stop_attack_from_vehicle(vehicle_name, origin="AI")

    def produce_all(self):
        self.create_vehicles()
        for p in self.producers.values():
            p.start()
        return "All producers started!"

    def stop_producing_all(self):
        for p in self.producers.values():
            p.stop()
        return "All producers stopped!"

    def consume_all(self):
        self.create_vehicles()
        for c in self.consumers.values():
            c.start()
        return "All consumers started!"

    def stop_consuming_all(self):
        for c in self.consumers.values():
            c.stop()
        return "All consumers stopped!"

    # -- federated learning ------------------------------------------------
    def start_federated_learning(self, params=None):
        if self.fl_manager is not None:
            return "Federated learning is already running"
        fl_config = build_fl_config(self.cfg)
        if params:
            fl_config.update(params)
        self.fl_manager = FLManagerNode(fl_config, self.bus)
        self.fl_manager.start()
        logger.info("Federated learning started!")
        return "Federated learning started!"

    def stop_federated_learning(self):
        if self.fl_manager is None:
            return "Federated learning is not running"
        self.fl_manager.stop()
        self.fl_manager = None
        logger.info("Federated learning stopped!")
        return "Federated learning stopped!"

    # -- wandb -------------------------------------------------------------
    def start_wandb(self):
        if self.wandber is not None:
            return "WandB is already running"
        self.wandber = WandbNode(build_wandb_args(self.cfg), self.bus, mode_override=self.wandb_mode)
        self.wandber.start()
        logger.info("WandB started!")
        return "WandB started!"

    def stop_wandb(self):
        if self.wandber is None:
            return "WandB is not running"
        self.wandber.stop()
        self.wandber = None
        logger.info("WandB stopped!")
        return "WandB stopped!"

    # -- full experiment (mirror dashboard /start-experiment) --------------
    def start_experiment(self, with_attacks=True, with_fl=True, with_wandb=True,
                         settle=1.0):
        """Bring the whole experiment up in the platform's order and cadence.

        The Dockerised /start-experiment staggers component startup with
        sleeps of 4s (after producers), 5s (after consumers), 3s (FL) and 3s
        (W&B) before enabling automatic attacks. That ~15s of *healthy* warmup
        is not cosmetic: it lets every vehicle's anomaly buffer fill past
        batch_size so local training can start before any vehicle is infected —
        otherwise a vehicle attacked too early never accumulates enough clean
        anomalies to train, and stays stuck. We reproduce those delays, scaled
        by ``settle`` (1.0 = platform timing; use a smaller value for quick,
        throwaway runs).
        """
        logger.info("Starting experiment")
        logger.info("Starting producers...")
        self.produce_all()
        time.sleep(4 * settle)
        logger.info("Starting consumers...")
        self.consume_all()
        time.sleep(5 * settle)
        if with_fl:
            logger.info("Starting federated learning...")
            self.start_federated_learning()
            time.sleep(3 * settle)
        if with_wandb:
            logger.info("Starting wandb...")
            self.start_wandb()
            time.sleep(3 * settle)
        if with_attacks:
            logger.info("Starting automatic attacks...")
            self.start_automatic_attacks()
        return "Experiment started"

    def shutdown(self):
        logger.info("Shutting down experiment...")
        self.stop_automatic_attacks()
        if self.fl_manager is not None:
            self.stop_federated_learning()
        self.stop_consuming_all()
        self.stop_producing_all()
        if self.wandber is not None:
            self.stop_wandb()
        logger.info("Shutdown complete.")
        return "Shutdown complete"

    # -- status ------------------------------------------------------------
    def status(self):
        return {
            'vehicles': {v: {
                'status': self.vehicle_status_dict[v],
                'producer': self.producers[v].status() if v in self.producers else None,
                'consumer': self.consumers[v].status() if v in self.consumers else None,
            } for v in self.vehicle_names},
            'federated_learning': self.fl_manager.status() if self.fl_manager else None,
            'wandb_running': self.wandber is not None,
            'automatic_attacks': self.attack_agent.alive,
        }
