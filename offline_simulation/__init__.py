"""Monolithic, single-process SereBench simulation.

Runs the entire SereBench federated-learning benchmark — data producing,
per-vehicle anomaly-detection training, adversarial robustness evaluation
(Gaussian sigma-grid + HopSkipJump), federated aggregation and W&B logging —
inside one Python process, with no Docker containers and no Kafka broker. An
in-process :class:`~offline_simulation.message_bus.MessageBus` replaces Kafka
and direct method calls replace the dashboard's HTTP/Docker control plane.

The algorithm cores (``simulator``, ``models``, ``hopskipjump``,
``aggregation``) are vendored verbatim from the SereBench sub-repositories, so
the numerical behaviour matches the Dockerised platform; only the transport and
orchestration are reimplemented.
"""

from .config_loader import load_config
from .orchestrator import Orchestrator

__all__ = ["load_config", "Orchestrator"]
