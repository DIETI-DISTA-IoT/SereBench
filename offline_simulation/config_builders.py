"""Per-node configuration assembly — a faithful port of the of-core managers.

On the Dockerised platform the dashboard's ``ContainerManager`` /
``ProducerManager`` / ``ConsumerManager`` turn the single Hydra config into the
exact HTTP ``/configure`` payload each container receives. The offline
simulation must build byte-for-byte the same dictionaries so every node runs
with identical hyperparameters. Each function below mirrors one of those
assembly paths and cites the source it reproduces.
"""

from omegaconf import OmegaConf

from .config_loader import vehicle_names


def _container(x):
    """Resolve an OmegaConf node to a plain (interpolation-free) Python object."""
    return OmegaConf.to_container(x, resolve=True)


def _vehicle_override(cfg, vehicle_name):
    """Return the per-vehicle override mapping for ``vehicle_name`` (or {})."""
    for vehicle in cfg.vehicles:
        if isinstance(vehicle, str):
            if vehicle == vehicle_name:
                return {}
        elif list(vehicle.keys())[0] == vehicle_name:
            return _container(vehicle[vehicle_name])
    return {}


def build_producer_config(cfg, vehicle_name):
    """Reproduce ProducerManager.vehicle_configs + _build_config_data.

    vehicle_config = default_vehicle_config + per-vehicle overrides; then the
    manager stamps on identity/mode/attack/probe fields.
    """
    vehicle_config = _container(cfg.default_vehicle_config)
    vehicle_config.update(_vehicle_override(cfg, vehicle_name))

    attack = _container(cfg.attack)
    config_data = dict(vehicle_config)
    config_data.update({
        'vehicle_name': vehicle_name,
        'logging_level': cfg.logging_level,
        'manager_port': cfg.container_manager_port,
        'mode': cfg.mode,
        'target_ip': attack.get('target_ip', '172.18.0.4'),
        'target_port': attack.get('target_port', 80),
        'bot_port': attack.get('bot_port', 5002),
        'duration': attack.get('duration', 0),
        'packet_size': attack.get('packet_size', 1024),
        'delay': attack.get('delay', 0.001),
        'probe_metrics': list(map(str, _container(cfg.security_manager.probe_metrics))),
    })
    return config_data


def build_consumer_config(cfg, vehicle_name):
    """Reproduce ConsumerManager.consumer_configs + start_consumer cfg_payload.

    consumer_config = default_consumer_config + kafka_topic_update_interval_secs
    + per-vehicle overrides; then anomaly_detection is merged on top, then the
    manager stamps identity/mode/reward/mitigation fields.
    """
    consumer_config = _container(cfg.default_consumer_config)
    consumer_config['kafka_topic_update_interval_secs'] = cfg.kafka_topic_update_interval_secs
    consumer_config.update(_vehicle_override(cfg, vehicle_name))

    cfg_payload = dict(consumer_config)
    cfg_payload.update(_container(cfg.anomaly_detection))

    override = cfg.get('override', '') or ''
    cfg_payload.update({
        'logging_level': cfg.logging_level.upper(),
        'model_saving_path': f"{vehicle_name}_{override}_model.pth",
        'probe_metrics': list(map(str, _container(cfg.security_manager.probe_metrics))),
        'mode': str(cfg.mode),
        'manager_port': int(cfg.dashboard.port),
        # On the platform this is the dashboard container IP used for HTTP
        # mitigation calls; offline, mitigation is an in-process call so the
        # value is unused (kept for parity / logging).
        'manager_ip': 'offline',
        'true_positive_reward': float(cfg.security_manager.true_positive_reward),
        'false_positive_reward': float(cfg.security_manager.false_positive_reward),
        'true_negative_reward': float(cfg.security_manager.true_negative_reward),
        'false_negative_reward': float(cfg.security_manager.false_negative_reward),
    })

    if cfg.security_manager.mitigation:
        cfg_payload['mitigation'] = True
    # no_proxy_host is a Docker-networking concern; irrelevant offline, omitted.

    # The consumer's start_consumer_runtime forces 3-class output regardless of
    # config; make that explicit here too.
    cfg_payload['output_dim'] = 3
    return cfg_payload


def build_fl_config(cfg):
    """Reproduce ContainerManager.start_federated_learning payload.

    fl_config = federated_learning + anomaly_detection + wandb (merged in that
    order — note anomaly_detection.learning_rate therefore overrides the FL
    server learning_rate, exactly as on the platform).
    """
    fl_config = _container(cfg.federated_learning)
    fl_config.update(_container(cfg.anomaly_detection))
    fl_config.update(_container(cfg.wandb))
    return fl_config


def build_wandb_args(cfg):
    """Reproduce the dashboard's start_wandb payload: the whole resolved config.

    ``Wandber.__init__`` indexes ``args['wandb']`` and ``args['logging_level']``,
    so it is handed the entire config dict.
    """
    return _container(cfg)


def all_vehicle_names(cfg):
    return vehicle_names(cfg)
