"""Configuration loading — identical semantics to the dashboard.

The dashboard (``of-dashboard/app.py``) loads its configuration with Hydra:

    @hydra.main(config_path="config", config_name="default")
    def create_app(cfg):
        if cfg.override != "":
            config_overrides = OmegaConf.load(.../config/overrides/{cfg.override}.yaml)
            cfg = OmegaConf.merge(cfg, config_overrides)

The offline simulation reads the very same ``config/`` directory of the
SereBench repo and applies overrides the same way, so an experiment configured
for the Dockerised platform runs unchanged here. On top of that we accept
Hydra-style ``key=value`` dotlist overrides on the command line (e.g.
``federated_learning.aggregation_interval_secs=5``) for quick experimentation,
which is exactly what ``python app.py override=dev foo.bar=1`` would do.
"""

import os
from omegaconf import OmegaConf


def default_config_dir():
    """Return the SereBench ``config/`` directory (repo root / config).

    offline_simulation lives at ``<repo>/offline_simulation``; the single
    source of truth for every experiment parameter is ``<repo>/config``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    return os.path.join(repo_root, "config")


def load_config(config_dir=None, override=None, dotlist=None, config_name="default"):
    """Load ``config/<config_name>.yaml`` and merge overrides on top.

    Parameters
    ----------
    config_dir : str, optional
        Path to the ``config`` directory. Defaults to the SereBench repo's.
    override : str, optional
        Name of an override file under ``config/overrides/`` (without ``.yaml``),
        merged on top of the base config — identical to the dashboard's
        ``override=<name>`` mechanism. If ``None``, falls back to the base
        config's own ``override`` field (again mirroring the dashboard).
    dotlist : list[str], optional
        Hydra-style ``key=value`` strings merged last (highest precedence).
    """
    config_dir = config_dir or default_config_dir()
    base_path = os.path.join(config_dir, f"{config_name}.yaml")
    cfg = OmegaConf.load(base_path)

    # Mirror the dashboard: if no override was passed explicitly, honour the
    # one baked into the base config's `override:` field (default is "").
    if override is None:
        override = cfg.get("override", "") or ""

    if override:
        override_path = os.path.join(config_dir, "overrides", f"{override}.yaml")
        if not os.path.exists(override_path):
            raise FileNotFoundError(
                f"Config override '{override}' not found at {override_path}"
            )
        overrides = OmegaConf.load(override_path)
        cfg = OmegaConf.merge(cfg, overrides)
        # Record which override is active so downstream code (e.g. the model
        # save path built by the consumer manager) can use it, as it does on
        # the platform.
        cfg.override = override

    if dotlist:
        # Allow adding brand-new keys, as Hydra's command-line overrides do.
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(dotlist)))

    return cfg


def vehicle_names(cfg):
    """Extract the ordered list of vehicle names from ``cfg.vehicles``.

    Reproduces the exact parsing used everywhere in of-core (a vehicle entry is
    either a bare string or a single-key mapping ``{name: {overrides}}``).
    """
    names = []
    for vehicle in cfg.vehicles:
        if isinstance(vehicle, str):
            names.append(vehicle)
        else:
            names.append(list(vehicle.keys())[0])
    return names
