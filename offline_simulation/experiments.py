#!/usr/bin/env python3
"""SereBench canonical experiment runner — OFFLINE (no Docker, no Kafka).

This is the in-process twin of ``tests/experiments.py`` (+ ``_cnn`` / ``_resnet``
/ ``_all``). Those runners drive the **Dockerised** platform: they shell out to
``tests/dash_cli.py``, which POSTs to the dashboard's HTTP control plane, which
in turn manages one producer/consumer container per vehicle plus the FL-manager
and Wandber containers over Kafka.

This module reproduces the *exact same experiment campaign* — same matrix, same
overrides, same seeds, same startup order and cadence — but against the
monolithic ``offline_simulation`` package instead of the container cluster.
Every "HTTP call to the dashboard" becomes a direct method call on an in-process
:class:`~offline_simulation.orchestrator.Orchestrator`; every container becomes
a thread. There is no Docker daemon, no Kafka broker and no network. See
``offline_simulation/README.md`` for how the two map onto each other.

Because the numerically-important cores are vendored verbatim from the
sub-repositories and the config is read from the same ``config/`` directory, the
results track the Dockerised runs — this is a laptop/CI-friendly way to
replicate the paper's campaign.

Experiment matrix (identical to tests/experiments.py)
-----------------------------------------------------
All experiments share an IDENTICAL, training-decoupled evaluation target:
HSJA on clean anchors (primary) plus a fixed Gaussian sigma-grid
(secondary). The eval no longer rides on each vehicle's Mp_std/Bp_std, so
cross-condition comparisons are unconfounded.

  Block A — ET1/ET2 (does adversarial training improve robustness?), no FL:
    1. et2-noadv-nofl       Baseline: default noise gradient, adversarial
                            training OFF, no FL (all models train clean).
    2. et2-adv-nofl         Adversarial training ON, default noise gradient
                            (bob/claude/daniel augment at 0.7/1.4/2.1), no FL.

  Block B — ET3 (does FL transfer robustness to a clean free-rider?):
    Common training-augmentation noise (1.0) for bob/claude/daniel; angela is
    the free-rider (adversarial_training=False, trains clean). The ONLY thing
    that differs across the block is FL on/off and the aggregation strategy.
    3. et3-freerider-nofl    Free-rider config, no FL (transfer baseline).
    4. et3-freerider-fedavg  Free-rider + FL (FedAvg).
    5. et3-freerider-fedprox   Same as 4 but FedProx (fedprox_mu=0.01).
    6. et3-freerider-fedyogi   Same as 4 but FedYogi.
    7. et3-freerider-fedmedian Same as 4 but FedMedian.

  Block C — ET4 (does FL help a NETWORK-impaired free-rider?), ids 8..39:
    Angela is the only vehicle with a degraded uplink (packet loss and/or added
    delay+jitter on her producer->consumer telemetry AND her consumer->FL
    weights upload); bob/claude/daniel keep a pristine network. Everyone trains
    clean and shares the ET3 decoupled eval. Generated as 8 impairment levels (3
    packet-loss rates, 3 delay/jitter settings, 2 combinations) x 4 FL modes
    (no-FL baseline + FedAvg + FedMedian + FedYogi) = 32 experiments:
        8..11  et4-loss20-*    12..15 et4-loss40-*    16..19 et4-loss60-*
        20..23 et4-delay100-*  24..27 et4-delay250-*  28..31 et4-delay500-*
        32..35 et4-combo-lo-*  36..39 et4-combo-hi-*

Run each across the three architectures (``--arch mlp|cnn|resnet|all``), which
merges the ``cnn`` / ``resnet`` config override on top (mlp is the default, no
extra override). W&B group / run names get an architecture suffix so results
land alongside, but distinct from, the other architectures — exactly like
``tests/experiments_cnn.py`` and ``tests/experiments_resnet.py``.

Startup order (per run) — mirrors tests/experiments.py exactly
--------------------------------------------------------------
  start-wandb               (open the W&B run first, so no metrics are missed)
  produce-all               (data must flow before consumers or FL start)
  sleep 5 s                 (let producers warm the anomaly buffers)
  start-automatic-attacks   (attack-class data present before classification)
  sleep 5 s                 (let the attack stream stabilise)
  consume-all               (consumers now find an active, mixed-class stream)
  sleep 5 s                 (let consumers initialise)
  start-federated-learning  (FL experiments only)
  --- experiment runs for --run-duration seconds ---
  shutdown                  (sequential teardown, same order as the dashboard)

Typical usage
-------------
  # All 39 experiments (7 canonical + 32 ET4, MLP), 3 seeds each, W&B offline
  python -m offline_simulation.experiments

  # Just the ET4 network-free-rider block (ids 8..39), every architecture
  python -m offline_simulation.experiments --arch all --experiments $(seq 8 39)

  # A quick smoke run: one experiment, one seed, 60 s
  python -m offline_simulation.experiments --experiments 1 --seeds 42 \
      --run-duration 60 --settle 0.2

  # Every architecture, FL experiments only, FedYogi/FedMedian/etc via the matrix
  python -m offline_simulation.experiments --arch all --experiments 4 5 6 7

  # Continue past failures instead of aborting the batch
  python -m offline_simulation.experiments --skip-on-error

Unlike the Dockerised runner there are no vehicle containers to create/delete —
every run builds a fresh in-process cluster from config and tears it down.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback

from omegaconf import OmegaConf

from .config_loader import default_config_dir
from .main import _effective_wandb_mode, _log_summary
from .orchestrator import Orchestrator

logger = logging.getLogger("OFFLINE-EXPERIMENTS")


# ---------------------------------------------------------------------------
# Defaults (kept in step with tests/experiments.py)
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_RUN_DURATION_SECS = 10 * 60      # 10 min per run
INTER_RUN_DELAY_SECS = 5                 # cooldown between successive runs

# Warmup delays within the startup sequence (seconds, scaled by --settle).
_DELAY_AFTER_PRODUCE = 5
_DELAY_AFTER_ATTACKS = 5
_DELAY_AFTER_CONSUME = 5


# ---------------------------------------------------------------------------
# Experiment matrix — identical to tests/experiments.py
# ---------------------------------------------------------------------------
# Each entry:
#   name     : W&B group name and run-name prefix
#   fl       : whether to start federated learning
#   override : config/overrides/<name>.yaml to apply (None = use defaults).
#              May also be a tuple of profile names, applied in sequence (each
#              deep-merged on top of the previous), exactly as the Dockerised
#              runner issues repeated `apply-override` calls.
EXPERIMENTS: dict[int, dict] = {
    # ---- Block A — ET1/ET2 (adversarial-training effect, no FL) ----
    1: {
        "name": "et2-noadv-nofl",
        "fl": False,
        "override": None,
        "description": "Baseline: default noise gradient, no adversarial training, no FL.",
    },
    2: {
        "name": "et2-adv-nofl",
        "fl": False,
        "override": "exp_advtraining",
        "description": "Adversarial training ON (default gradient augmentation), no FL.",
    },
    # ---- Block B — ET3 (FL robustness transfer, clean free-rider) ----
    3: {
        "name": "et3-freerider-nofl",
        "fl": False,
        "override": "exp_et3_freerider",
        "description": (
            "Free-rider transfer baseline: bob/claude/daniel train adversarially "
            "at common noise=1.0, angela trains clean. No FL."
        ),
    },
    4: {
        "name": "et3-freerider-fedavg",
        "fl": True,
        "override": "exp_et3_freerider",
        "description": "Free-rider + FL (FedAvg). Robustness transfer to angela via aggregation.",
    },
    5: {
        "name": "et3-freerider-fedprox",
        "fl": True,
        "override": ("exp_et3_freerider", "fedprox"),
        "description": "Same as et3-freerider-fedavg but FedProx (fedprox_mu=0.01).",
    },
    6: {
        "name": "et3-freerider-fedyogi",
        "fl": True,
        "override": ("exp_et3_freerider", "fedyogi"),
        "description": "Same as et3-freerider-fedavg but FedYogi (server-side adaptive LR).",
    },
    7: {
        "name": "et3-freerider-fedmedian",
        "fl": True,
        "override": ("exp_et3_freerider", "fedmedian"),
        "description": "Same as et3-freerider-fedavg but FedMedian.",
    },
}


# ---------------------------------------------------------------------------
# Block C — ET4 (does FL help a network-impaired free-rider?) — ids 8..39
# ---------------------------------------------------------------------------
# Identical block to the one appended in tests/experiments.py (kept in step by
# hand, exactly as the two EXPERIMENTS dicts already are). Angela is the only
# vehicle with a degraded uplink — packet loss and/or added delay+jitter on BOTH
# her producer->consumer telemetry pipeline and her consumer->FL weights upload
# (offline_simulation/packet_loss.py + network_delay.py apply these on the
# sending side, the exact counterpart of the Dockerised OpenFAIR modules);
# bob/claude/daniel keep a pristine network. Everyone trains clean and shares
# the ET3 decoupled eval, so the impairment on angela is the only moving part
# besides FL. Generated as 8 impairment levels (3 packet-loss rates, 3
# delay/jitter settings, 2 combinations) x 4 FL modes (no-FL baseline + FedAvg +
# FedMedian + FedYogi) = 32 experiments, runnable across --arch mlp|cnn|resnet|all.

# (short_tag, override_profile, human description of angela's impairment)
_ET4_LEVELS: list[tuple[str, str, str]] = [
    ("loss20",   "exp_et4_angela_loss20",   "packet loss 0.2"),
    ("loss40",   "exp_et4_angela_loss40",   "packet loss 0.4"),
    ("loss60",   "exp_et4_angela_loss60",   "packet loss 0.6"),
    ("delay100", "exp_et4_angela_delay100", "delay 100ms / jitter 25ms"),
    ("delay250", "exp_et4_angela_delay250", "delay 250ms / jitter 60ms"),
    ("delay500", "exp_et4_angela_delay500", "delay 500ms / jitter 125ms"),
    ("combo-lo", "exp_et4_angela_combo_lo", "packet loss 0.2 + delay 100ms/25ms"),
    ("combo-hi", "exp_et4_angela_combo_hi", "packet loss 0.4 + delay 250ms/60ms"),
]

# (mode_tag, fl_on, extra aggregation-strategy override applied on top, or None)
_ET4_MODES: list[tuple[str, bool, str | None]] = [
    ("nofl",      False, None),
    ("fedavg",    True,  "fedavg"),
    ("fedmedian", True,  "fedmedian"),
    ("fedyogi",   True,  "fedyogi"),
]


def _extend_with_et4(experiments: dict[int, dict]) -> None:
    """Append the 32-cell ET4 network-free-rider block to *experiments* in place."""
    next_id = max(experiments) + 1
    for level_tag, level_profile, level_desc in _ET4_LEVELS:
        for mode_tag, fl_on, extra in _ET4_MODES:
            override = level_profile if extra is None else (level_profile, extra)
            fl_desc = "no FL" if not fl_on else f"FL ({mode_tag[len('fed'):].upper()})"
            experiments[next_id] = {
                "name": f"et4-{level_tag}-{mode_tag}",
                "fl": fl_on,
                "override": override,
                "description": (
                    f"ET4 network free-rider: angela {level_desc}, peers pristine "
                    f"network. {fl_desc}."
                ),
            }
            next_id += 1


_extend_with_et4(EXPERIMENTS)


# ---------------------------------------------------------------------------
# Architecture variants — mirror experiments_cnn.py / experiments_resnet.py
# ---------------------------------------------------------------------------
# Each architecture is just an extra config override merged last (mlp is the
# default and needs none), plus a run/group name suffix so results stay
# distinct — exactly what the Dockerised per-architecture runners do.
ARCH_OVERRIDES: dict[str, str | None] = {
    "mlp": None,
    "cnn": "cnn",
    "resnet": "resnet",
}


# ---------------------------------------------------------------------------
# Config assembly — replicates the dashboard / dash_cli override chain
# ---------------------------------------------------------------------------

def build_config(exp: dict, arch: str, seed: int, run_name: str, group: str,
                 config_dir: str | None = None):
    """Assemble the OmegaConf config for one run.

    Reproduces the Dockerised chain (``init-config`` -> repeated
    ``apply-override`` -> ``set``): load ``config/default.yaml``, deep-merge each
    experiment override file in order, then the architecture override, then
    stamp the per-run seed / W&B run-name / group as dotlist overrides. Merge
    semantics are OmegaConf's (same engine Hydra uses in the dashboard), so
    lists — e.g. the per-vehicle ``vehicles`` block redefined by
    ``exp_et3_freerider`` — are replaced wholesale, matching ``deep_merge`` in
    ``tests/dash_cli.py``.
    """
    config_dir = config_dir or default_config_dir()
    import os

    cfg = OmegaConf.load(os.path.join(config_dir, "default.yaml"))

    # Collect the ordered list of override profiles: experiment override(s)
    # first, then the architecture override (if any).
    profiles: list[str] = []
    exp_override = exp["override"]
    if exp_override:
        if isinstance(exp_override, (list, tuple)):
            profiles.extend(exp_override)
        else:
            profiles.append(exp_override)
    arch_override = ARCH_OVERRIDES.get(arch)
    if arch_override:
        profiles.append(arch_override)

    for name in profiles:
        override_path = os.path.join(config_dir, "overrides", f"{name}.yaml")
        if not os.path.exists(override_path):
            raise FileNotFoundError(
                f"Config override '{name}' not found at {override_path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(override_path))

    # Record which override is active (parity with config_loader.load_config;
    # only cosmetic — used for the per-vehicle model save path label).
    if profiles:
        cfg.override = profiles[-1]

    # Per-run stamps, applied as Hydra-style dotlist overrides (highest
    # precedence), mirroring the `set` commands in tests/experiments.py:
    #   set default_consumer_config.seed <seed>
    #   set default_vehicle_config.seed  <seed>
    #   set wandb.run_name <run_name>
    #   set wandb.group    <group>
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([
        f"default_consumer_config.seed={seed}",
        f"default_vehicle_config.seed={seed}",
        f"wandb.run_name={run_name}",
        f"wandb.group={group}",
    ]))
    return cfg


# ---------------------------------------------------------------------------
# Sleep helper (progress line every 60 s), ported from tests/experiments.py
# ---------------------------------------------------------------------------

def _sleep(seconds: float, label: str) -> None:
    if seconds <= 0:
        return
    logger.info(f"[wait] {label} ({seconds:.0f}s) ...")
    end = time.monotonic() + seconds
    tick = min(60, seconds)
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(tick, remaining))
        remaining = end - time.monotonic()
        if remaining > 5:
            logger.info(f"[wait] {remaining:.0f}s remaining")


# ---------------------------------------------------------------------------
# Single-run execution — mirrors _run_one in tests/experiments.py
# ---------------------------------------------------------------------------

def run_one(exp: dict, arch: str, seed: int, run_idx: int, *,
            run_duration: float, wandb_mode: str | None,
            with_attacks: bool, with_wandb: bool, settle: float,
            config_dir: str | None = None) -> None:
    """Execute one experiment run (one seed value) fully in-process.

    Builds a fresh Orchestrator (the in-process dashboard + ContainerManager)
    from config, brings the cluster up in the platform's order and cadence, runs
    for ``run_duration`` seconds, then tears it down cleanly.
    """
    arch_suffix = "" if arch == "mlp" else f"-{arch}"
    name = f"{exp['name']}{arch_suffix}"
    run_name = f"{name}_seed{seed}_run{run_idx}"

    logger.info("=" * 60)
    logger.info(f"  Experiment  : {name}")
    logger.info(f"  Description : {exp['description']}")
    logger.info(f"  Arch        : {arch}")
    logger.info(f"  Run index   : {run_idx}  (seed={seed})")
    logger.info(f"  W&B run name: {run_name}")
    logger.info(f"  Duration    : {run_duration:.0f}s  (~{run_duration/3600:.2f} h)")
    logger.info(f"  FL          : {exp['fl']}")
    logger.info("=" * 60)

    cfg = build_config(exp, arch, seed, run_name, name, config_dir=config_dir)
    resolved_mode = _effective_wandb_mode(wandb_mode, cfg) if with_wandb else "disabled"

    orch = Orchestrator(cfg, wandb_mode=resolved_mode)
    try:
        # 1. W&B logger first — so no metrics are missed.
        if with_wandb:
            orch.start_wandb()

        # 2. Producers — data must flow before consumers or FL start.
        orch.produce_all()
        _sleep(_DELAY_AFTER_PRODUCE * settle, "letting producers warm up")

        # 3. Automatic attacks — attack-class data present before classifying.
        if with_attacks:
            orch.start_automatic_attacks()
            _sleep(_DELAY_AFTER_ATTACKS * settle, "letting attack stream stabilise")

        # 4. Consumers — now find an active, mixed-class data stream.
        orch.consume_all()
        _sleep(_DELAY_AFTER_CONSUME * settle, "letting consumers initialise")

        # 5. Federated learning (FL experiments only).
        if exp["fl"]:
            orch.start_federated_learning()

        # 6. Run.
        _sleep(run_duration, "experiment running")

        # 7. Summary + graceful shutdown (same teardown order as the dashboard).
        _log_summary(orch, logger)
    finally:
        orch.shutdown()

    logger.info(f"[ok] Run '{run_name}' complete.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="offline_simulation.experiments",
        description="SereBench multi-run experiment campaign, run fully in-process "
                    "(no Docker, no Kafka) — the offline twin of tests/experiments.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--experiments", nargs="+", type=int, choices=list(EXPERIMENTS),
                   default=list(EXPERIMENTS), metavar="N",
                   help="Experiment numbers to run (default: all; 1..7 canonical + 8..39 ET4).")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, metavar="S",
                   help=f"Random seeds — one independent run per seed (default: {DEFAULT_SEEDS}).")
    p.add_argument("--arch", nargs="+", choices=["mlp", "cnn", "resnet", "all"],
                   default=["mlp"],
                   help="Model architecture(s) to run (default: mlp). 'all' expands "
                        "to mlp cnn resnet.")
    p.add_argument("--run-duration", type=float, default=DEFAULT_RUN_DURATION_SECS,
                   metavar="SECS",
                   help=f"Duration of each run in seconds (default: {DEFAULT_RUN_DURATION_SECS}).")
    p.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"],
                   help="W&B mode for every run (default: offline — logs locally to "
                        "./wandb, no account needed). Pass 'online' after `wandb login`.")
    p.add_argument("--settle", type=float, default=1.0,
                   help="Scale for the staggered warmup delays before/between startup "
                        "steps (1.0 = platform timing ~15s; smaller = quicker start).")
    p.add_argument("--no-attacks", action="store_true",
                   help="Do not launch automatic cyber-attacks during the runs.")
    p.add_argument("--no-wandb", action="store_true",
                   help="Do not start the W&B logging node (metrics are not recorded).")
    p.add_argument("--skip-on-error", action="store_true",
                   help="Log failures and continue to the next run instead of aborting.")
    p.add_argument("--config-dir", default=None,
                   help="Path to the SereBench config/ dir (defaults to the repo's).")
    p.add_argument("--logging-level", default="info",
                   help="Root logging level for the runner (default: info).")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=str(args.logging_level).upper())

    # Expand the architecture selection ('all' -> mlp cnn resnet), de-duped and
    # kept in a stable order.
    archs: list[str] = []
    for a in args.arch:
        for expanded in (["mlp", "cnn", "resnet"] if a == "all" else [a]):
            if expanded not in archs:
                archs.append(expanded)

    wandb_mode = None if args.no_wandb else args.wandb_mode

    total = len(args.experiments) * len(archs) * len(args.seeds)
    completed = 0
    failed: list[str] = []

    logger.info("SereBench OFFLINE experiment runner (no Docker, no Kafka)")
    logger.info(f"  Experiments  : {args.experiments}")
    logger.info(f"  Architectures: {archs}")
    logger.info(f"  Seeds        : {args.seeds}")
    logger.info(f"  Total runs   : {total}")
    logger.info(f"  Run duration : {args.run_duration:.0f}s  (~{args.run_duration/3600:.2f} h)")
    logger.info(f"  W&B mode     : {'disabled (--no-wandb)' if args.no_wandb else args.wandb_mode}")
    logger.info(f"  Attacks      : {not args.no_attacks}")

    for arch in archs:
        for exp_id in args.experiments:
            exp = EXPERIMENTS[exp_id]
            for seed in args.seeds:
                run_idx = args.seeds.index(seed) + 1
                arch_suffix = "" if arch == "mlp" else f"-{arch}"
                run_label = f"{exp['name']}{arch_suffix}_seed{seed}"
                try:
                    run_one(
                        exp, arch, seed, run_idx,
                        run_duration=args.run_duration,
                        wandb_mode=wandb_mode,
                        with_attacks=not args.no_attacks,
                        with_wandb=not args.no_wandb,
                        settle=args.settle,
                        config_dir=args.config_dir,
                    )
                    completed += 1
                except KeyboardInterrupt:
                    logger.warning("Interrupted by user — aborting batch.")
                    return 1
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    logger.error(f"Run '{run_label}' FAILED:\n{tb}")
                    failed.append(run_label)
                    if not args.skip_on_error:
                        logger.error("Aborting batch. Re-run with --skip-on-error "
                                     "to continue past failures.")
                        return 1

                remaining = total - completed - len(failed)
                if remaining > 0:
                    _sleep(INTER_RUN_DELAY_SECS, "inter-run cooldown")

    logger.info("=" * 60)
    logger.info(f"  Completed : {completed} / {total}")
    if failed:
        logger.info(f"  Failed    : {len(failed)}")
        for name in failed:
            logger.info(f"    - {name}")
    logger.info("=" * 60)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
