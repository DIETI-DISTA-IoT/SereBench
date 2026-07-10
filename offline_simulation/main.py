"""Single entry point for the monolithic, no-Docker / no-Kafka SereBench run.

Reads the very same ``config/`` directory the dashboard reads (Hydra-style
overrides included), spins up every SereBench service as a thread inside this
one process, runs the full experiment (producers + consumers + federated
learning + W&B, with automatic cyber-attacks) for a while, then tears it all
down cleanly.

Examples
--------
    # 4 vehicles, FedAvg, log to W&B offline, run for 3 minutes
    python -m offline_simulation.main --duration 180 --wandb-mode offline

    # use the fedyogi override file, faster FL rounds, no attacks
    python -m offline_simulation.main override=fedyogi \
        federated_learning.aggregation_interval_secs=5 --no-attacks

    # launch the local dashboard instead of a headless run (press buttons)
    python -m offline_simulation.main --dashboard
"""

import os
import sys
import time
import logging
import argparse

from .config_loader import load_config
from .orchestrator import Orchestrator


def _split_overrides(tokens):
    """Separate Hydra-style ``key=value`` tokens from an ``override=<name>`` selector."""
    override = None
    dotlist = []
    for tok in tokens:
        if '=' not in tok:
            raise SystemExit(f"Unrecognised argument: {tok!r} (expected key=value)")
        key, value = tok.split('=', 1)
        if key == 'override':
            override = value
        else:
            dotlist.append(tok)
    return override, dotlist


def _effective_wandb_mode(requested, cfg):
    """Resolve the W&B mode, defaulting to a login-free mode when appropriate.

    If the user did not force a mode and the config asks for online logging but
    no W&B credentials are present, fall back to 'offline' (still logs locally)
    so the simulation runs out-of-the-box without a W&B account.
    """
    if requested:
        return requested
    wants_online = bool(cfg.wandb.online)
    if wants_online and not (os.environ.get('WANDB_API_KEY') or os.environ.get('WANDB_MODE')):
        logging.getLogger("MAIN").warning(
            "wandb.online=true but no WANDB_API_KEY found — falling back to "
            "'offline' mode (metrics still logged locally to ./wandb). "
            "Pass --wandb-mode online once you've run `wandb login`.")
        return 'offline'
    return None  # let the node map online/disabled from cfg.wandb.online


def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="offline_simulation",
        description="Monolithic, single-process SereBench simulation (no Docker, no Kafka).")
    p.add_argument('--config-dir', default=None,
                   help="Path to the SereBench config/ dir (defaults to the repo's).")
    p.add_argument('--duration', type=float, default=120.0,
                   help="Headless run length in seconds (default 120). Ignored with --dashboard.")
    p.add_argument('--wandb-mode', default=None, choices=['online', 'offline', 'disabled'],
                   help="Force the W&B mode (default: derive from config.wandb.online).")
    p.add_argument('--no-attacks', action='store_true', help="Do not launch automatic cyber-attacks.")
    p.add_argument('--no-fl', action='store_true', help="Do not start federated learning.")
    p.add_argument('--no-wandb', action='store_true', help="Do not start the W&B logging node.")
    p.add_argument('--settle', type=float, default=1.0,
                   help="Scale for the staggered warmup delays before attacks "
                        "(1.0 = platform timing ~15s; smaller = quicker start).")
    p.add_argument('--seed', type=int, default=None,
                   help="Override default_vehicle_config.seed / default_consumer_config.seed.")
    p.add_argument('--dashboard', action='store_true',
                   help="Launch the local no-Docker dashboard instead of a headless run.")
    p.add_argument('--dashboard-host', default='127.0.0.1')
    p.add_argument('--dashboard-port', type=int, default=8000)
    # Hydra-style key=value overrides are accepted as free positional tokens
    # anywhere on the command line (collected via parse_known_args in main).
    p.epilog = ("Hydra-style overrides (key=value, and override=<name>) may appear "
                "anywhere, e.g.:  override=fedyogi anomaly_detection.model_type=cnn")
    return p


def main(argv=None):
    args, extras = build_arg_parser().parse_known_args(argv)
    # Any leftover token must be a Hydra-style key=value override; a stray flag
    # (starts with '-') is a genuine error rather than an override.
    bad = [e for e in extras if e.startswith('-')]
    if bad:
        raise SystemExit(f"Unrecognised option(s): {' '.join(bad)}")
    override, dotlist = _split_overrides(extras)
    if args.seed is not None:
        dotlist.append(f"default_vehicle_config.seed={args.seed}")
        dotlist.append(f"default_consumer_config.seed={args.seed}")

    cfg = load_config(config_dir=args.config_dir, override=override, dotlist=dotlist)

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=str(cfg.logging_level).upper())
    logger = logging.getLogger("MAIN")

    wandb_mode = _effective_wandb_mode(args.wandb_mode, cfg)

    if args.dashboard:
        # Imported lazily so a headless run does not require Flask.
        from .dashboard import run_dashboard
        logger.info(f"Launching offline dashboard on http://{args.dashboard_host}:{args.dashboard_port}")
        run_dashboard(cfg, wandb_mode=wandb_mode,
                      host=args.dashboard_host, port=args.dashboard_port)
        return

    orch = Orchestrator(cfg, wandb_mode=wandb_mode)
    orch.start_experiment(
        with_attacks=not args.no_attacks,
        with_fl=not args.no_fl,
        with_wandb=not args.no_wandb,
        settle=args.settle,
    )

    logger.info(f"Simulation running for {args.duration:.0f}s. Press Ctrl-C to stop early.")
    try:
        deadline = time.time() + args.duration
        while time.time() < deadline:
            time.sleep(2.0)
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down.")
    finally:
        _log_summary(orch, logger)
        orch.shutdown()
    logger.info("Done.")


def _log_summary(orch, logger):
    """Print a compact end-of-run summary (per-vehicle progress + FL rounds)."""
    s = orch.status()
    logger.info("================ run summary ================")
    for v, d in s['vehicles'].items():
        c = d.get('consumer') or {}
        p = d.get('producer') or {}
        p_loss = p.get('packet_loss') or {}
        c_loss = c.get('packet_loss') or {}
        logger.info(
            f"  {v:8s} status={d['status']:8s} epoch={c.get('epoch', 0):>4} "
            f"consumed={c.get('records_processed', 0):>7} "
            f"produced={p.get('records_produced', 0):>7} "
            f"(atk/anom/diag processed={c.get('attacks_processed', 0)}/"
            f"{c.get('anoms_processed', 0)}/{c.get('diagnostics_processed', 0)}) "
            f"packet_loss[producer]={p_loss.get('packets_dropped', 0)}/{p_loss.get('packets_sent', 0) + p_loss.get('packets_dropped', 0)} "
            f"packet_loss[consumer]={c_loss.get('packets_dropped', 0)}/{c_loss.get('packets_sent', 0) + c_loss.get('packets_dropped', 0)}")
    fl = s.get('federated_learning')
    if fl:
        fl_loss = fl.get('packet_loss') or {}
        logger.info(f"  federated_learning: rounds={fl['aggregation_rounds']} strategy={fl['strategy']} "
                    f"packet_loss={fl_loss.get('packets_dropped', 0)}/{fl_loss.get('packets_sent', 0) + fl_loss.get('packets_dropped', 0)}")
    else:
        logger.info("  federated_learning: not started")
    logger.info("=============================================")


if __name__ == "__main__":
    main(sys.argv[1:])
