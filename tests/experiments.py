#!/usr/bin/env python3
"""SeReBench canonical experiment runner.

Reproduces the 6 experiments with multiple independent runs per experiment
(one per seed) for statistical significance.

Experiment matrix
-----------------
  1. noadvtraining-nofl    Baseline: default noise gradient (angela=0, bob=0.7,
                           claude=1.4, daniel=2.1), no adversarial training,
                           no federated learning.
  2. advtraining-nofl      Adversarial training ON for all vehicles (each trains
                           on its own noise level), no FL.
  3. dynamic-noise-fl      FL; angela starts at noise=0, then noise=1.0 is
                           injected at ~50% of run duration via POST /reset-noise.
  4. dynamic-noise-nofl    Same as 3 but without FL.
  5. noadvtraining-fl      Federated learning only, no adversarial training,
                           default noise gradient.
  6. advtraining-fl        FL + adversarial training; uniform noise=1.0 across
                           all vehicles; only bob/claude/daniel train adversarially,
                           angela trains clean.

Pre-requisites
--------------
Before running this script, vehicle containers must already exist:

    python tests/dash_cli.py create-vehicles

This only needs to be done ONCE.  The runner does NOT create or delete vehicles
between runs — it only starts and stops the services running inside them.

Typical usage
-------------
  # All 6 experiments, 5 seeds each (~30 h at 1 h/run)
  python tests/experiments.py

  # Specific experiments only
  python tests/experiments.py --experiments 1 2

  # Fewer seeds for a quick sanity check
  python tests/experiments.py --seeds 42 123 --run-duration 600

  # Continue past failures instead of aborting
  python tests/experiments.py --skip-on-error

  # Custom dashboard URL
  python tests/experiments.py --base-url http://192.168.1.10:5000

Startup order (per run)
-----------------------
  start-wandb               (synchronous; timeout 120 s — W&B can be slow)
  produce-all               (synchronous)
  sleep 15 s
  start-automatic-attacks   (synchronous)
  sleep 15 s
  consume-all               (synchronous)
  sleep 10 s
  start-federated-learning  (synchronous; FL experiments only)
  --- experiment runs for --run-duration seconds ---
  shutdown                  (synchronous, sequential teardown in the dashboard)

Vehicle containers are NOT touched between runs.

Dynamic-noise experiments (3 & 4)
----------------------------------
At the midpoint of each run the script POSTs to POST /reset-noise on the
dashboard to switch angela from noise=0 to Mp_std=Bp_std=1.0.  This endpoint
already exists in the dashboard (app.py).  If the request fails a warning is
printed and the run continues at the original noise level.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DASH_CLI = REPO_ROOT / "tests" / "dash_cli.py"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [42, 123, 456, 789, 1234]
DEFAULT_RUN_DURATION_SECS = 3600     # 1 hour per run
INTER_RUN_DELAY_SECS = 30             # cooldown between successive runs

# Delays within the startup sequence (seconds)
_DELAY_AFTER_PRODUCE  = 15
_DELAY_AFTER_ATTACKS  = 15
_DELAY_AFTER_CONSUME  = 10

# Timeout for start-wandb specifically — W&B can be slow to initialise
_WANDB_TIMEOUT_SECS = 120


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Each entry:
#   name          : W&B group name and run-name prefix
#   fl            : whether to start federated learning
#   override      : config/overrides/<name>.yaml to apply (None = use defaults)
#   dynamic_noise : whether to inject mid-run noise into angela via /reset-noise
EXPERIMENTS: dict[int, dict] = {
    1: {
        "name": "noadvtraining-nofl",
        "fl": False,
        "override": None,
        "dynamic_noise": False,
        "description": "Baseline: default noise gradient, no adversarial training, no FL.",
    },
    2: {
        "name": "advtraining-nofl",
        "fl": False,
        "override": "exp_advtraining",
        "dynamic_noise": False,
        "description": "Adversarial training ON for all vehicles (each trains on own noise), no FL.",
    },
    3: {
        "name": "dynamic-noise-fl",
        "fl": True,
        "override": "exp_dynamic_noise_phase1",
        "dynamic_noise": True,
        "description": "FL + mid-run noise injection on angela (noise 0 -> 1.0 at 50% duration).",
    },
    4: {
        "name": "dynamic-noise-nofl",
        "fl": False,
        "override": "exp_dynamic_noise_phase1",
        "dynamic_noise": True,
        "description": "Mid-run noise injection on angela, no FL.",
    },
    5: {
        "name": "noadvtraining-fl",
        "fl": True,
        "override": None,
        "dynamic_noise": False,
        "description": "Federated learning only, no adversarial training, default noise gradient.",
    },
    6: {
        "name": "advtraining-fl",
        "fl": True,
        "override": "exp_advtraining_fl",
        "dynamic_noise": False,
        "description": (
            "FL + adversarial training; uniform noise=1.0; only bob/claude/daniel "
            "train adversarially, angela trains clean."
        ),
    },
}


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _cli(*args: str, base_url: str, timeout: int = 180) -> None:
    """Run one dash_cli.py command and block until it completes.

    Execution is fully synchronous: this function blocks until the underlying
    HTTP request to the dashboard returns (or the timeout is hit).  The
    dashboard routes are synchronous — they only return 200 once the
    requested action has been dispatched.  We therefore never issue the next
    step before the current one has completed.

    Raises RuntimeError if the command exits with a non-zero status code.
    """
    cmd = [
        sys.executable, str(DASH_CLI),
        "--base-url", base_url,
        "--timeout", str(timeout),
    ] + list(args)
    label = " ".join(args)
    print(f"  [cli] {label}", flush=True)
    result = subprocess.run(cmd, text=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"dash_cli command failed (exit {result.returncode}): {label}")


# ---------------------------------------------------------------------------
# Sleep helper
# ---------------------------------------------------------------------------

def _sleep(seconds: int, label: str) -> None:
    """Sleep for *seconds* with a progress line printed every 60 s."""
    print(f"  [wait] {label} ({seconds}s) ...", flush=True)
    end = time.monotonic() + seconds
    tick = min(60, seconds)
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(tick, remaining))
        remaining = end - time.monotonic()
        if remaining > 5:
            print(f"  [wait] {remaining:.0f}s remaining", flush=True)
    print(f"  [wait] done.", flush=True)


# ---------------------------------------------------------------------------
# Dynamic noise injection
# ---------------------------------------------------------------------------

def _inject_angela_noise(base_url: str, mp_std: float = 1.0, bp_std: float = 1.0) -> None:
    """Switch angela's adversarial noise level at runtime via POST /reset-noise.

    Payload expected by the dashboard::

        {"vehicle_name": "angela", "Mp_std": <float>, "Bp_std": <float>}

    If the call fails a warning is printed and the run continues at the
    original noise level.
    """
    url = f"{base_url.rstrip('/')}/reset-noise"
    payload = {"vehicle_name": "angela", "Mp_std": mp_std, "Bp_std": bp_std}
    print(
        f"  [noise] Injecting noise into angela "
        f"(Mp_std={mp_std}, Bp_std={bp_std}) via {url} ...",
        flush=True,
    )
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.ok:
            print("  [noise] Injection succeeded.", flush=True)
        else:
            print(
                f"  [noise] WARNING: /reset-noise returned HTTP {resp.status_code} — "
                f"{resp.text[:200]}. "
                "Dynamic noise injection was NOT applied for this run.",
                flush=True,
            )
    except requests.RequestException as exc:
        print(
            f"  [noise] WARNING: /reset-noise request failed ({exc}). "
            "Dynamic noise injection was NOT applied for this run.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

def _run_one(
    exp: dict,
    seed: int,
    run_idx: int,
    run_duration: int,
    base_url: str,
) -> None:
    """Execute one experiment run (one seed value).

    Vehicle containers are assumed to already exist.  This function only
    starts and stops the services running inside them.
    """
    name = exp["name"]
    run_name = f"{name}_seed{seed}_run{run_idx}"

    print(f"\n{'='*60}", flush=True)
    print(f"  Experiment  : {name}", flush=True)
    print(f"  Description : {exp['description']}", flush=True)
    print(f"  Run index   : {run_idx}  (seed={seed})", flush=True)
    print(f"  W&B run name: {run_name}", flush=True)
    print(f"  Duration    : {run_duration}s  (~{run_duration/3600:.2f} h)", flush=True)
    print(f"  FL          : {exp['fl']}", flush=True)
    print(f"  Dynamic noise: {exp['dynamic_noise']}", flush=True)
    print(f"{'='*60}", flush=True)

    def cli(*args: str, timeout: int = 180) -> None:
        _cli(*args, base_url=base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # 1. Configure CLI state
    # ------------------------------------------------------------------
    cli("init-config")

    if exp["override"]:
        cli("apply-override", exp["override"])

    # Seed both the MLP initialisation (consumer) and the data-generation
    # simulator (producer) with the same per-run value so experiments are
    # reproducible end-to-end.
    cli("set", "default_consumer_config.seed", str(seed))
    cli("set", "default_vehicle_config.seed", str(seed))
    cli("set", "wandb.run_name", run_name)
    cli("set", "wandb.group", name)

    # ------------------------------------------------------------------
    # 2. Start W&B logger first — so no metrics are missed.
    #    W&B can take a while to initialise; use a dedicated generous timeout.
    # ------------------------------------------------------------------
    cli("start-wandb", timeout=_WANDB_TIMEOUT_SECS)

    # ------------------------------------------------------------------
    # 3. Start producers — data must flow before consumers or FL start.
    # ------------------------------------------------------------------
    cli("produce-all")
    _sleep(_DELAY_AFTER_PRODUCE, "letting producers warm up")

    # ------------------------------------------------------------------
    # 4. Start automatic attacks — so attack-class data is present in the
    #    stream before consumers begin classifying.
    # ------------------------------------------------------------------
    cli("start-automatic-attacks")
    _sleep(_DELAY_AFTER_ATTACKS, "letting attack stream stabilise")

    # ------------------------------------------------------------------
    # 5. Start consumers — they now find an active, mixed-class data stream.
    # ------------------------------------------------------------------
    cli("consume-all")
    _sleep(_DELAY_AFTER_CONSUME, "letting consumers initialise")

    # ------------------------------------------------------------------
    # 6. Start federated learning (FL experiments only).
    # ------------------------------------------------------------------
    if exp["fl"]:
        cli("start-federated-learning")

    # ------------------------------------------------------------------
    # 7. Run — with optional mid-run noise injection (experiments 3 & 4).
    # ------------------------------------------------------------------
    if exp["dynamic_noise"]:
        half = run_duration // 2
        _sleep(half, "phase 1 — angela noise=0")
        _inject_angela_noise(base_url, mp_std=1.0, bp_std=1.0)
        _sleep(run_duration - half, "phase 2 — angela noise=1.0")
    else:
        _sleep(run_duration, "experiment running")

    # ------------------------------------------------------------------
    # 8. Graceful shutdown.
    #    POST /shutdown in app.py performs a sequential teardown:
    #      stop_security_manager -> stop_fl -> stop_consumers ->
    #      stop_producers -> stop_attacks -> stop_wandb
    #    It is synchronous and returns only when all steps are done.
    # ------------------------------------------------------------------
    cli("shutdown")
    # Vehicle containers are intentionally NOT deleted here.
    # They are reused across runs; delete them manually when done.

    print(f"  [ok] Run '{run_name}' complete.\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SereBench multi-run experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=int,
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
        metavar="N",
        help="Experiment numbers to run (default: all 6).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        metavar="S",
        help=f"Random seeds — one independent run per seed (default: {DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--run-duration",
        type=int,
        default=DEFAULT_RUN_DURATION_SECS,
        metavar="SECS",
        help=f"Duration of each run in seconds (default: {DEFAULT_RUN_DURATION_SECS}).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:5000",
        help="Dashboard HTTP base URL (default: http://localhost:5000).",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help=(
            "Log failures and continue to the next run instead of aborting "
            "the whole batch."
        ),
    )
    args = parser.parse_args()

    total = len(args.experiments) * len(args.seeds)
    completed = 0
    failed: list[str] = []

    print("SereBench experiment runner")
    print(f"  Experiments  : {args.experiments}")
    print(f"  Seeds        : {args.seeds}")
    print(f"  Total runs   : {total}")
    print(f"  Run duration : {args.run_duration}s  (~{args.run_duration/3600:.1f} h)")
    print(f"  Dashboard    : {args.base_url}")
    print(
        "\nNOTE: vehicle containers must already exist before running this script.\n"
        "      If they don't, run:  python tests/dash_cli.py create-vehicles\n"
    )

    for exp_id, seed in ((e, s) for e in args.experiments for s in args.seeds):
        exp = EXPERIMENTS[exp_id]
        run_idx = args.seeds.index(seed) + 1
        run_label = f"{exp['name']}_seed{seed}"
        try:
            _run_one(
                exp=exp,
                seed=seed,
                run_idx=run_idx,
                run_duration=args.run_duration,
                base_url=args.base_url,
            )
            completed += 1
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"\n[ERROR] Run '{run_label}' FAILED:\n{tb}", flush=True)
            failed.append(run_label)
            if not args.skip_on_error:
                print(
                    "Aborting batch.  Re-run with --skip-on-error to continue past failures.",
                    flush=True,
                )
                return 1
            # Best-effort cleanup so the next run starts from a clean state.
            # We do NOT delete vehicles.
            print("  [cleanup] Attempting best-effort shutdown before next run ...", flush=True)
            try:
                subprocess.run(
                    [
                        sys.executable, str(DASH_CLI),
                        "--base-url", args.base_url,
                        "--timeout", "60",
                        "shutdown",
                    ],
                    timeout=90,
                )
            except Exception:  # noqa: BLE001
                print("  [cleanup] WARNING: cleanup shutdown also failed.", flush=True)

        remaining_runs = total - completed - len(failed)
        if remaining_runs > 0:
            _sleep(INTER_RUN_DELAY_SECS, "inter-run cooldown")

    print(f"\n{'='*60}")
    print(f"  Completed : {completed} / {total}")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for name in failed:
            print(f"    - {name}")
    print(f"{'='*60}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
