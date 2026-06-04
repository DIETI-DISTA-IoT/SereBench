#!/usr/bin/env python3
"""SeReBench canonical experiment runner.

Reproduces the 8 experiments described in the SereBench paper with multiple
independent runs per experiment (one per seed) for statistical significance.

Experiment matrix
-----------------
  1. noadvtraining-nofl    Baseline: default noise gradient (angela=0, bob=0.7,
                           claude=1.4, daniel=2.1), no adversarial training,
                           no federated learning.
  2. advtraining-nofl      Adversarial training ON for all vehicles (each trains
                           on its own noise level), no FL.
  3. noadvtraining-fl      Federated learning only, no adversarial training,
                           default noise gradient.
  4. advtraining-fl        FL + adversarial training; uniform noise=1.0 across
                           all vehicles; only bob/claude/daniel train adversarially,
                           angela trains clean.
  5. dynamic-noise-fl      FL; angela starts at noise=0, then noise=1.0 is
                           injected at ~50% of run duration.
  6. dynamic-noise-nofl    Same as 5 but without FL.
  7. noadvtraining-fl-seed Experiment 3 repeated with a fixed shared seed for all
                           vehicles (faster FL convergence baseline).
  8. advtraining-fl-seed   Experiment 4 repeated with a fixed shared seed.

Usage
-----
  # All 8 experiments, 5 seeds each (default ~40 h at 1 h/run)
  python tests/experiments.py

  # Specific experiments only
  python tests/experiments.py --experiments 1 2 3

  # Fewer seeds for a quick sanity check
  python tests/experiments.py --seeds 42 123 --run-duration 600

  # Custom dashboard URL
  python tests/experiments.py --base-url http://192.168.1.10:5000

Dynamic-noise experiments (5 & 6)
----------------------------------
At the midpoint of each run the script attempts to update angela's noise level
to 1.0 by POSTing to the dashboard endpoint POST /update-vehicle-noise.  If that
endpoint is not yet implemented the call fails gracefully (warning printed, run
continues with original noise level) and you can perform the injection manually.
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
INIT_DELAY_SECS = 15                  # wait after create-vehicles before producing
INTER_RUN_DELAY_SECS = 30             # wait between successive runs for containers to settle


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Each entry:
#   name          : W&B group name and run-name prefix
#   fl            : whether to start federated learning
#   override      : config/overrides/<name>.yaml to apply (None = use defaults)
#   dynamic_noise : whether to inject mid-run noise into angela's producer
#   fixed_seed    : if True, all runs use seed=777 (same-init FL baseline)
EXPERIMENTS: dict[int, dict] = {
    1: {
        "name": "noadvtraining-nofl",
        "fl": False,
        "override": None,
        "dynamic_noise": False,
        "fixed_seed": False,
        "description": (
            "Baseline: default noise gradient, no adversarial training, no FL."
        ),
    },
    2: {
        "name": "advtraining-nofl",
        "fl": False,
        "override": "exp_advtraining",
        "dynamic_noise": False,
        "fixed_seed": False,
        "description": (
            "Adversarial training ON for all vehicles (each trains on own noise), no FL."
        ),
    },
    3: {
        "name": "noadvtraining-fl",
        "fl": True,
        "override": None,
        "dynamic_noise": False,
        "fixed_seed": False,
        "description": "Federated learning only, no adversarial training.",
    },
    4: {
        "name": "advtraining-fl",
        "fl": True,
        "override": "exp_advtraining_fl",
        "dynamic_noise": False,
        "fixed_seed": False,
        "description": (
            "FL + adversarial training; uniform noise=1.0; only bob/claude/daniel "
            "train adversarially, angela trains clean."
        ),
    },
    5: {
        "name": "dynamic-noise-fl",
        "fl": True,
        "override": "exp_dynamic_noise_phase1",
        "dynamic_noise": True,
        "fixed_seed": False,
        "description": (
            "FL + mid-run noise injection on angela (noise 0 → 1.0 at 50% duration)."
        ),
    },
    6: {
        "name": "dynamic-noise-nofl",
        "fl": False,
        "override": "exp_dynamic_noise_phase1",
        "dynamic_noise": True,
        "fixed_seed": False,
        "description": "Mid-run noise injection on angela, no FL.",
    },
    7: {
        "name": "noadvtraining-fl-seed",
        "fl": True,
        "override": None,
        "dynamic_noise": False,
        "fixed_seed": True,
        "description": (
            "Experiment 3 with same-seed model initialisation across all vehicles "
            "(demonstrates faster FL convergence when models share their starting point)."
        ),
    },
    8: {
        "name": "advtraining-fl-seed",
        "fl": True,
        "override": "exp_advtraining_fl",
        "dynamic_noise": False,
        "fixed_seed": True,
        "description": (
            "Experiment 4 with same-seed model initialisation across all vehicles."
        ),
    },
}

# Seed used for experiments 7 & 8 (same-init FL baseline)
FIXED_SEED = 777


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cli(*args: str, base_url: str, timeout: int = 180) -> None:
    """Run a dash_cli.py command, raise RuntimeError on non-zero exit."""
    cmd = [
        sys.executable, str(DASH_CLI),
        "--base-url", base_url,
        "--timeout", str(timeout),
    ] + list(args)
    label = " ".join(args)
    print(f"  [cli] {label}", flush=True)
    result = subprocess.run(cmd, text=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"dash_cli command failed: {label}")


def _sleep(seconds: int, label: str) -> None:
    """Sleep with a simple countdown printed every 60 s."""
    print(f"  [wait] {label} ({seconds}s)", flush=True)
    end = time.monotonic() + seconds
    interval = min(60, seconds)
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
        remaining = end - time.monotonic()
        if remaining > 0:
            print(f"  [wait] {remaining:.0f}s remaining ...", flush=True)
    print(f"  [wait] done.", flush=True)


def _inject_angela_noise(base_url: str, mp_std: float = 1.0, bp_std: float = 1.0) -> None:
    """Attempt to update angela's noise level at runtime via the dashboard API.

    The dashboard must expose POST /update-vehicle-noise with JSON payload::

        {"vehicle_name": "angela", "Mp_std": <float>, "Bp_std": <float>}

    If the endpoint is absent (404) or the request fails, a warning is printed
    and the run continues unchanged.
    """
    url = f"{base_url.rstrip('/')}/update-vehicle-noise"
    payload = {"vehicle_name": "angela", "Mp_std": mp_std, "Bp_std": bp_std}
    print(
        f"  [noise] Attempting runtime noise injection for angela "
        f"(Mp_std={mp_std}, Bp_std={bp_std}) via {url} ...",
        flush=True,
    )
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            print("  [noise] Injection succeeded.", flush=True)
        else:
            print(
                f"  [noise] WARNING: endpoint returned HTTP {resp.status_code}. "
                "Dynamic noise injection was NOT applied. "
                "Implement POST /update-vehicle-noise in the dashboard to enable it.",
                flush=True,
            )
    except requests.RequestException as exc:
        print(
            f"  [noise] WARNING: noise injection request failed ({exc}). "
            "Dynamic noise injection was NOT applied. "
            "Implement POST /update-vehicle-noise in the dashboard to enable it.",
            flush=True,
        )


def _run_one(
    exp: dict,
    seed: int,
    run_idx: int,
    run_duration: int,
    base_url: str,
) -> None:
    """Execute a single experiment run (one seed)."""
    name = exp["name"]
    run_name = f"{name}_seed{seed}_run{run_idx}"
    print(f"\n{'='*60}", flush=True)
    print(f"  Experiment : {name}", flush=True)
    print(f"  Run        : {run_idx}  (seed={seed})", flush=True)
    print(f"  W&B name   : {run_name}", flush=True)
    print(f"  Duration   : {run_duration}s", flush=True)
    print(f"  FL         : {exp['fl']}", flush=True)
    print(f"  Adv noise  : {exp['dynamic_noise']}", flush=True)
    print(f"{'='*60}", flush=True)

    def cli(*args: str) -> None:
        _cli(*args, base_url=base_url)

    # 1. Initialise CLI state from default config
    cli("init-config")

    # 2. Apply experiment-specific override (vehicle noise & adversarial_training)
    if exp["override"]:
        cli("apply-override", exp["override"])

    # 3. Per-run parameter overrides
    effective_seed = FIXED_SEED if exp["fixed_seed"] else seed
    cli("set", "default_consumer_config.seed", str(effective_seed))
    cli("set", "wandb.run_name", run_name)
    # Store group name so W&B runs can be filtered by experiment
    cli("set", "wandb.group", name)

    # 4. Create vehicle containers
    cli("create-vehicles")
    _sleep(INIT_DELAY_SECS, "waiting for containers to initialise")

    # 5. Start producers, consumers, W&B logger
    cli("produce-all")
    cli("consume-all")
    cli("start-wandb")

    # 6. Optionally start federated learning
    if exp["fl"]:
        cli("start-federated-learning")

    # 7. Run — with optional mid-run noise injection for experiments 5 & 6
    if exp["dynamic_noise"]:
        half = run_duration // 2
        _sleep(half, "phase 1 (angela noise=0)")
        _inject_angela_noise(base_url, mp_std=1.0, bp_std=1.0)
        _sleep(run_duration - half, "phase 2 (angela noise=1.0)")
    else:
        _sleep(run_duration, "experiment running")

    # 8. Graceful shutdown
    cli("shutdown")
    _sleep(INIT_DELAY_SECS, "waiting for shutdown to complete")

    # 9. Remove vehicle containers
    cli("delete-vehicles")

    print(f"  [ok] Run {run_name} completed.\n", flush=True)


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
        help="Experiment numbers to run (default: all 8).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        metavar="S",
        help=f"Random seeds / run indices (default: {DEFAULT_SEEDS}).",
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
            "If a run fails, log the error and continue to the next run "
            "instead of aborting the entire batch."
        ),
    )
    args = parser.parse_args()

    total = len(args.experiments) * len(args.seeds)
    completed = 0
    failed: list[str] = []

    print(f"SereBench experiment runner")
    print(f"  Experiments : {args.experiments}")
    print(f"  Seeds       : {args.seeds}")
    print(f"  Total runs  : {total}")
    print(f"  Run duration: {args.run_duration}s  (~{args.run_duration/3600:.1f} h)")
    print(f"  Dashboard   : {args.base_url}")
    print()

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
            msg = f"Run {run_label} FAILED:\n{tb}"
            print(f"\n[ERROR] {msg}", flush=True)
            failed.append(run_label)
            if not args.skip_on_error:
                print(
                    "Aborting batch. Re-run with --skip-on-error to continue past failures.",
                    flush=True,
                )
                return 1
            # Best-effort cleanup before next run
            try:
                import subprocess as _sp
                _sp.run(
                    [sys.executable, str(DASH_CLI), "--base-url", args.base_url, "shutdown"],
                    timeout=60,
                )
                _sp.run(
                    [sys.executable, str(DASH_CLI), "--base-url", args.base_url, "delete-vehicles"],
                    timeout=60,
                )
            except Exception:  # noqa: BLE001
                pass

        if completed + len(failed) < total:
            _sleep(INTER_RUN_DELAY_SECS, "inter-run cooldown")

    print(f"\n{'='*60}")
    print(f"  Completed : {completed}/{total}")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*60}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
