#!/usr/bin/env python3
"""SeReBench canonical experiment runner.

Reproduces the redesigned campaign (Block A: ET1/ET2; Block B: ET3) with
multiple independent runs per experiment (one per seed) for statistical
significance.

Experiment matrix (redesigned campaign)
---------------------------------------
All experiments share an IDENTICAL, training-decoupled evaluation target:
HSJA on clean anchors (primary) plus a fixed Gaussian sigma-grid
(sigma_grid_evaluation, secondary). The eval no longer rides on each vehicle's
Mp_std/Bp_std, so cross-condition comparisons are unconfounded.

  Block A — ET1/ET2 (does adversarial training improve robustness?), no FL:
    1. et2-noadv-nofl       Baseline: default noise gradient, adversarial
                            training OFF, no FL (all models train clean).
    2. et2-adv-nofl         Adversarial training ON, default noise gradient
                            (bob/claude/daniel augment at 0.7/1.4/2.1), no FL.

  Block B — ET3 (does FL transfer robustness to a clean free-rider?):
    Common training-augmentation noise (1.0) for bob/claude/daniel; angela is
    the free-rider (adversarial_training=False, trains clean). The ONLY thing
    that differs across the block is FL on/off and the aggregation strategy.
    3. et3-freerider-nofl   Free-rider config, no FL (transfer baseline).
    4. et3-freerider-fedavg Free-rider + FL (FedAvg).
    5. et3-freerider-fedprox  Same as 4 but FedProx (fedprox_mu=0.01).
    6. et3-freerider-fedyogi  Same as 4 but FedYogi.
    7. et3-freerider-fedmedian Same as 4 but FedMedian.

  Block C — ET4 (does FL help a NETWORK-impaired free-rider?), ids 8..39:
    Angela is the only vehicle with a degraded uplink (packet loss and/or added
    delay+jitter on her producer->consumer telemetry AND her consumer->FL
    weights upload); bob/claude/daniel keep a pristine network. Everyone trains
    clean and shares the ET3 decoupled eval, so the impairment on angela is the
    only moving part besides FL. Generated as the cross-product of 8 impairment
    levels (3 packet-loss rates, 3 delay/jitter settings, 2 combinations) x 4 FL
    modes (no-FL baseline + FedAvg + FedMedian + FedYogi):
        8..11  et4-loss20-{nofl,fedavg,fedmedian,fedyogi}      angela loss 0.2
        12..15 et4-loss40-*                                    angela loss 0.4
        16..19 et4-loss60-*                                    angela loss 0.6
        20..23 et4-delay100-*                    angela delay 100ms / jitter 25ms
        24..27 et4-delay250-*                    angela delay 250ms / jitter 60ms
        28..31 et4-delay500-*                   angela delay 500ms / jitter 125ms
        32..35 et4-combo-lo-*             angela loss 0.2 + delay 100ms / jitter 25ms
        36..39 et4-combo-hi-*             angela loss 0.4 + delay 250ms / jitter 60ms

  Block C — ET4, class-imbalance sub-axis (does FL help a vehicle whose OWN
  data is class-imbalanced?), ids 40..71:
    Sibling block to the network-impairment one above (added alongside it, not
    replacing it). Angela's producer generation rate for one class is
    throttled — mu_anomalies ("abnormalscarce", starves ANOMALY+ATTACK) or
    mu_normal ("normalscarce", starves NORMAL) — while bob/claude/daniel keep
    the default, balanced rates. Everyone trains clean and shares the same
    decoupled eval as ET3/network-ET4; the consumer's eval-anchor fallback
    (default_vehicle_config.eval_anchor_interval_secs) keeps that eval
    reporting even once angela's own buffers run thin. 8 imbalance levels (4
    abnormalscarce + 4 normalscarce, each mild/moderate/severe/extreme; see
    config/overrides/exp_et4_angela_abnormalscarce_mild.yaml for the
    calibration writeup) x the same 4 FL modes:
        40..43 et4-abnormalscarce-mild-*        mu_anomalies x20
        44..47 et4-abnormalscarce-moderate-*    mu_anomalies x100
        48..51 et4-abnormalscarce-severe-*      mu_anomalies x500
        52..55 et4-abnormalscarce-extreme-*     mu_anomalies x2000
        56..59 et4-normalscarce-mild-*          mu_normal x20
        60..63 et4-normalscarce-moderate-*      mu_normal x100
        64..67 et4-normalscarce-severe-*        mu_normal x500
        68..71 et4-normalscarce-extreme-*       mu_normal x2000

Run each across the three architectures via experiments.py (mlp),
experiments_cnn.py (cnn) and experiments_resnet.py (resnet). Dynamic-noise
experiments were dropped from the paper and are no longer part of the matrix.

Pre-requisites
--------------
Before running this script, vehicle containers must already exist:

    python tests/dash_cli.py create-vehicles

This only needs to be done ONCE.  The runner does NOT create or delete vehicles
between runs — it only starts and stops the services running inside them.

Typical usage
-------------
  # All 71 experiments (7 canonical + 32 network-ET4 + 32 imbalance-ET4), 5 seeds each
  python tests/experiments.py

  # Just the ET4 network-free-rider block (ids 8..39)
  python tests/experiments.py --experiments $(seq 8 39)

  # Just the ET4 class-imbalance free-rider block (ids 40..71)
  python tests/experiments.py --experiments $(seq 40 71)

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

Dynamic-noise injection (legacy, unused by the current matrix)
--------------------------------------------------------------
The mid-run /reset-noise injection helper is retained for ad-hoc use but no
experiment in the current matrix sets dynamic_noise=True. If re-enabled, at the
midpoint of the run the script POSTs to /reset-noise on the dashboard to switch
angela's noise level; on failure a warning is printed and the run continues.
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

# DEFAULT_SEEDS = [42, 123, 456, 789, 1234]
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_RUN_DURATION_SECS = 10 * 60   # 10 mins per run
INTER_RUN_DELAY_SECS = 15             # cooldown between successive runs

# Delays within the startup sequence (seconds)
_DELAY_AFTER_PRODUCE  = 5
_DELAY_AFTER_ATTACKS  = 5
_DELAY_AFTER_CONSUME  = 5

# Timeout for start-wandb specifically — W&B can be slow to initialise
_WANDB_TIMEOUT_SECS = 120


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Each entry:
#   name          : W&B group name and run-name prefix
#   fl            : whether to start federated learning
#   override      : config/overrides/<name>.yaml to apply (None = use defaults).
#                   May also be a list/tuple of profile names, applied in
#                   sequence (each one deep-merged on top of the previous).
#   dynamic_noise : whether to inject mid-run noise into angela via /reset-noise
EXPERIMENTS: dict[int, dict] = {
    # ---- Block A — ET1/ET2 (adversarial-training effect, no FL) ----
    1: {
        "name": "et2-noadv-nofl",
        "fl": False,
        "override": None,
        "dynamic_noise": False,
        "description": "Baseline: default noise gradient, no adversarial training, no FL.",
    },
    2: {
        "name": "et2-adv-nofl",
        "fl": False,
        "override": "exp_advtraining",
        "dynamic_noise": False,
        "description": "Adversarial training ON (default gradient augmentation), no FL.",
    },
    # ---- Block B — ET3 (FL robustness transfer, clean free-rider) ----
    # Identical decoupled eval for all vehicles; only FL on/off + aggregation
    # strategy vary across the block.
    3: {
        "name": "et3-freerider-nofl",
        "fl": False,
        "override": "exp_et3_freerider",
        "dynamic_noise": False,
        "description": (
            "Free-rider transfer baseline: bob/claude/daniel train adversarially "
            "at common noise=1.0, angela trains clean. No FL."
        ),
    },
    4: {
        "name": "et3-freerider-fedavg",
        "fl": True,
        "override": "exp_et3_freerider",
        "dynamic_noise": False,
        "description": "Free-rider + FL (FedAvg). Robustness transfer to angela via aggregation.",
    },
    5: {
        "name": "et3-freerider-fedprox",
        "fl": True,
        "override": ("exp_et3_freerider", "fedprox"),
        "dynamic_noise": False,
        "description": "Same as et3-freerider-fedavg but FedProx (fedprox_mu=0.01).",
    },
    6: {
        "name": "et3-freerider-fedyogi",
        "fl": True,
        "override": ("exp_et3_freerider", "fedyogi"),
        "dynamic_noise": False,
        "description": "Same as et3-freerider-fedavg but FedYogi (server-side adaptive LR).",
    },
    7: {
        "name": "et3-freerider-fedmedian",
        "fl": True,
        "override": ("exp_et3_freerider", "fedmedian"),
        "dynamic_noise": False,
        "description": "Same as et3-freerider-fedavg but FedMedian.",
    },
}


# ---------------------------------------------------------------------------
# Block C — ET4 (does FL help a network-impaired free-rider?)
# ---------------------------------------------------------------------------
# Angela is the ONLY vehicle with a degraded uplink (packet loss and/or added
# delay+jitter on BOTH her producer->consumer telemetry pipeline and her
# consumer->FL-manager weights upload); bob/claude/daniel keep a pristine
# network (0 loss, 0 delay, 0 jitter). Everyone trains clean and the eval is the
# same decoupled HSJA-clean-anchors + fixed sigma-grid target as ET3, so ET4
# numbers are directly comparable — the network impairment on angela is the only
# moving part besides FL on/off + aggregation strategy.
#
# The block is generated programmatically (and appended to EXPERIMENTS with
# contiguous integer ids after the hand-written 1..7 above) as the cross-product
# of:
#   * 8 impairment levels for angela — 3 packet-loss rates, 3 delay/jitter
#     settings, 2 combinations of both (self-contained config overrides under
#     config/overrides/exp_et4_angela_*.yaml, each re-listing all four vehicles
#     since OmegaConf/Hydra replace the `vehicles:` list wholesale on merge);
#   * 4 FL modes — a no-FL baseline plus FedAvg / FedMedian / FedYogi — so the
#     benefit FL brings to angela can be read off directly at each level.
# That is 8 * 4 = 32 experiments (ids 8..39). Every architecture runner
# (experiments_cnn.py / experiments_resnet.py / experiments_all.py) imports this
# same EXPERIMENTS dict, so the ET4 block runs across mlp/cnn/resnet unchanged.

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
                "dynamic_noise": False,
                "description": (
                    f"ET4 network free-rider: angela {level_desc}, peers pristine "
                    f"network. {fl_desc}."
                ),
            }
            next_id += 1


_extend_with_et4(EXPERIMENTS)


# ---------------------------------------------------------------------------
# Block C — ET4 (does FL help a CLASS-IMBALANCED free-rider?), ids 40..71
# ---------------------------------------------------------------------------
# Sibling sub-axis of the network-impairment block above, added alongside it
# rather than replacing it (the network block is untouched). Angela is the
# only vehicle whose producer generation rate for one class is throttled —
# via mu_anomalies (starving ANOMALY+ATTACK, "abnormalscarce") or mu_normal
# (starving NORMAL, "normalscarce"); bob/claude/daniel keep the default,
# balanced rates. Everyone trains clean and shares the ET3/network-ET4
# decoupled eval (HSJA on clean anchors + fixed sigma-grid), rescued from
# buffer starvation by the consumer's eval-anchor fallback (see
# default_vehicle_config.eval_anchor_interval_secs and
# config/overrides/exp_et4_angela_abnormalscarce_mild.yaml for the full
# mechanism writeup and the Monte-Carlo calibration behind the multipliers).
# Generated as the cross-product of 8 imbalance levels (4 abnormalscarce + 4
# normalscarce, each mild/moderate/severe/extreme) x 4 FL modes (no-FL
# baseline + FedAvg / FedMedian / FedYogi) = 32 experiments (ids 40..71).

# (short_tag, override_profile, human description of angela's imbalance)
_ET4_IMBALANCE_LEVELS: list[tuple[str, str, str]] = [
    ("abnormalscarce-mild",     "exp_et4_angela_abnormalscarce_mild",
     "mu_anomalies x20 (~635 anomaly+attack samples/600s, vs. ~12,600 baseline)"),
    ("abnormalscarce-moderate", "exp_et4_angela_abnormalscarce_moderate",
     "mu_anomalies x100 (~139 anomaly+attack samples/600s)"),
    ("abnormalscarce-severe",   "exp_et4_angela_abnormalscarce_severe",
     "mu_anomalies x500 (~32 anomaly+attack samples/600s, borderline batch_size)"),
    ("abnormalscarce-extreme",  "exp_et4_angela_abnormalscarce_extreme",
     "mu_anomalies x2000 (~11 anomaly+attack samples/600s, local training on that class effectively blocked)"),
    ("normalscarce-mild",       "exp_et4_angela_normalscarce_mild",
     "mu_normal x20 (~857 normal samples/600s, vs. ~16,900 baseline)"),
    ("normalscarce-moderate",   "exp_et4_angela_normalscarce_moderate",
     "mu_normal x100 (~181 normal samples/600s)"),
    ("normalscarce-severe",     "exp_et4_angela_normalscarce_severe",
     "mu_normal x500 (~42 normal samples/600s, borderline batch_size)"),
    ("normalscarce-extreme",    "exp_et4_angela_normalscarce_extreme",
     "mu_normal x2000 (~14 normal samples/600s, local training on NORMAL effectively blocked)"),
]


def _extend_with_et4_imbalance(experiments: dict[int, dict]) -> None:
    """Append the 32-cell ET4 class-imbalance free-rider block to *experiments* in place."""
    next_id = max(experiments) + 1
    for level_tag, level_profile, level_desc in _ET4_IMBALANCE_LEVELS:
        for mode_tag, fl_on, extra in _ET4_MODES:
            override = level_profile if extra is None else (level_profile, extra)
            fl_desc = "no FL" if not fl_on else f"FL ({mode_tag[len('fed'):].upper()})"
            experiments[next_id] = {
                "name": f"et4-{level_tag}-{mode_tag}",
                "fl": fl_on,
                "override": override,
                "dynamic_noise": False,
                "description": (
                    f"ET4 class-imbalance free-rider: angela {level_desc}, peers "
                    f"balanced rates. {fl_desc}."
                ),
            }
            next_id += 1


_extend_with_et4_imbalance(EXPERIMENTS)


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

    overrides = exp["override"]
    if overrides:
        if isinstance(overrides, (list, tuple)):
            for profile in overrides:
                cli("apply-override", profile)
        else:
            cli("apply-override", overrides)

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
    # 7. Run — with optional mid-run noise injection (legacy; unused by the
    #    current matrix, all entries set dynamic_noise=False).
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
    #      stop_attacks -> stop_fl -> stop_consumers ->
    #      stop_producers -> stop_wandb
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
        help="Experiment numbers to run (default: all; 1..7 canonical + 8..39 network-ET4 + 40..71 imbalance-ET4).",
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
