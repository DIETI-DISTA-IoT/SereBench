#!/usr/bin/env python3
"""Parallel driver for the SereBench OFFLINE experiment sweep.

The offline campaign is embarrassingly parallel: every ``(experiment, seed,
arch)`` cell builds and tears down its own in-process cluster, with no shared
state between cells. This driver runs the whole grid as independent
subprocesses, ``--jobs`` at a time, so a many-core box (e.g. 50 vCPUs) is
actually used — the plain ``python -m offline_simulation.experiments`` runner
walks the grid strictly sequentially.

Each cell is one child invocation of that same runner::

    python -m offline_simulation.experiments \
        --experiments <ID> --seeds <SEED> --arch <ARCH> \
        --run-duration <D> --settle <S> --wandb-mode <MODE> [--no-attacks]

so results are byte-for-byte identical to a sequential run — only the outer
scheduling changes. Per-cell stdout/stderr is captured to its own log file
under ``--log-dir``; W&B (offline by default) is pointed at a shared
``WANDB_DIR`` so the whole batch can be ``wandb sync``'d in one go afterwards.

Sizing the pool
---------------
A cell is surprisingly light on CPU: its training loop is sleep-throttled
(``training_freq_seconds``), so a measured cell averages only ~0.4 effective
cores (spiking higher only inside the periodic HopSkipJump benchmark windows)
and ~1 GB RAM. The sweep is therefore **memory-bound, not CPU-bound** — you can
run many more cells at once than you have cores. This driver pins each child's
math-library intra-op threads to ``--threads-per-job`` (default 1) so those
benchmark spikes stay contained, and defaults ``--jobs`` to the vCPU count.
Guidance:
  * CPU: budget ~0.5–1 core per concurrent cell, so ~50 cells comfortably fills
    50 vCPUs with headroom for HSJA spikes. If ``htop`` shows the cores idling,
    raise ``--jobs`` *above* the core count (the bottleneck is RAM, not CPU).
  * RAM: budget ~1 GB per concurrent cell. On a RAM-limited box lower ``--jobs``
    (``--jobs`` × ~1 GB should fit comfortably in memory).

Examples
--------
  # The ET5 FL-flow-stress block (ids 40..76), 3 seeds, MLP, 10-min runs.
  # --jobs defaults to the vCPU count; ET5-mlp is 37x3 = 111 cells.
  python -m offline_simulation.run_sweep --jobs 50

  # Fill 50 vCPUs: the whole campaign (1..76) x 3 seeds x 3 archs, W&B offline
  python -m offline_simulation.run_sweep \
      --experiments $(seq 1 76) --arch all --jobs 50 --run-duration 600

  # Just ET5, every architecture, quicker 5-min runs
  python -m offline_simulation.run_sweep --arch all --run-duration 300 --jobs 50

  # See exactly what would run, without running it
  python -m offline_simulation.run_sweep --dry-run

After an offline sweep, push every run to W&B in one shot:
  wandb sync <log-dir>/wandb/*-run-*
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Import the canonical matrix so --experiments validates against the same ids
# the sequential runner exposes (1..7 canonical, 8..39 ET4, 40..76 ET5).
from .experiments import EXPERIMENTS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ET5 = [i for i in EXPERIMENTS if EXPERIMENTS[i]["name"].startswith("et5-")]
DEFAULT_SEEDS = [42, 123, 456]

# Env vars that cap per-process math-library intra-op threads. Pinning these
# keeps each child roughly to its own vehicle-thread parallelism so N children
# don't fight over the cores.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _expand_archs(archs: list[str]) -> list[str]:
    out: list[str] = []
    for a in archs:
        for x in (["mlp", "cnn", "resnet"] if a == "all" else [a]):
            if x not in out:
                out.append(x)
    return out


def build_jobs(experiments, seeds, archs):
    """The full (arch, experiment, seed) grid, one tuple per subprocess."""
    return [(a, e, s) for a in archs for e in experiments for s in seeds]


def _child_cmd(exp_id, seed, arch, args):
    cmd = [
        sys.executable, "-m", "offline_simulation.experiments",
        "--experiments", str(exp_id),
        "--seeds", str(seed),
        "--arch", arch,
        "--run-duration", str(args.run_duration),
        "--settle", str(args.settle),
        "--wandb-mode", args.wandb_mode,
    ]
    if args.no_attacks:
        cmd.append("--no-attacks")
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.config_dir:
        cmd += ["--config-dir", args.config_dir]
    return cmd


def run_job(job, args, log_dir, record_wandb):
    """Run one (arch, experiment, seed) cell as a child process; return a record."""
    arch, exp_id, seed = job
    name = EXPERIMENTS[exp_id]["name"]
    arch_suffix = "" if arch == "mlp" else f"-{arch}"
    tag = f"{name}{arch_suffix}_seed{seed}"
    log_path = log_dir / f"{tag}.log"

    env = os.environ.copy()
    for var in _THREAD_ENV_VARS:
        env[var] = str(args.threads_per_job)
    if record_wandb:
        # wandb nests a `wandb/` subdir under WANDB_DIR, so every child's run
        # lands in <log-dir>/wandb/*-run-* — one place to `wandb sync` later.
        env["WANDB_DIR"] = str(log_dir)
        env.setdefault("WANDB_SILENT", "true")

    cmd = _child_cmd(exp_id, seed, arch, args)
    start = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"# {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT),
                              stdout=lf, stderr=subprocess.STDOUT)
    return {
        "tag": tag, "arch": arch, "experiment": exp_id, "name": name,
        "seed": seed, "returncode": proc.returncode,
        "duration_s": round(time.time() - start, 1), "log": str(log_path),
    }


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="offline_simulation.run_sweep",
        description="Run the SereBench OFFLINE experiment grid in parallel across "
                    "many CPU cores (one subprocess per experiment/seed/arch cell).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--experiments", nargs="+", type=int, choices=list(EXPERIMENTS),
                   default=DEFAULT_ET5, metavar="N",
                   help="Experiment ids to sweep (default: the ET5 block 40..76). "
                        "1..7 canonical, 8..39 ET4, 40..76 ET5.")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, metavar="S",
                   help=f"Seeds — one run per seed (default: {DEFAULT_SEEDS}).")
    p.add_argument("--arch", nargs="+", choices=["mlp", "cnn", "resnet", "all"],
                   default=["mlp"],
                   help="Architecture(s); 'all' expands to mlp cnn resnet, one job each.")
    p.add_argument("--run-duration", type=float, default=600.0, metavar="SECS",
                   help="Seconds per run (default: 600).")
    p.add_argument("--jobs", type=int, default=None, metavar="J",
                   help="Concurrent subprocesses (default: the vCPU count). Each cell "
                        "averages ~0.5-1 core + ~1 GB RAM, so the sweep is memory-bound "
                        "— raise this above the core count if CPUs idle, lower it if RAM-limited.")
    p.add_argument("--threads-per-job", type=int, default=1, metavar="T",
                   help="Math-library intra-op threads per child (OMP/MKL/...); "
                        "default 1 so children don't oversubscribe.")
    p.add_argument("--wandb-mode", default="offline",
                   choices=["online", "offline", "disabled"],
                   help="W&B mode for every run (default: offline — safest for a big "
                        "parallel batch; `wandb sync` the log-dir/wandb afterwards).")
    p.add_argument("--settle", type=float, default=1.0,
                   help="Startup warmup scale passed to each run (default 1.0 = platform "
                        "timing). Keep >= 0.5 so consumers warm their anomaly buffers "
                        "before attacks — a smaller value can leave a model at epoch 0.")
    p.add_argument("--no-attacks", action="store_true",
                   help="Pass --no-attacks to every run (NB: stalls training — see docs).")
    p.add_argument("--no-wandb", action="store_true",
                   help="Pass --no-wandb to every run (no metrics recorded).")
    p.add_argument("--config-dir", default=None,
                   help="Path to the SereBench config/ dir (defaults to the repo's).")
    p.add_argument("--log-dir", default=None,
                   help="Where to write per-cell logs + a manifest (default: "
                        "./sweep_logs/<timestamp>).")
    p.add_argument("--dry-run", action="store_true",
                   help="List the jobs that would run, then exit.")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    archs = _expand_archs(args.arch)
    jobs = build_jobs(args.experiments, args.seeds, archs)

    vcpus = os.cpu_count() or 8
    n_jobs = args.jobs if args.jobs and args.jobs > 0 else vcpus
    n_jobs = min(n_jobs, len(jobs))

    log_dir = Path(args.log_dir) if args.log_dir else (
        REPO_ROOT / "sweep_logs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir.mkdir(parents=True, exist_ok=True)
    record_wandb = not args.no_wandb and args.wandb_mode != "disabled"

    _log("SereBench OFFLINE parallel sweep")
    _log(f"  Detected vCPUs : {vcpus}")
    _log(f"  Experiments    : {args.experiments}")
    _log(f"  Seeds          : {args.seeds}")
    _log(f"  Architectures  : {archs}")
    _log(f"  Total cells    : {len(jobs)}")
    _log(f"  Concurrency    : {n_jobs}  (threads/job={args.threads_per_job})")
    _log(f"  Run duration   : {args.run_duration:.0f}s each")
    _log(f"  W&B mode       : {args.wandb_mode}{' (disabled)' if not record_wandb else ''}")
    _log(f"  Log dir        : {log_dir}")
    est_waves = (len(jobs) + n_jobs - 1) // n_jobs
    _log(f"  Rough ETA      : ~{_fmt_hms(est_waves * args.run_duration)} "
         f"(~{est_waves} waves x {args.run_duration:.0f}s, ignoring per-run overhead)")

    if args.dry_run:
        for arch, exp_id, seed in jobs:
            nm = EXPERIMENTS[exp_id]["name"]
            _log(f"  [dry] id={exp_id:<3} {nm}{'' if arch=='mlp' else '-'+arch} seed={seed}")
        return 0

    results: list[dict] = []
    failed: list[dict] = []
    completed = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(run_job, job, args, log_dir, record_wandb): job
                   for job in jobs}
        try:
            for fut in as_completed(futures):
                rec = fut.result()
                results.append(rec)
                completed += 1
                ok = rec["returncode"] == 0
                if not ok:
                    failed.append(rec)
                elapsed = time.time() - started
                avg = elapsed / completed
                remaining = len(jobs) - completed
                eta = avg * remaining / n_jobs
                status = "ok " if ok else "FAIL"
                _log(f"  [{status}] {completed}/{len(jobs)} {rec['tag']} "
                     f"({rec['duration_s']:.0f}s, rc={rec['returncode']}) | "
                     f"elapsed {_fmt_hms(elapsed)} | eta ~{_fmt_hms(eta)}"
                     + ("" if ok else f" | see {rec['log']}"))
        except KeyboardInterrupt:
            _log("Interrupted — cancelling pending cells (running ones finish)...")
            for f in futures:
                f.cancel()
            raise

    manifest = {
        "started": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - started, 1),
        "vcpus": vcpus, "jobs": n_jobs, "threads_per_job": args.threads_per_job,
        "run_duration_s": args.run_duration, "wandb_mode": args.wandb_mode,
        "total": len(jobs), "completed": completed, "failed": len(failed),
        "cells": sorted(results, key=lambda r: (r["arch"], r["experiment"], r["seed"])),
    }
    manifest_path = log_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _log("=" * 60)
    _log(f"  Completed : {completed - len(failed)} / {len(jobs)}")
    _log(f"  Wall time : {_fmt_hms(time.time() - started)}")
    _log(f"  Manifest  : {manifest_path}")
    if record_wandb and args.wandb_mode == "offline":
        _log(f"  Sync W&B  : wandb sync {log_dir / 'wandb'}/*-run-*")
    if failed:
        _log(f"  FAILED ({len(failed)}):")
        for rec in failed:
            _log(f"    - {rec['tag']}  (rc={rec['returncode']})  {rec['log']}")
    _log("=" * 60)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
