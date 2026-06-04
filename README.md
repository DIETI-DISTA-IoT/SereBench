
# SereBench.

This is the official implementation of the paper:

 ## SereBench *Adversarial Robustness Benchmarking for Security and Reliability Event Classification in Networked Railway CPS*

by _Alessandra Rizzardi, Raffaele Della Corte, Jesus F. Cevallos M., Simona De Vivo, Sabrina Sicari, Domenico Cotroneo, and Alberto Coen-Porisini_


## Usage

### Clone the repo:

This is a nested repo, so be sure to clone the repo recursively, using the `--recursive` flag with the `git clone` command. Here's the syntax:

```bash
git clone --recursive https://github.com/DIETI-DISTA-IoT/SereBench
```

This will clone the repository and all its submodules.

Alternatively, if you've already cloned the repository without the `--recursive` flag, you can use the following command to initialize and update the submodules:

```bash
git submodule update --init --recursive
```

This will fetch and checkout the submodules recursively.


## Set up:

### Installing Docker

**On Windows:**

Download [Docker Desktop](https://www.docker.com/products/docker-desktop/) and run the program.

**On WSL:**

Follow [this guide](https://docs.docker.com/engine/install/ubuntu/) to install Docker CE for the Ubuntu distribution (commands for other distributions can be found on the same website).
Perform the post installation steps, as shown [here](https://docs.docker.com/engine/install/linux-postinstall/).
(Optional) You can also make Docker Daemon start on WSL initialization, by adding:
```
[boot]
systemd=true
```
to your `/etc/wsl.conf` within your WSL distribution and restart it with `wsl.exe --shutdown`.

To verify that Docker works, run `docker version` and `docker compose version` to verify that Docker Compose got installed as well.

**On Linux:**

Run `sudo apt-get update` and `sudo apt-get install docker.io docker-compose`.
Verify it works by executing `docker --version` and `docker compose --version`.

## Installing Make
**On Windows:**

Download Make from the [GnuWin32](https://gnuwin32.sourceforge.net/packages/make.htm) project.
Run the installer and add the `bin` directory to your `PATH` (inside Windows' Environment Variables).
Verify it works by opening a command prompt and executing `make --version`.

**On WSL or Linux:**

Run `sudo apt-get update` and `sudo apt-get install make`.
Verify it works by executing `make --version`.


## Building vehicle images and dashboard:

Use `make all` to build all images from scratch:

    make all

### Rebuilding after code changes

There are two selective-cache rebuild modes. Both skip the expensive base-image and system-package layers; the difference is whether Python dependencies are re-installed.

#### `scache` — re-pull code **and** re-install pip dependencies

Use this when you have modified `requirements.txt` in any subrepo, or when you have added a new Python library to the project code:

    make all-scache

Or for a single service:

    make build-producer-scache
    make build-consumer-scache
    make build-dashboard-scache
    make build-wandber-scache

Docker layer order (simplified):
```
[cached]  base image + apt packages
[busted]  pip install -r requirements.txt   ← re-runs
[busted]  git clone <app code>              ← re-runs
[busted]  git clone of-core                 ← re-runs
```

#### `scache-nolib` — re-pull code only, **keep pip cache**

Use this when your code changes do **not** introduce new libraries (the common case during development). The pip install layers are reused from the last full build, so only the git clones are repeated. This is significantly faster when packages like `torch` are involved:

    make all-scache-nolib

Or for a single service:

    make build-producer-scache-nolib
    make build-consumer-scache-nolib
    make build-dashboard-scache-nolib
    make build-wandber-scache-nolib

Docker layer order (simplified):
```
[cached]  base image + apt packages
[cached]  pip install -r requirements.txt   ← reused from last build
[busted]  git clone <app code>              ← re-runs
[busted]  git clone of-core                 ← re-runs
```

> **How it works:** Each Dockerfile now contains two `ARG` placeholders at different positions.
> `CACHE_BUST` sits before the `pip install` step — passing a fresh timestamp invalidates pip and everything below it (`scache`).
> `CODE_BUST` sits after `pip install` but before the `git clone` steps — passing only this timestamp leaves pip cached and only re-runs the clones (`scache-nolib`).
> `requirements.txt` is copied from the checked-out submodule directory in the build context (not from the git clone inside Docker), so pip can run before the clone.

### Launching:

> **IMPORTANT** For using the wandb logging dashboard, you should have a file called **".env"** in the project's root folder containing your wandb api key under the WANDB_API_KEY voice:

```.env file
# .env file content:

WANDB_API_KEY=your_wandb_api_key_here

```

Use `docker compose up` to start the whole cluster:
    docker compose up -d

Otherwise, you can load different services separately:

    docker compose up -d zookeeper kafka wandber dashboard
    
    docker compose up -d dashboard

You can use `docker compose stop` to stop a service and 'docker compose up -d' to reload it (e.g. after changing the configs or code):

    docker compose stop dashboard && docker compose up -d dashboard

---

## Running Experiments from the Command Line

SereBench ships with a CLI client (`tests/dash_cli.py`) that lets you launch, configure, and monitor experiments entirely from a terminal, without opening the web dashboard.

The CLI sends HTTP requests to the dashboard container's REST API (exposed on `http://localhost:5000` by default) and keeps a local copy of the experiment configuration in `~/.serebench/dash_cli_state.yaml`.

> **Note:** The CLI only controls an already-running dashboard container. You still need to bring up the infrastructure with `docker compose up -d` first.

### Prerequisites

Install the CLI dependencies on the **host** (outside Docker):

```bash
pip install -r requirements.txt
```

`hydra-core` and `omegaconf` are optional — the CLI degrades gracefully without them, but `init-config`, `apply-override`, and the `set` key-validation feature all require them.

---

### Quick-start: full experiment in five commands

```bash
# 1. Start the infrastructure
docker compose up -d

# 2. Initialise CLI state from your Hydra config
python tests/dash_cli.py init-config

# 3. Create the vehicle containers (dashboard spawns them via the Docker socket)
python tests/dash_cli.py create-vehicles

# 4. Start the experiment
python tests/dash_cli.py start-experiment

# 5. When done, shut everything down and remove vehicles
python tests/dash_cli.py shutdown
python tests/dash_cli.py delete-vehicles
```

---

### Configuration management

The CLI state is a plain YAML file (`~/.serebench/dash_cli_state.yaml`) that mirrors the experiment configuration. You edit it before starting the experiment — changes made afterwards have no effect on a running cluster.

#### Initialise state from the Hydra config

```bash
python tests/dash_cli.py init-config
```

This reads `config/default.yaml` (plus any Hydra overrides you pass) and writes the resolved config to the state file. Re-run it any time you want to reset to the file-based defaults.

#### Inspect the current state

```bash
python tests/dash_cli.py show-config
```

#### List available override profiles

```bash
python tests/dash_cli.py list-overrides
```

Output example:
```
Available override profiles (…/config/overrides):
  dev
  fedavg
  fedmedian
  fedprox
  fedyogi
  nofed
  slowsere
  sw
```

#### Apply an override profile

Because the dashboard container loads its config at startup and cannot reload it at runtime, the CLI replicates the `OmegaConf.merge` logic client-side. `apply-override` reads a `config/overrides/<profile>.yaml` file and deep-merges it into your local state, then shows you exactly what changed:

```bash
python tests/dash_cli.py apply-override fedyogi
```

Example output:
```diff
Override file: …/config/overrides/fedyogi.yaml
Changes that will be applied:
--- state (before)
+++ state (after fedyogi)
@@ … @@
-  aggregation_strategy: fedavg
+  aggregation_strategy: fedyogi
-  aggregation_interval_secs: 25
+  aggregation_interval_secs: 20
-  run_name: My_awesome_FAIR_run
+  run_name: fed_YOGI
…
Applied override 'fedyogi' and saved ~/.serebench/dash_cli_state.yaml
```

Use `--dry-run` to preview without saving:

```bash
python tests/dash_cli.py apply-override dev --dry-run
```

You can stack multiple overrides — each call merges on top of the previous state:

```bash
python tests/dash_cli.py init-config
python tests/dash_cli.py apply-override fedyogi   # sets FL strategy + intervals
python tests/dash_cli.py apply-override dev        # adds proxy settings on top
```

You can also point to any arbitrary YAML file by path:

```bash
python tests/dash_cli.py apply-override /path/to/my_custom_override.yaml
```

#### Set or unset individual keys

Use dot-path notation. Values are parsed as YAML so you can pass scalars, booleans, lists, and inline dicts:

```bash
# scalars
python tests/dash_cli.py set federated_learning.aggregation_strategy fedyogi
python tests/dash_cli.py set federated_learning.aggregation_interval_secs 30
python tests/dash_cli.py set wandb.online false

# a list
python tests/dash_cli.py set security_manager.probe_metrics '[RTT, CPU, MEM]'

# remove a key
python tests/dash_cli.py unset default_vehicle_config.common_env_vars.http_proxy
```

If `hydra-core` is installed, `set` validates the key against the composed schema and suggests corrections for typos.

---

### Step-by-step experiment control

Instead of `start-experiment` (which runs everything automatically), you can drive each phase manually:

```bash
# Start data producers (one per vehicle)
python tests/dash_cli.py produce-all

# Start anomaly detection consumers (one per vehicle)
python tests/dash_cli.py consume-all

# Start the security manager (sends security_manager section from state)
python tests/dash_cli.py start-security-manager

# Start federated learning aggregation
python tests/dash_cli.py start-federated-learning

# Start W&B logging (sends wandb section from state)
python tests/dash_cli.py start-wandb

# Enable automatic attack injection
python tests/dash_cli.py start-automatic-attacks
```

Each step has a corresponding stop command:

```bash
python tests/dash_cli.py stop-automatic-attacks
python tests/dash_cli.py stop-wandb
python tests/dash_cli.py stop-federated-learning
python tests/dash_cli.py stop-security-manager
python tests/dash_cli.py stop-consuming-all
python tests/dash_cli.py stop-producing-all
```

Or call `shutdown` to stop everything in one shot:

```bash
python tests/dash_cli.py shutdown
```

---

### Attack control

```bash
# Trigger an attack from a specific vehicle
python tests/dash_cli.py start-attack angela
python tests/dash_cli.py stop-attack angela

# Use the pre-configured attacking vehicles list (from config attack.preconf_attacking_vehicles)
python tests/dash_cli.py start-preconf-attack
python tests/dash_cli.py stop-preconf-attack

# Toggle scheduled automatic attacks
python tests/dash_cli.py start-automatic-attacks
python tests/dash_cli.py stop-automatic-attacks

# Toggle mitigation on the security manager
python tests/dash_cli.py start-mitigation
python tests/dash_cli.py stop-mitigation
```

---

### Status and logs

```bash
# Check that the dashboard container is reachable
python tests/dash_cli.py health

# Get the running status of a specific vehicle (producer + consumer)
python tests/dash_cli.py vehicle-status angela

# Print the last 100 log lines from a container
python tests/dash_cli.py logs angela_producer
python tests/dash_cli.py logs angela_consumer

# Stream new log lines every 5 seconds (Ctrl-C to stop)
python tests/dash_cli.py logs angela_producer --watch
python tests/dash_cli.py logs dashboard --watch --interval-secs 10
```

---

### Monitoring experiment metrics via W&B

The `wandb-monitor` command polls the W&B GraphQL API and prints the latest numeric metrics for your run. It reads `wandb.entity`, `wandb.project_name`, and `wandb.run_name` from the CLI state (or you can pass them as flags).

```bash
# One-shot summary
python tests/dash_cli.py wandb-monitor

# Continuous polling every 15 seconds (Ctrl-C to stop)
python tests/dash_cli.py wandb-monitor --watch

# Override entity / project / run name without touching the state
python tests/dash_cli.py wandb-monitor --entity my_org --project OPEN_FAIR --run-name fed_YOGI

# Show up to 50 metrics, polling every 30 s
python tests/dash_cli.py wandb-monitor --watch --max-metrics 50 --interval-secs 30
```

The `WANDB_API_KEY` is read from the environment or from the `.env` file in the repo root.

---

### Global flags

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url URL` | `http://localhost:5000` | Dashboard container URL |
| `--state-file PATH` | `~/.serebench/dash_cli_state.yaml` | Local config state file |
| `--config-dir PATH` | `<repo>/config` | Hydra config directory |
| `--profile NAME` | _(none)_ | Override profile to apply at `init-config` time |
| `--hydra-override KEY=VAL` | _(none)_ | Extra Hydra override (repeatable, used with `init-config`) |
| `--timeout SECS` | `120` | HTTP request timeout |

All flags must come **before** the subcommand:

```bash
python tests/dash_cli.py --base-url http://192.168.1.10:5000 start-experiment
python tests/dash_cli.py --state-file ./my_run.yaml show-config
```

---

### Complete command reference

| Command | Description |
|---------|-------------|
| `init-config` | Write `~/.serebench/dash_cli_state.yaml` from Hydra config |
| `show-config` | Print the current persisted state |
| `list-overrides` | List override profiles in `config/overrides/` |
| `apply-override <profile>` | Deep-merge an override profile into the state |
| `set <key> <value>` | Set a single key by dot-path |
| `unset <key>` | Remove a key from the state |
| `create-vehicles` | Spawn all vehicle containers (producer + consumer pairs) |
| `delete-vehicles` | Remove all vehicle containers |
| `produce-all` | Start all producers |
| `stop-producing-all` | Stop all producers |
| `consume-all` | Start all consumers |
| `stop-consuming-all` | Stop all consumers |
| `start-federated-learning` | Start FL aggregation in the wandber node |
| `stop-federated-learning` | Stop FL aggregation |
| `start-wandb` | Start W&B logging |
| `stop-wandb` | Stop W&B logging |
| `start-security-manager` | Start the security manager |
| `stop-security-manager` | Stop the security manager |
| `start-mitigation` | Enable attack mitigation |
| `stop-mitigation` | Disable attack mitigation |
| `start-attack <vehicle>` | Trigger an attack from the named vehicle |
| `stop-attack <vehicle>` | Stop the attack from the named vehicle |
| `start-automatic-attacks` | Enable scheduled automatic attack injection |
| `stop-automatic-attacks` | Disable scheduled automatic attack injection |
| `start-preconf-attack` | Start the pre-configured attack campaign |
| `stop-preconf-attack` | Stop the pre-configured attack campaign |
| `start-experiment` | Run the full startup sequence automatically |
| `shutdown` | Stop everything in reverse order |
| `health` | Check dashboard container reachability |
| `vehicle-status <vehicle>` | Print running status of a vehicle |
| `logs <container>` | Print / stream container logs |
| `wandb-monitor` | Poll W&B and print metric summary |

---

## Automated Multi-Run Experiments (`tests/experiments.py`)

For systematic benchmarking with statistical significance, SereBench ships with
`tests/experiments.py` — a batch runner that executes all canonical experiments
automatically, repeating each one with multiple random seeds.

Unlike `dash_cli.py` (which is interactive and issues one command at a time),
`experiments.py` drives the full lifecycle of every run: it configures the CLI
state, starts and stops services in the correct order, sleeps for the run
duration, and shuts down cleanly before moving to the next run.

Vehicle containers are created **once** before the batch starts and are reused
across all runs. Only the services running inside them (producers, consumers,
W&B logger, FL aggregator) are cycled per run.

---

### Pre-requisites

```bash
# 1. Infrastructure must be up
docker compose up -d

# 2. Python dependencies on the host (same as for dash_cli.py)
pip install requests pyyaml hydra-core omegaconf

# 3. Create vehicle containers once — do NOT repeat between runs
python tests/dash_cli.py create-vehicles
```

---

### Running the canonical experiments

```bash
# All 6 experiments, 5 seeds each (~30 h at 1 h/run)
python tests/experiments.py

# A specific subset of experiments
python tests/experiments.py --experiments 1 2

# Fewer seeds for a quick validation
python tests/experiments.py --seeds 42 123 --run-duration 600

# Continue past a failed run instead of aborting the whole batch
python tests/experiments.py --skip-on-error

# Non-default dashboard URL
python tests/experiments.py --base-url http://192.168.1.10:5000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--experiments N [N ...]` | all (1–6) | Experiment numbers to run |
| `--seeds S [S ...]` | `42 123 456 789 1234` | One independent run per seed |
| `--run-duration SECS` | `3600` | Duration of each run (seconds) |
| `--base-url URL` | `http://localhost:5000` | Dashboard URL |
| `--skip-on-error` | off | Log failures and continue instead of aborting |

---

### Startup order within each run

Every run follows this fixed sequence, fully synchronous — the next step only
begins once the HTTP call for the current step has returned successfully:

```
1. init-config               reset CLI state to defaults
2. apply-override <profile>  merge experiment-specific vehicle config (if any)
3. set seed + wandb.run_name configure per-run parameters
4. start-wandb               W&B logger first, so no metrics are missed
                             (timeout: 120 s — W&B can be slow to initialise)
5. produce-all               start data producers
   sleep 15 s               let producers warm up
6. start-automatic-attacks   inject attack-class events into the stream
   sleep 15 s               let the attack stream stabilise
7. consume-all               start consumers (they now see a live, mixed stream)
   sleep 10 s               let consumers initialise
8. start-federated-learning  FL experiments only

   --- sleep(run_duration) ---
   (for dynamic-noise experiments: sleep(run_duration/2), inject noise
    into angela via POST /reset-noise, sleep(run_duration/2))

9. shutdown                  sequential teardown inside the dashboard:
                             security-manager → FL → consumers →
                             producers → attacks → wandb
   sleep 30 s               inter-run cooldown
```

Vehicle containers are **not** deleted after each run.

---

### Canonical experiment matrix

| # | W&B group | FL | Adversarial training | Config override |
|---|-----------|----|-----------------------|-----------------|
| 1 | `noadvtraining-nofl` | No | No | _(none)_ |
| 2 | `advtraining-nofl` | No | Yes (all vehicles, own noise level) | `exp_advtraining.yaml` |
| 3 | `dynamic-noise-fl` | Yes | Yes (bob/claude/daniel); angela noise injected mid-run | `exp_dynamic_noise_phase1.yaml` |
| 4 | `dynamic-noise-nofl` | No | Yes (bob/claude/daniel); angela noise injected mid-run | `exp_dynamic_noise_phase1.yaml` |
| 5 | `noadvtraining-fl` | Yes | No | _(none)_ |
| 6 | `advtraining-fl` | Yes | Yes (bob/claude/daniel); angela trains clean | `exp_advtraining_fl.yaml` |

W&B runs are named `{group}_seed{seed}_run{N}` and tagged with `wandb.group`
so they can be filtered together in the W&B UI.

---

### Editing an existing experiment

Each experiment is defined by a single dict entry in `EXPERIMENTS` at the top
of `tests/experiments.py`:

```python
EXPERIMENTS: dict[int, dict] = {
    1: {
        "name": "noadvtraining-nofl",   # W&B group name and run-name prefix
        "fl": False,                     # whether to start federated learning
        "override": None,                # config/overrides/<name>.yaml, or None
        "dynamic_noise": False,          # inject noise into angela mid-run?
        "description": "...",
    },
    ...
}
```

To change a parameter for an existing experiment — for example to use
`fedyogi` instead of the default `fedavg` strategy for experiment 5 — you
have two options:

**Option A — point to a different override file:**

```python
5: {
    "name": "noadvtraining-fl-yogi",
    "fl": True,
    "override": "fedyogi",   # config/overrides/fedyogi.yaml already exists
    "dynamic_noise": False,
    "description": "FL with FedYogi, no adversarial training.",
},
```

**Option B — add `set` calls inside `_run_one` for one-off tweaks:**

```python
# inside _run_one, after apply-override:
if exp["name"] == "noadvtraining-fl-yogi":
    cli("set", "federated_learning.aggregation_strategy", "fedyogi")
```

Option A is cleaner and keeps experiment definitions declarative; option B is
useful for quick experiments that do not warrant a new override file.

---

### Adding a new experiment

**Step 1 — define the vehicle configuration in a new override file**

Create `config/overrides/my_experiment.yaml`. The file is deep-merged on top of
`config/default.yaml`, so you only need to list the keys that differ. The
`vehicles` list **replaces** the default list entirely when merged, so include
all four vehicles if you override it:

```yaml
# config/overrides/my_experiment.yaml

# Example: all vehicles train adversarially at their own noise level,
# but with a faster FL aggregation interval.
default_vehicle_config:
  adversarial_training: True

federated_learning:
  aggregation_interval_secs: 15
```

Or with per-vehicle noise:

```yaml
# config/overrides/my_experiment.yaml
vehicles:
  - angela:
      Mp_std: 0.5
      Bp_std: 0.5
      adversarial_training: True
  - bob:
      Mp_std: 0.5
      Bp_std: 0.5
      adversarial_training: True
  - claude:
      Mp_std: 0.5
      Bp_std: 0.5
      adversarial_training: True
  - daniel:
      Mp_std: 0.5
      Bp_std: 0.5
      adversarial_training: True
```

**Step 2 — register the experiment in `tests/experiments.py`**

Add a new entry to the `EXPERIMENTS` dict. Pick an unused integer key:

```python
7: {
    "name": "my-experiment",
    "fl": True,
    "override": "my_experiment",   # name without .yaml extension
    "dynamic_noise": False,
    "description": "Uniform noise=0.5, adversarial training, FL.",
},
```

**Step 3 — run it**

```bash
python tests/experiments.py --experiments 7 --seeds 42 123 456 --run-duration 3600
```

That's it. The runner picks up the new entry, applies your override file before
each run, and logs results to W&B under the group name `my-experiment`.

---

### Writing a new experiment script from scratch

If the batch-runner model does not fit your use case (e.g. you need to sweep a
continuous hyperparameter, interleave runs with post-processing, or integrate
with an external scheduler), you can write a standalone script that calls
`dash_cli.py` via `subprocess`. The pattern used in `experiments.py` is
straightforward to replicate:

```python
import subprocess, sys, time
from pathlib import Path

DASH_CLI = Path("tests/dash_cli.py")
BASE_URL  = "http://localhost:5000"

def cli(*args, timeout=180):
    """Run one dash_cli.py command synchronously. Raises on failure."""
    cmd = [sys.executable, str(DASH_CLI),
           "--base-url", BASE_URL,
           "--timeout", str(timeout)] + list(args)
    result = subprocess.run(cmd, text=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"CLI command failed: {' '.join(args)}")

# ---- configure ----
cli("init-config")
cli("apply-override", "my_experiment")       # optional
cli("set", "default_consumer_config.seed", "42")
cli("set", "wandb.run_name", "my_run_seed42")
cli("set", "wandb.group",    "my-experiment")

# ---- start (order matters) ----
cli("start-wandb", timeout=120)              # W&B first
cli("produce-all")
time.sleep(15)
cli("start-automatic-attacks")
time.sleep(15)
cli("consume-all")
time.sleep(10)
cli("start-federated-learning")              # omit if no FL

# ---- run ----
time.sleep(3600)                             # your experiment duration

# ---- stop ----
cli("shutdown")                              # safe sequential teardown
```

Key rules to follow:

- Always call `init-config` at the start of each run to reset the state to a
  known baseline before applying overrides.
- Call `start-wandb` **before** `produce-all`. The wandber container must be
  listening before metrics start flowing or the first observations will be lost.
- Leave 15 s between `produce-all` and `start-automatic-attacks`, and another
  15 s before `consume-all`. This lets the data stream stabilise before
  consumers start classifying.
- Do **not** call `create-vehicles` or `delete-vehicles` inside the run loop.
  Vehicle containers are expensive to create and their internal state (Kafka
  offsets, buffer contents) resets on recreation. Create them once before the
  loop and delete them manually when the full batch is done.
- `shutdown` is the correct way to end a run. It performs a sequential,
  ordered teardown inside the dashboard container and returns only when all
  services have acknowledged the stop command.
