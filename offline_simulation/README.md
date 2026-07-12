# SereBench — Offline Simulation (no Docker, no Kafka)

A **monolithic, single-process** re-implementation of the whole SereBench
federated-learning benchmark. It runs the exact same experiment the Docker
Compose cluster runs — data producing, per-vehicle anomaly-detection training,
adversarial robustness evaluation (Gaussian sigma-grid **and** HopSkipJump),
federated aggregation and Weights & Biases logging — but everything lives inside
**one Python process**. There is no Kafka broker, no Docker daemon, no HTTP
control plane, no network at all.

One entry point. Press a button (or run one command), a Python thread starts.
Press another, it stops. That's it.

```bash
# from the SereBench repo root
pip install -r offline_simulation/requirements.txt

# Install pytorch is appart, so you can install cpu-only version if you want:
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# headless: run the full experiment for 3 minutes, log to W&B offline
python -m offline_simulation.main --duration 180 --wandb-mode offline

# or bring up the local dashboard and click the buttons (still no Docker!)
python -m offline_simulation.main --dashboard
#   -> open http://127.0.0.1:8000

# use specific host/port overrides, e.g:
python -m offline_simulation.main --dashboard --dashboard-host 192.168.183.146 --dashboard-port 5000
```

---

## Why this exists

The full platform (`docker-compose.yml`) wires ~10 services together over Kafka
and a Flask/Docker control plane. That is great for a realistic distributed
deployment and painful for quick experimentation, debugging an aggregation
strategy, or running on a laptop/CI box. This package collapses the whole thing
into threads + in-process queues so you can `python -m offline_simulation.main`
and iterate.

**It reuses the real algorithms, verbatim.** The numerically-important cores are
*vendored unchanged* from the sub-repositories (see
[Fidelity](#fidelity--how-faithful-is-it)), so results track the Dockerised
platform; only the transport (Kafka → in-process bus) and orchestration (Docker
+ HTTP → direct method calls) are reimplemented.

---

## How it maps onto the real platform

| Real platform (Docker + Kafka)                              | Offline simulation                                   |
|-------------------------------------------------------------|------------------------------------------------------|
| Kafka topics (`{v}_anomalies`, `{v}_weights`, `global_weights`, `{v}_statistics`, …) | `message_bus.MessageBus` — in-process topic pub/sub with the same subscription rules (`^` prefix = regex) |
| `Train_IoT_data_producer` container (per vehicle)           | `producer.ProducerNode` (2 threads/vehicle)          |
| `Train_IoT_local_anomaly_detection` container (per vehicle) | `consumer.ConsumerNode` (consume/train/push/pull threads) |
| `of-fedlearningmanager` container                           | `fl_manager.FLManagerNode`                           |
| `Wandber` container                                         | `wandb_logger.WandbNode`                             |
| Dashboard + `ContainerManager` (Docker socket, HTTP)        | `orchestrator.Orchestrator` (direct calls)           |
| Dashboard web UI                                            | `dashboard.py` + `templates/index.html` (optional)   |
| Hydra `config/` + `override=` mechanism                     | `config_loader.load_config` (same files, same merge) |
| HTTP `/configure` payloads built by the managers            | `config_builders.py` (same dictionaries)             |

The data flow is unchanged:

```
ProducerNode ──{v}_anomalies / {v}_eval_anomalies / {v}_normal_data──▶ ConsumerNode
ConsumerNode ──{v}_weights──▶ FLManagerNode ──global_weights──▶ ConsumerNode (pull)
ConsumerNode ──{v}_statistics──▶ WandbNode ──▶ wandb.log
```

---

## Configuration — identical to the dashboard

The simulation reads the repo's single source of truth, `../config/`, exactly
the way `of-dashboard/app.py` does: it loads `config/default.yaml`, then merges
`config/overrides/<name>.yaml` when an override is selected, then applies any
command-line `key=value` overrides (Hydra-style).

```bash
# use an override file (config/overrides/fedyogi.yaml)
python -m offline_simulation.main override=fedyogi

# ad-hoc overrides on top (dotted keys, just like Hydra)
python -m offline_simulation.main \
    federated_learning.aggregation_strategy=fedmedian \
    anomaly_detection.model_type=cnn \
    default_vehicle_config.time_emulation=true
```

Everything the platform understands works here: the four aggregation strategies
(`fedavg` / `fedyogi` / `fedmedian` / `fedprox`), the three architectures
(`mlp` / `cnn` / `resnet`, all sharing the 2-D manifold bottleneck),
adversarial training, the per-vehicle noise gradient (`Mp_std`/`Bp_std`
0.0 → 0.7 → 1.4 → 2.1 across angela/bob/claude/daniel), the decoupled
adversarial eval stream (`eval_Mp_std`/`eval_Bp_std`), HopSkipJump settings,
the fixed sigma-grid, FedProx `μ`, FedYogi moments, seeds, and so on.

### Useful CLI flags

| Flag | Meaning |
|------|---------|
| `--duration N` | headless run length in seconds (default 120) |
| `--wandb-mode {online,offline,disabled}` | force the W&B mode (default: derive from `config.wandb.online`; auto-falls back to `offline` if online is requested but no `WANDB_API_KEY` is set) |
| `--no-attacks` / `--no-fl` / `--no-wandb` | skip a component |
| `--seed N` | set the producer & consumer RNG seed |
| `--dashboard` | launch the local UI instead of a headless run |
| `--dashboard-host/--dashboard-port` | where to serve it (default `127.0.0.1:8000`) |
| `override=<name> key=value ...` | Hydra-style config overrides (positional) |

---

## The dashboard (optional, still no Docker)

`--dashboard` serves a small control panel whose buttons map one-to-one onto the
real dashboard's controls — **Start full experiment**, produce/consume all,
start/stop federated learning, start/stop W&B, automatic & preconfigured
attacks, per-vehicle manual attack/heal, and a live fleet-status table (each
vehicle's HEALTHY/INFECTED state, epoch counter, records processed, and the FL
round count). Each button just calls an `Orchestrator` method in a background
thread. No container is ever created.

---

## Replicating the paper campaign — `experiments.py`

`offline_simulation/experiments.py` is the in-process twin of the Dockerised
`tests/experiments.py` (+ `_cnn` / `_resnet` / `_all`). Those runners drive the
container cluster over the dashboard's HTTP control plane (via `tests/dash_cli.py`);
this one drives an in-process `Orchestrator` instead — **same experiment matrix,
same config overrides, same seeds, same startup order and cadence, no Docker and
no Kafka.** It's the way to reproduce the full benchmark campaign on a laptop or
CI box.

```bash
pip install -r offline_simulation/requirements.txt
pip3 install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only

# All 76 experiments (7 canonical + 32 ET4 + 37 ET5, MLP), 3 seeds each, W&B offline
python -m offline_simulation.experiments

# Quick smoke: one experiment, one seed, 60 s, fast warmup
python -m offline_simulation.experiments \
    --experiments 1 --seeds 42 --run-duration 60 --settle 0.2

# Every architecture, FL block only (FedAvg/FedProx/FedYogi/FedMedian)
python -m offline_simulation.experiments --arch all --experiments 4 5 6 7

# ET4 network-free-rider block (ids 8..39) across every architecture
python -m offline_simulation.experiments --arch all --experiments $(seq 8 39)

# ET5 FL-flow-stress block (ids 40..76): FL-algorithm tolerance to a lossy/slow
# FL flow, identical fleet, no free rider
python -m offline_simulation.experiments --experiments $(seq 40 76)

# Continue past failures instead of aborting the batch
python -m offline_simulation.experiments --skip-on-error
```

### The experiment matrix (identical to `tests/experiments.py`)

| # | Name | FL | Config override(s) |
|---|------|----|--------------------|
| 1 | `et2-noadv-nofl`        | – | *(defaults)* |
| 2 | `et2-adv-nofl`          | – | `exp_advtraining` |
| 3 | `et3-freerider-nofl`    | – | `exp_et3_freerider` |
| 4 | `et3-freerider-fedavg`  | ✓ | `exp_et3_freerider` |
| 5 | `et3-freerider-fedprox` | ✓ | `exp_et3_freerider` + `fedprox` |
| 6 | `et3-freerider-fedyogi` | ✓ | `exp_et3_freerider` + `fedyogi` |
| 7 | `et3-freerider-fedmedian`| ✓ | `exp_et3_freerider` + `fedmedian` |

**Block C — ET4 (does FL help a *network*-impaired free-rider?), ids 8..39.**
Angela is the only vehicle with a degraded uplink — packet loss and/or added
delay+jitter on *both* her producer→consumer telemetry pipeline and her
consumer→FL weights upload (`offline_simulation/packet_loss.py` +
`network_delay.py`); bob/claude/daniel keep a pristine network. Everyone trains
clean and shares the ET3 decoupled eval (HSJA clean anchors + fixed sigma grid),
so angela's impairment is the only moving part besides FL. The block is the
cross-product of 8 impairment levels × 4 FL modes (no-FL baseline + FedAvg +
FedMedian + FedYogi):

| ids | Level | Angela's impairment (peers = 0/0/0) | Config override |
|-----|-------|-------------------------------------|-----------------|
| 8..11  | `et4-loss20-{nofl,fedavg,fedmedian,fedyogi}`   | packet loss 0.2                     | `exp_et4_angela_loss20` |
| 12..15 | `et4-loss40-*`                                 | packet loss 0.4                     | `exp_et4_angela_loss40` |
| 16..19 | `et4-loss60-*`                                 | packet loss 0.6                     | `exp_et4_angela_loss60` |
| 20..23 | `et4-delay100-*`                               | delay 100 ms / jitter 25 ms         | `exp_et4_angela_delay100` |
| 24..27 | `et4-delay250-*`                               | delay 250 ms / jitter 60 ms         | `exp_et4_angela_delay250` |
| 28..31 | `et4-delay500-*`                               | delay 500 ms / jitter 125 ms        | `exp_et4_angela_delay500` |
| 32..35 | `et4-combo-lo-*`                               | loss 0.2 + delay 100 ms / 25 ms     | `exp_et4_angela_combo_lo` |
| 36..39 | `et4-combo-hi-*`                               | loss 0.4 + delay 250 ms / 60 ms     | `exp_et4_angela_combo_hi` |

The `fedavg` / `fedmedian` / `fedyogi` modes append the matching aggregation
override on top of the level file (the `nofl` mode applies none); the FL
coordinator's `global_weights` downlink is held pristine in every ET4 cell so the
FL benefit under test is never itself sabotaged — only angela's *uplink* is
impaired.

**Block D — ET5 (which FL strategy best tolerates a degraded FL *flow*?), ids 40..76.**
ET5 asks the complement of ET4. There is **no free rider**: all four vehicles are
identical (clean config, no adversarial training) and the impairment is applied
**uniformly to the whole fleet _and_ to the FL coordinator**, so the question is
which aggregation strategy degrades most gracefully as the FL flow itself gets
lossy/slow. The **only** degraded path is the FL flow — the consumer→FL-manager
weights *uplink* (`default_consumer_config.*`) **and** the FL-manager→consumers
`global_weights` *downlink* (`federated_learning.*`); the producer→consumer
telemetry pipeline is held **pristine** (0 loss / 0 delay), so local training data
is never starved and only the aggregation transport is under stress. Everyone
shares the ET3/ET4 decoupled eval (HSJA clean anchors + fixed sigma-grid), so ET5
numbers line up with the rest of the campaign.

Two **calibrated** design choices make the stress observable (see
`config/README.md`, *ET5 calibration*, and the header of any
`config/overrides/exp_et5_flow_*.yaml`):

1. **Fast FL cadence** — push/pull every **15 s**, aggregate every **5 s**, instead
   of the platform default 120 s / 20 s. At the default cadence a 10-min run yields
   only a handful of weight pushes, so packet loss is statistically coarse and a
   sub-second delay is a no-op against a 120 s push interval. The fast cadence
   exercises the flow **~40×/run**.
2. **Delay scaled to that interval** — 1.5 s / 4 s / 8 s, not the few-hundred-ms of
   ET4. A few hundred ms is meaningful on ET4's high-frequency *telemetry* path but
   negligible on the low-frequency FL *weights* path.

The block is a network-invariant no-FL floor (id 40) plus the cross-product of 9
FL-flow conditions × 4 strategies (FedAvg / FedMedian / FedYogi / FedProx):

| ids | Level | FL-flow impairment (uplink + downlink, whole fleet) | Config override |
|-----|-------|-----------------------------------------------------|-----------------|
| 40      | `et5-flow-nofl-floor`                                     | *(no FL — network-invariant floor)* | `exp_et5_flow_clean` |
| 41..44  | `et5-clean-{fedavg,fedmedian,fedyogi,fedprox}`            | none (pristine FL flow, same fast cadence) | `exp_et5_flow_clean` |
| 45..48  | `et5-loss20-*`                                            | packet loss 0.2                     | `exp_et5_flow_loss20` |
| 49..52  | `et5-loss40-*`                                            | packet loss 0.4                     | `exp_et5_flow_loss40` |
| 53..56  | `et5-loss60-*`                                            | packet loss 0.6                     | `exp_et5_flow_loss60` |
| 57..60  | `et5-delay1500-*`                                         | delay 1.5 s / jitter 375 ms         | `exp_et5_flow_delay1500` |
| 61..64  | `et5-delay4000-*`                                         | delay 4 s / jitter 1 s              | `exp_et5_flow_delay4000` |
| 65..68  | `et5-delay8000-*`                                         | delay 8 s / jitter 2 s              | `exp_et5_flow_delay8000` |
| 69..72  | `et5-combo-lo-*`                                          | loss 0.2 + delay 1.5 s / 375 ms     | `exp_et5_flow_combo_lo` |
| 73..76  | `et5-combo-hi-*`                                          | loss 0.6 + delay 8 s / 2 s          | `exp_et5_flow_combo_hi` |

Each FL cell applies the aggregation-strategy profile **first** and the level file
**last** (`override=(strategy, level)`), so the level file's fast cadence and
FL-flow network settings win while the strategy file still supplies
`aggregation_strategy` and its hyper-parameters (this is the reverse of ET4's
`(level, strategy)` order — fine, because ET4's level files never touch the
cadence). The no-FL floor is a single cell, not one per level, because without FL
the FL-flow impairment has no effect. Read the whole block off W&B by grouping on
run name: for each condition, the four strategy curves show which one holds
robustness/accuracy up best as the flow degrades.

`--arch {mlp,cnn,resnet,all}` merges the `cnn`/`resnet` override on top (mlp is
the default) and suffixes the W&B run/group names (`-cnn`, `-resnet`), exactly as
the per-architecture Dockerised runners do.

### How it maps onto `tests/experiments.py`

| Dockerised (`tests/experiments.py` → `dash_cli.py` → HTTP) | Offline (`experiments.py` → `Orchestrator` method) |
|-------------------------------------------------------------|----------------------------------------------------|
| `init-config` + repeated `apply-override` + `set …`         | `build_config()` (OmegaConf load + deep-merge + dotlist) |
| `start-wandb`                                                | `orchestrator.start_wandb()` |
| `produce-all`                                               | `orchestrator.produce_all()` |
| `start-automatic-attacks`                                  | `orchestrator.start_automatic_attacks()` |
| `consume-all`                                               | `orchestrator.consume_all()` |
| `start-federated-learning`                                 | `orchestrator.start_federated_learning()` |
| `shutdown`                                                  | `orchestrator.shutdown()` |

The startup order and inter-step warmup (produce → wait → attacks → wait →
consume → wait → FL) is preserved; the warmup delays are scaled by `--settle`
(1.0 = platform timing). Unlike the Dockerised runner there are no vehicle
containers to `create-vehicles` / `delete-vehicles` — each run builds a fresh
in-process cluster from config and tears it down.

### Extra flags (beyond the shared matrix)

| Flag | Meaning |
|------|---------|
| `--arch mlp\|cnn\|resnet\|all` | architecture(s) to sweep (default `mlp`) |
| `--run-duration N` | seconds per run (default 600) |
| `--wandb-mode {online,offline,disabled}` | W&B mode for every run (default `offline`) |
| `--settle F` | scale the startup warmup delays (default 1.0) |
| `--no-attacks` / `--no-wandb` | skip a component across all runs |
| `--skip-on-error` | log a failed run and continue the batch |

---

## Fidelity — how faithful is it?

**Vendored verbatim** (byte-identical copies of the sereBench-branch sources —
do not edit these, edit upstream and re-vendor):

- `simulator.py`   ← `of-core/train_simulator.py` (the physically-grounded 40-feature railway simulator + `EventType`)
- `models.py`      ← `of-fedlearningmanager/modules.py` (MLP / CNN1D / TabResNet + `build_model`, state-dict-compatible with the consumer)
- `hopskipjump.py` ← `Train_IoT_local_anomaly_detection/hopskipjump.py` (batched HSJA decision-based attack)
- `aggregation.py` ← `of-fedlearningmanager/aggregation.py` (FedAvg / FedYogi / FedMedian / FedProx)

**Reimplemented faithfully** (same logic, Kafka/HTTP/Docker stripped out):
`producer.py`, `consumer.py` (+ `brain.py`, `buffers.py`), `fl_manager.py`,
`wandb_logger.py`, `orchestrator.py`. The training loop, buffer routing, metric
formulas, benchmark cadence, HSJA threat model, FL buffering/aggregation and
W&B plotting all mirror the originals line for line, including quirks preserved
on purpose (e.g. the consumer `Buffer`'s independent feature/label sampling, and
`anomaly_detection.learning_rate` overriding the FL server learning rate when
the FL payload is assembled).

### Producer rate & the `time_emulation` knob

The producer honours `default_vehicle_config.time_emulation`. With it **on**
(the config default) inter-arrival times follow the same lognormal model as the
platform, so producers and consumers stay balanced. With it **off** producers
generate flat-out — handy to warm buffers fast, but they then vastly outpace the
consumers; the in-process bus caps each telemetry queue and drops the oldest
messages under lag (exactly the guarantee Kafka retention gives), so memory
stays bounded and the consumer simply trains on the freshest data.

### Intentionally out of scope

- **`SecurityManager`** (the health-probe intrusion detector in Wandber): on the
  sereBench branch the producer no longer emits the `{v}_HEALTH` stream it feeds
  on, so it receives no data in the current pipeline and is omitted here. The
  cyber-attack dynamics it used to react to are fully present — attacks flip the
  producer's ANOMALY→ATTACK generation and the *consumer's* model performs the
  detection + reactive mitigation (with the mitigation-time measured in-process),
  which is what the FL benchmark actually measures.
- **The UDP flood `attack.py`** and network probes (RTT/CPU/MEM): those are
  real-network artefacts with no meaning in a single process. "Under attack" is
  modelled where it matters — the data distribution shift — not as real packets.

---

## Module map

```
offline_simulation/
├── main.py            entry point (CLI): load config → orchestrator → run/shutdown
├── orchestrator.py    in-process control plane (replaces dashboard + ContainerManager)
├── message_bus.py     in-process Kafka shim (topic pub/sub, bounded queues)
├── config_loader.py   OmegaConf load + override merge (same as the dashboard)
├── config_builders.py per-node config payloads (ports the of-core managers)
├── producer.py        ProducerNode  — data generation threads
├── consumer.py        ConsumerNode  — training / robustness / HSJA / FL push-pull
├── brain.py           per-vehicle model + optimizer + train step
├── buffers.py         reservoir + FIFO buffers
├── fl_manager.py      FLManagerNode — federated aggregation
├── wandb_logger.py    WandbNode     — statistics → wandb.log (+ plots)
├── dashboard.py       optional Flask control panel (no Docker)
├── templates/index.html
├── simulator.py  models.py  hopskipjump.py  aggregation.py   ← vendored cores
└── requirements.txt
```
