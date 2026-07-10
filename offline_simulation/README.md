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
