# Notes on Experiment Template 5 (ET5): FL-Flow Network Stress Analysis

This document provides a comprehensive report on the implementation of packet loss and delay simulation in the SereBench platform, along with the rationale, calibration decisions, and configurations for **Experiment Template 5 (ET5)**.

---

## 1. Introduction & Context of ET5

In federated learning (FL) campaigns, network instability can significantly degrade performance. While **Experiment Template 4 (ET4)** focused on a single-vehicle uplink network failure (a "network free-rider") to see if federated learning could rescue it, **Experiment Template 5 (ET5)** poses a different architectural question:

> **Which federated aggregation strategy (FedAvg, FedMedian, FedYogi, FedProx) degrades most gracefully under uniform network stress applied directly to the FL flow itself?**

In ET5, there are **no free riders**:
1. All four vehicles are identical (clean telemetry configurations, no local adversarial training).
2. The producer $\rightarrow$ consumer telemetry pipeline is kept **pristine** (0% loss, 0 ms delay), meaning local training data is never starved.
3. Network impairment is applied **uniformly to the entire fleet and to the FL coordinator** on the FL flows only:
   - **Weights Uplink:** Consumer $\rightarrow$ FL manager (`default_consumer_config.*`)
   - **Global Weights Downlink:** FL manager $\rightarrow$ Consumers (`federated_learning.*`)

This isolates the network stress entirely to the coordination transport layer, allowing an unperturbed evaluation of robust aggregation strategies.

---

## 2. Technical Implementation of Network Impairments

Because message brokers like Apache Kafka and the offline in-process `MessageBus` do not inherently drop or delay messages after accepting a write, network impairments must be injected **client-side on the sending node**.

The platform provides two primary simulation utilities, implemented symmetrically across both the **Dockerised platform** (`OpenFAIR/`) and the **offline simulation environment** (`offline_simulation/`).

### A. Packet Loss Simulation (`packet_loss.py`)
Packet loss is modeled as a Bernoulli trial using a configured packet drop probability $p \in [0, 1]$ (`packet_loss_rate`).
- **Mechanism:** Before handing a message to the broker (or message bus), a random uniform draw $r \sim \mathcal{U}(0, 1)$ is performed. If $r < p$, the message is silently discarded, incrementing the simulator's internal drop counters.
- **Exclusions:** To ensure telemetry monitoring and logging stay operational, packet loss is bypassed for anything that is WandB-bound (e.g., `{v}_statistics`, `global_metrics`).
- **Symmetric Files:**
  - `OpenFAIR/packet_loss.py`
  - `offline_simulation/packet_loss.py`

### B. Network Delay & Jitter Simulation (`network_delay.py`)
Latency and jitter are simulated by deferring message delivery using an asynchronous scheduler.
- **Latent Draw Formula:**
  $$\text{delay\_ms} = \max\left(0, \text{delay\_mean\_ms} + \mathcal{N}(0, \text{jitter\_std\_ms}^2)\right)$$
- **Mechanism:**
  - When the delay is 0, the message is delivered inline on the caller's thread to avoid overhead.
  - When delay is active, a daemon worker thread is lazily spawned.
  - The worker maintains a time-ordered heap (`_heap`) of scheduled sends.
  - Messages are pushed to the heap and executed once their monotonic delivery timestamp is reached.
  - Under high jitter, messages can be reordered, mimicking realistic packet arrival in unstable IP networks.
- **Symmetric Files:**
  - `OpenFAIR/network_delay.py`
  - `offline_simulation/network_delay.py`

---

## 3. Configuration & Flow Isolation

Three independent flows can be degraded using the network-impairment knobs configured in YAML files (e.g., `config/default.yaml` or overrides):

| Flow | Configuration Section | Affects |
| :--- | :--- | :--- |
| **Telemetry** | `default_vehicle_config.*` | `{v}_anomalies`, `{v}_eval_anomalies`, `{v}_normal_data` |
| **Weights Uplink** | `default_consumer_config.*` | `{v}_weights` |
| **Global Weights Downlink** | `federated_learning.*` | `global_weights` |

### The Per-Vehicle Override Trick
In Hydra/OmegaConf, a per-vehicle override under `vehicles:` merges into **both** its producer and consumer configurations. To degrade *only* the FL flow uniformly across the fleet without touching telemetry, ET5 avoids per-vehicle network blocks. Instead, it:
1. Keeps `default_vehicle_config` at `0.0` loss and delay.
2. Injects the desired loss and delay directly into `default_consumer_config` (uplink) and `federated_learning` (downlink).
3. Leaves the individual vehicles under `vehicles:` Carrying only `Mp_std: 0` and `Bp_std: 0`.

---

## 4. ET5 Rationale, Calibration, and Design Choices

Running network degradation on a low-frequency federated learning flow required deliberate calibration to produce meaningful, smooth evaluation signals rather than binary success/failure noise.

### A. The Cadence Challenge (Fast FL Cadence)
Under the default platform settings, the push frequency is `120 s` and the aggregation interval is `20 s`. In a typical 10-minute experiment run, each vehicle would push its weights only about 5 times:
- Packet loss is statistically coarse (e.g., dropping 20% of 5 pushes means dropping exactly 1 weight update).
- Sub-second delays are mathematically insignificant compared to a 2-minute interval.

**The Solution:** ET5 overrides the defaults to run a **fast FL cadence**:
- **Weights Push Frequency:** `15 s` (instead of 120s)
- **Weights Pull Frequency:** `15 s` (instead of 120s)
- **Aggregation Interval:** `5 s` (instead of 20s)

This fast cadence exercises the FL weights flow ~40 times per 10-minute run, ensuring smooth and statistically significant packet drop rates and delay scenarios.

### B. Scaled Delay Magnitudes
Because pushes occur every 15 seconds, delays must be meaningful fractions of this interval to trigger staleness or round-omissions:
- High-frequency telemetry (ET4) uses small delays (100–500 ms).
- ET5 scales delay levels to **1.5s, 4.0s, and 8.0s** (representing roughly $10\%$, $27\%$, and $53\%$ of the push interval). Jitter is scaled proportionally to $25\%$ of the mean delay. At 8 seconds, weight uploads routinely arrive late and bleed into subsequent aggregation rounds.

### C. Campaign-Wide Startup Order Fix (Offline Simulation)
During calibration of the offline simulation runner, an architectural discrepancy was uncovered:
- **Symptom:** In offline runs with automatic attacks enabled, models would stall at `epoch 0` and never train.
- **Root Cause:** A consumer only forms a training batch when its normal, anomaly, and attack buffers are all warmed up. The anomaly thread emits attacks whenever a vehicle is infected. The offline runner was starting attacks *before* the consumers subscribed (`produce -> attacks -> consume`). Since the in-process `MessageBus` doesn't replay backlog to late subscribers, the consumers missed the pre-attack anomaly stream, meaning their anomaly buffers never filled and training stalled.
- **The Fix:** In `offline_simulation/experiments.py`, the startup order was corrected to start consumers and FL **before** launching automatic attacks. This ensures buffers warm up properly, matching the "attacks last" behavior of the Dockerised orchestrator.

---

## 5. ET5 Config Files & Impairment Matrix

The ET5 block consists of a network-invariant no-FL floor baseline and 9 uniform network impairment levels crossed with 4 aggregation strategies (FedAvg, FedMedian, FedYogi, FedProx), yielding 37 experiments in total (IDs 40 to 76).

### Impairment Levels Matrix
The 9 levels are defined in `config/overrides/exp_et5_flow_*.yaml` files:

| Level Identifier | YAML File Name | Uplink & Downlink Loss | Uplink & Downlink Delay / Jitter |
| :--- | :--- | :--- | :--- |
| **clean** | `exp_et5_flow_clean.yaml` | `0.0` (0%) | `0 ms` / `0 ms` (Pristine Reference) |
| **loss20** | `exp_et5_flow_loss20.yaml` | `0.2` (20%) | `0 ms` / `0 ms` |
| **loss40** | `exp_et5_flow_loss40.yaml` | `0.4` (40%) | `0 ms` / `0 ms` |
| **loss60** | `exp_et5_flow_loss60.yaml` | `0.6` (60%) | `0 ms` / `0 ms` |
| **delay1500** | `exp_et5_flow_delay1500.yaml` | `0.0` (0%) | `1500 ms` / `375 ms` |
| **delay4000** | `exp_et5_flow_delay4000.yaml` | `0.0` (0%) | `4000 ms` / `1000 ms` |
| **delay8000** | `exp_et5_flow_delay8000.yaml` | `0.0` (0%) | `8000 ms` / `2000 ms` |
| **combo-lo** | `exp_et5_flow_combo_lo.yaml` | `0.2` (20%) | `1500 ms` / `375 ms` |
| **combo-hi** | `exp_et5_flow_combo_hi.yaml` | `0.6` (60%) | `8000 ms` / `2000 ms` |

### Running the Offline ET5 Campaign Sweep

To run the full ET5 experiment block in the offline simulation environment:

```bash
python -m offline_simulation.experiments --experiments $(seq 40 76)
```

---

## 6. Analysis & Visualisation

The results of the ET5 campaign are cached and analyzed in Section 11 of the `serebench_2_plots.ipynb` notebook. Since all four fleet vehicles undergo identical network stress, performance is evaluated using **fleet-wide averages** of metrics (accuracy, precision, and robustness against a fixed Gaussian noise sigma grid, e.g., $\sigma = 1.0$), plotting them side-by-side across all levels to highlight which aggregation strategy is the most resilient to transport-layer degradation.
