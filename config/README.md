# Configuration System

This directory contains configuration files for the OpenFAIR system, implementing a secure and maintainable approach to configuration management.

## Overview

The new configuration system replaces unsafe command-line argument injection with:
1. **Environment Variables** - For basic, non-changing parameters
2. **YAML Configuration File** - For complex, vehicle-specific settings
3. **HTTP API** - For dynamic configuration and control
4. **Validation** - Input validation and type safety
5. **Health Checks** - Built-in monitoring endpoints


## Configuration Methods

### 1. Environment Variables

Basic configuration via environment variables:

```bash
# Required
export VEHICLE_NAME=angela

# Optional (with defaults)
export KAFKA_BROKER=kafka:9092
export LOGGING_LEVEL=INFO
export MANAGER_PORT=5000
export MODE=OF
```

### 2. YAML Configuration Files

Detailed configuration via YAML files:

```yaml
# config/producers/angela.yaml
vehicle:
  name: angela
  flotta: ETR700

data_generation:
  mu_anomalies: 157
  mu_normal: 115
  alpha: 0.2
  beta: 1.9
  time_emulation: false
  anomaly_classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  diagnostics_classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

probe:
  frequency_seconds: 2
  timeout: 5
  host: "www.google.com"
  metrics: [RTT, INBOUND, OUTBOUND, CPU, MEM]

attack:
  target_ip: "172.18.0.4"
  target_port: 80
  duration: 0
  packet_size: 1024
  delay: 0.001
  bot_port: 5002

system:
  mode: "OF"
  logging_level: "INFO"
  manager_port: 5000
```

### 3. HTTP API

Dynamic configuration and control via HTTP API:

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/configure` | Configure the producer with new settings |
| `POST` | `/start` | Start the producer with current configuration |
| `POST` | `/stop` | Stop the producer |
| `GET` | `/status` | Get current status and statistics |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/config` | Get current configuration |
| `PUT` | `/config` | Update specific configuration parameters |

#### Example Usage

```python
import requests

# Configure producer
config_data = {
    'vehicle_name': 'angela',
    'kafka_broker': 'kafka:9092',
    'mu_anomalies': 157,
    'mu_normal': 115,
    'alpha': 0.2,
    'beta': 1.9,
    'time_emulation': False,
    'probe_frequency_seconds': 2,
    'ping_thread_timeout': 5,
    'ping_host': 'www.google.com',
    'probe_metrics': ['RTT', 'INBOUND', 'OUTBOUND', 'CPU', 'MEM'],
    'anomaly_classes': list(range(0, 19)),
    'diagnostics_classes': list(range(0, 15))
}

response = requests.post('http://localhost:5000/configure', json=config_data)
print(response.json())

# Start producer
response = requests.post('http://localhost:5000/start')
print(response.json())

# Check status
response = requests.get('http://localhost:5000/status')
status = response.json()
print(f"Running: {status['running']}")
print(f"Records produced: {status['records_produced']}")

# Update configuration
updates = {'mu_anomalies': 200, 'probe_frequency_seconds': 3}
response = requests.put('http://localhost:5000/config', json=updates)
print(response.json())

# Stop producer
response = requests.post('http://localhost:5000/stop')
print(response.json())
```

#### Using the API Client

```python
from test_api_client import ProducerAPIClient

client = ProducerAPIClient("http://localhost:5000")

# Configure and start
client.configure(config_data)
client.start()

# Monitor
status = client.status()
print(f"Status: {status}")

# Update and restart
client.update_config({'mu_anomalies': 300})
client.stop()
client.start()

# Cleanup
client.stop()
```

## Configuration Priority

Configuration is loaded in the following order (later sources override earlier ones):

1. **Environment Variables** - Basic settings
2. **YAML Configuration File** - Detailed settings
3. **HTTP API Updates** - Dynamic runtime changes

## Validation

All configuration parameters are validated:

- **Required fields**: `vehicle_name`, `kafka_broker`
- **Numeric ranges**: `mu_anomalies` (0-1000), `alpha` (0-10), etc.
- **Port numbers**: Valid port ranges (1-65535)
- **Data types**: Proper type conversion and validation

## Security Improvements

### Before (Unsafe)
```python
# Command injection vulnerability
command = f"python produce.py --kafka_broker={user_input} --mu_anomalies={user_input}"
subprocess.run(command, shell=True)  # DANGEROUS!
```

### After (Safe)
```python
# HTTP API with validation
config_data = {
    'kafka_broker': user_input,  # Validated
    'mu_anomalies': user_input   # Validated
}
requests.post('/configure', json=config_data)  # SAFE!
```

## Testing

### Test Configuration Loading
```bash
python test_producer_config.py
```

### Test HTTP API
```bash
python test_api_client.py
```

### Test with Docker
```bash
# Start containers
docker-compose up -d

# Test API endpoints
curl http://localhost:5000/health
curl http://localhost:5000/status
```

## Migration Guide

### From Command-Line Arguments

**Old approach:**
```bash
python produce.py \
  --kafka_broker=kafka:9092 \
  --mu_anomalies=157 \
  --mu_normal=115 \
  --alpha=0.2 \
  --beta=1.9 \
  --anomaly_classes=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 \
  --diagnostics_classes=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
  --time_emulation \
  --ping_thread_timeout=5 \
  --ping_host=www.google.com \
  --probe_frequency_seconds=2 \
  --probe_metrics=RTT,INBOUND,OUTBOUND,CPU,MEM \
  --mode=OF \
  --manager_port=5000 \
  --target_ip=172.18.0.4 \
  --target_port=80 \
  --duration=0 \
  --packet_size=1024 \
  --delay=0.001
```

**New approach:**
```bash
# Set environment variables
export VEHICLE_NAME=angela
export KAFKA_BROKER=kafka:9092

# Use YAML configuration
python produce.py  # Loads from config/producers/angela.yaml

# Or use HTTP API
curl -X POST http://localhost:5000/configure \
  -H "Content-Type: application/json" \
  -d @config/producers/angela.json
```

## Benefits

1. **Security**: No command injection vulnerabilities
2. **Maintainability**: Clear separation of configuration and code
3. **Flexibility**: Runtime configuration changes
4. **Validation**: Type safety and parameter validation
5. **Monitoring**: Built-in health checks and status endpoints
6. **Scalability**: Easy to manage multiple producers
7. **Debugging**: Better error messages and logging

---

## Network-impairment knobs (packet loss + delay)

Three flows in the platform can be independently degraded to study robustness to
a lossy/slow network. Each flow reads three keys — `packet_loss_rate`
(Bernoulli drop probability, `[0,1]`), `delay_mean_ms` and `jitter_std_ms`
(added latency `max(0, delay_mean_ms + N(0, jitter_std_ms))`). Impairment is
applied **client-side, on the sending node**, because neither Kafka nor the
offline in-process bus ever drops or delays an accepted write. The same keys
behave identically in the Dockerised platform (`OpenFAIR/packet_loss.py` +
`network_delay.py`) and in `offline_simulation/` (`packet_loss.py` +
`network_delay.py`).

| Flow | Config section | Applies to | Excluded (never impaired) |
|------|----------------|------------|----------------------------|
| producer → consumer **telemetry** | `default_vehicle_config.*` (+ per-vehicle override) | `{v}_anomalies`, `{v}_eval_anomalies`, `{v}_normal_data` | health probes, anything W&B-bound |
| consumer → FL-manager **weights uplink** | `default_consumer_config.*` (+ per-vehicle override) | `{v}_weights` | `{v}_statistics` (W&B-bound) |
| FL-manager → consumers **global-weights downlink** | `federated_learning.*` | `global_weights` | `global_metrics` (W&B-bound) |

Because a per-vehicle override under `vehicles:` merges into **both** the
producer and the consumer config, degrading **only** the FL flow (and leaving
telemetry pristine) is done by setting the *section* defaults rather than
per-vehicle keys: zero out `default_vehicle_config.{packet_loss_rate,
delay_mean_ms, jitter_std_ms}`, set the impairment on
`default_consumer_config.*` (uplink) and `federated_learning.*` (downlink), and
leave the per-vehicle blocks carrying only `Mp_std`/`Bp_std`. This is exactly
what the ET5 override files do.

## Experiment overrides & the ET5 FL-flow-stress block

`config/overrides/*.yaml` are partial files merged on top of `default.yaml`
(Hydra/OmegaConf semantics: maps deep-merge, lists — like `vehicles:` — are
replaced wholesale). The experiment campaign is defined in
`tests/experiments.py` (Dockerised) and `offline_simulation/experiments.py`
(in-process twin), which merge these files in a fixed order. See either file's
docstring for the full matrix (Block A/B canonical, Block C = ET4, Block D = ET5).

The **ET5 block** (`config/overrides/exp_et5_flow_*.yaml`) answers *"which
aggregation strategy best tolerates a degraded FL flow?"*. Unlike ET4 (a single
network free-rider), ET5 has **no free rider**: all four vehicles are identical
(clean config, no adversarial training) and the impairment is applied
**uniformly to the whole fleet and to the FL coordinator**, on the FL flow only
(weights uplink + global-weights downlink), with telemetry held pristine. Each
cell crosses a network condition with an aggregation strategy
(FedAvg/FedMedian/FedYogi/FedProx); the runner applies the strategy profile
first and the `exp_et5_flow_*` level file last, so the level's settings win while
the strategy still supplies `aggregation_strategy` and its hyper-parameters.

### ET5 calibration

The ET5 levels were **calibrated in `offline_simulation`** (no Docker needed) so
the impairment produces a smooth, meaningful signal rather than noise. Two
findings drove the numbers, both a consequence of the FL flow being **low
frequency** (one weight push per vehicle per push-interval) unlike the
high-frequency telemetry path:

1. **Cadence.** At the platform default (`weights_push_freq_seconds: 120`,
   `aggregation_interval_secs: 20`) a 10-minute run produces only ~5 weight
   pushes per vehicle — a calibration run of 90 s produced **zero** pushes and
   **zero** FL rounds. Packet loss is then statistically coarse (dropping 20 % of
   5 pushes is 1 push) and any sub-second delay is negligible against a 120 s
   interval. ET5 therefore runs a **fast FL cadence — push/pull every 15 s,
   aggregate every 5 s** — which exercises the flow ~40×/run; a 150 s calibration
   run then showed ~10 pushes/vehicle and clean, proportional drops
   (`weights->FL = 3/10, 4/10, …` at loss 0.4; `rounds=5 packet_loss=2/5` on the
   downlink). The effective aggregation cadence is gated by the push interval
   (weights only refresh every 15 s), so 15 s is the binding knob.
2. **Delay scale.** Because pushes are 15 s apart, delay must be a **meaningful
   fraction of that interval** to change aggregation staleness/ordering. A few
   hundred ms (ET4's telemetry-path magnitudes) is a no-op here, so ET5 uses
   **1.5 s / 4 s / 8 s** (≈ 0.1× / 0.27× / 0.53× of the push interval, with jitter
   at 25 % of the mean). At 8 s, weights routinely land more than half an interval
   late and jitter reorders them across rounds.

A calibration run also confirmed the flow is cleanly isolated
(`packet_loss[telemetry] = 0/37256`, `delay[telemetry] ~0 ms` while the weights
path is degraded).

3. **Startup order (offline runner).** A consumer only forms a training batch
   when its NORMAL, ANOMALY **and** ATTACK buffers are all full, and the
   producer's anomaly thread emits ATTACK (not ANOMALY) whenever the vehicle is
   INFECTED (`producer.py`). So the consumers must warm their ANOMALY buffers on
   a still-healthy stream **before the first attack**. Calibration exposed that
   the offline campaign runner was starting automatic attacks *before* the
   consumers subscribed (`produce -> attacks -> consume`): because the in-process
   `MessageBus` does not replay backlog to a late subscriber, the anomaly stream
   had already flipped to ATTACK by the time a consumer was listening, so the
   ANOMALY buffer never filled and every model sat at `epoch 0` — verified for a
   full 300 s run on both canonical ET1 and an ET5 cell (the FL flow itself
   logged 20 clean rounds; the models simply never trained). The fix is in
   `offline_simulation/experiments.py`: it now starts consumers (and FL) **before**
   automatic attacks, matching `orchestrator.start_experiment`'s documented
   "attacks last" order — with which a normal realtime-rate run trains fine
   (epochs ~20, anomalies flowing, vehicles healing). This is a **campaign-wide**
   fix (ET1–ET5 all benefit), not ET5-specific, and it is the ONE point where the
   offline runner deliberately diverges from the Dockerised `tests/experiments.py`
   (which keeps `produce -> attacks -> consume`): on Docker a Kafka consumer with
   `auto_offset_reset=earliest` replays the pre-attack anomaly backlog, so that
   order is safe there. (`--no-attacks` is a separate trap — it leaves the ATTACK
   buffer empty and also stalls training; keep automatic attacks on.)

| Level file | uplink+downlink loss | uplink+downlink delay / jitter |
|------------|----------------------|--------------------------------|
| `exp_et5_flow_clean`     | 0.0 | 0 / 0 (reference, same fast cadence) |
| `exp_et5_flow_loss20`    | 0.2 | 0 / 0 |
| `exp_et5_flow_loss40`    | 0.4 | 0 / 0 |
| `exp_et5_flow_loss60`    | 0.6 | 0 / 0 |
| `exp_et5_flow_delay1500` | 0.0 | 1500 ms / 375 ms |
| `exp_et5_flow_delay4000` | 0.0 | 4000 ms / 1000 ms |
| `exp_et5_flow_delay8000` | 0.0 | 8000 ms / 2000 ms |
| `exp_et5_flow_combo_lo`  | 0.2 | 1500 ms / 375 ms |
| `exp_et5_flow_combo_hi`  | 0.6 | 8000 ms / 2000 ms |

Run the block (offline, no Docker) with:

```bash
python -m offline_simulation.experiments --experiments $(seq 40 76)
```
