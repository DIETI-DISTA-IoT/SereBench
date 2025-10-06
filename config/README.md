# Configuration System

This directory contains configuration files for the OpenFAIR system, implementing a secure and maintainable approach to configuration management.

## Overview

The new configuration system replaces unsafe command-line argument injection with:
1. **Environment Variables** - For basic, non-changing parameters
2. **YAML Configuration Files** - For complex, vehicle-specific settings
3. **HTTP API** - For dynamic configuration and control
4. **Validation** - Input validation and type safety
5. **Health Checks** - Built-in monitoring endpoints

## Directory Structure

```
config/
├── producers/           # Producer-specific configurations
│   ├── angela.yaml     # Configuration for producer-angela
│   ├── bob.yaml        # Configuration for producer-bob
│   └── ...             # Additional vehicle configurations
└── README.md           # This file
```

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
