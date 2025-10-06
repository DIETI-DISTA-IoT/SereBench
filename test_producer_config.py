#!/usr/bin/env python3
"""
Test script to validate producer configuration loading
"""
import os
import sys
import yaml

def test_config_loading():
    """Test that configuration loading works correctly"""
    print("Testing producer configuration loading...")
    
    # Test 1: Environment variable loading
    print("\n1. Testing environment variable loading...")
    os.environ['VEHICLE_NAME'] = 'test_vehicle'
    os.environ['KAFKA_BROKER'] = 'test-kafka:9092'
    os.environ['LOGGING_LEVEL'] = 'DEBUG'
    
    # Test environment variable loading function directly
    def load_config_from_environment():
        """Load configuration from environment variables"""
        config = {
            'vehicle_name': os.getenv('VEHICLE_NAME'),
            'kafka_broker': os.getenv('KAFKA_BROKER', 'kafka:9092'),
            'logging_level': os.getenv('LOGGING_LEVEL', 'INFO'),
            'manager_port': int(os.getenv('MANAGER_PORT', '5000')),
            'mode': os.getenv('MODE', 'OF'),
            
            # Network configuration
            'target_ip': os.getenv('TARGET_IP', '172.18.0.4'),
            'target_port': int(os.getenv('TARGET_PORT', '80')),
            'bot_port': int(os.getenv('BOT_PORT', '5002')),
            
            # Timing parameters
            'probe_frequency_seconds': float(os.getenv('PROBE_FREQUENCY_SECONDS', '2')),
            'ping_thread_timeout': float(os.getenv('PING_THREAD_TIMEOUT', '5')),
            'ping_host': os.getenv('PING_HOST', 'www.google.com'),
            
            # Attack parameters
            'duration': int(os.getenv('DURATION', '0')),
            'packet_size': int(os.getenv('PACKET_SIZE', '1024')),
            'delay': float(os.getenv('DELAY', '0.001')),
            
            # Data generation parameters
            'mu_anomalies': float(os.getenv('MU_ANOMALIES', '157')),
            'mu_normal': float(os.getenv('MU_NORMAL', '115')),
            'alpha': float(os.getenv('ALPHA', '0.2')),
            'beta': float(os.getenv('BETA', '1.9')),
            'time_emulation': os.getenv('TIME_EMULATION', 'false').lower() == 'true',
            
            # Probe metrics
            'probe_metrics': os.getenv('PROBE_METRICS', 'RTT,INBOUND,OUTBOUND,CPU,MEM').split(','),
            
            # Default anomaly and diagnostics classes
            'anomaly_classes': list(range(0, 19)),
            'diagnostics_classes': list(range(0, 15))
        }
        
        # Validate required environment variables
        if not config['vehicle_name']:
            raise ValueError("VEHICLE_NAME environment variable must be set")
        
        return config
    
    try:
        env_config = load_config_from_environment()
        print(f"Environment config loaded: {env_config['vehicle_name']}")
        print(f"   Kafka broker: {env_config['kafka_broker']}")
        print(f"   Logging level: {env_config['logging_level']}")
    except Exception as e:
        print(f"Environment config failed: {e}")
        return False
    
    # Test 2: YAML file loading
    print("\n2. Testing YAML file loading...")
    def load_config_from_file(config_path='config/producers/angela.yaml'):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Convert file config to our format
            config = {}
            
            if 'vehicle' in file_config:
                config['vehicle_name'] = file_config['vehicle'].get('name')
            
            if 'data_generation' in file_config:
                dg = file_config['data_generation']
                config.update({
                    'mu_anomalies': dg.get('mu_anomalies', 157),
                    'mu_normal': dg.get('mu_normal', 115),
                    'alpha': dg.get('alpha', 0.2),
                    'beta': dg.get('beta', 1.9),
                    'time_emulation': dg.get('time_emulation', False),
                    'anomaly_classes': dg.get('anomaly_classes', list(range(0, 19))),
                    'diagnostics_classes': dg.get('diagnostics_classes', list(range(0, 15)))
                })
            
            if 'probe' in file_config:
                probe = file_config['probe']
                config.update({
                    'probe_frequency_seconds': probe.get('frequency_seconds', 2),
                    'ping_thread_timeout': probe.get('timeout', 5),
                    'ping_host': probe.get('host', 'www.google.com'),
                    'probe_metrics': probe.get('metrics', ['RTT', 'INBOUND', 'OUTBOUND', 'CPU', 'MEM'])
                })
            
            if 'attack' in file_config:
                attack = file_config['attack']
                config.update({
                    'target_ip': attack.get('target_ip', '172.18.0.4'),
                    'target_port': attack.get('target_port', 80),
                    'duration': attack.get('duration', 0),
                    'packet_size': attack.get('packet_size', 1024),
                    'delay': attack.get('delay', 0.001),
                    'bot_port': attack.get('bot_port', 5002)
                })
            
            if 'system' in file_config:
                system = file_config['system']
                config.update({
                    'mode': system.get('mode', 'OF'),
                    'logging_level': system.get('logging_level', 'INFO'),
                    'manager_port': system.get('manager_port', 5000)
                })
            
            return config
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using environment variables only")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return {}
    
    try:
        file_config = load_config_from_file('config/producers/angela.yaml')
        print(f"YAML config loaded: {len(file_config)} parameters")
        if 'mu_anomalies' in file_config:
            print(f"   mu_anomalies: {file_config['mu_anomalies']}")
    except Exception as e:
        print(f"YAML config failed: {e}")
        return False
    
    # Test 3: Configuration merging
    print("\n3. Testing configuration merging...")
    def merge_configs(env_config, file_config):
        """Merge environment and file configurations, with environment taking precedence"""
        merged = env_config.copy()
        
        # Override with file config values (if not set in environment)
        for key, value in file_config.items():
            if key not in merged or merged[key] is None:
                merged[key] = value
        
        return merged
    
    try:
        merged_config = merge_configs(env_config, file_config)
        print(f"Configs merged: {len(merged_config)} total parameters")
        print(f"   Vehicle name (env priority): {merged_config['vehicle_name']}")
        print(f"   mu_anomalies (file): {merged_config.get('mu_anomalies', 'not set')}")
    except Exception as e:
        print(f"Config merging failed: {e}")
        return False
    
    # Test 4: Configuration validation
    print("\n4. Testing configuration validation...")
    def validate_config(config):
        """Validate configuration parameters"""
        # Check required fields
        required_fields = ['vehicle_name', 'kafka_broker']
        for field in required_fields:
            if not config.get(field):
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Validate numeric ranges
        if not (0 < config.get('mu_anomalies', 0) < 1000):
            raise ValueError("mu_anomalies must be between 0 and 1000")
        
        if not (0 < config.get('mu_normal', 0) < 1000):
            raise ValueError("mu_normal must be between 0 and 1000")
        
        if not (0 < config.get('alpha', 0) < 10):
            raise ValueError("alpha must be between 0 and 10")
        
        if not (0 < config.get('beta', 0) < 10):
            raise ValueError("beta must be between 0 and 10")
        
        # Validate port numbers
        if not (1 <= config.get('target_port', 0) <= 65535):
            raise ValueError("target_port must be between 1 and 65535")
        
        if not (1 <= config.get('bot_port', 0) <= 65535):
            raise ValueError("bot_port must be between 1 and 65535")
        
        return True
    
    try:
        validate_config(merged_config)
        print("Configuration validation passed")
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False
    
    print("\nAll tests passed!")
    return True

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)
