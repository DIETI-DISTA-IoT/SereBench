import docker
import logging
import requests
import time
import yaml
import os
from omegaconf import ListConfig, DictConfig, OmegaConf

class ProducerManager:
    def __init__(self, cfg, producers, containers_ips, PRODUCER_COMMAND="python produce.py"):
        self.cfg = cfg
        self.producers = producers
        self.containers_ips = containers_ips
        self.producer_command = PRODUCER_COMMAND
        self.logging_level = cfg.logging_level
        self.mode = cfg.mode
        self.manager_port = cfg.container_manager_port
        self.probe_metrics = cfg.security_manager.probe_metrics
        self.no_proxy_host = cfg.dashboard.proxy
        self.attack_config = cfg.attack
        
        # HTTP session configured to ignore system proxy settings for internal Docker IPs
        self.http = requests.Session()
        # Do not use environment proxies (prevents corporate proxy from intercepting 172.* calls)
        self.http.trust_env = False
        
        # Initialize vehicle configurations
        self.vehicle_configs = {}
        self.vehicle_names = []
        
        for vehicle in cfg.vehicles:
            if type(vehicle) == str:
                vehicle_name = vehicle
                vehicle_config = cfg.default_vehicle_config.copy()
            else:
                vehicle_name = list(vehicle.keys())[0]
                vehicle_config = cfg.default_vehicle_config.copy()
                vehicle_config.update(vehicle[vehicle_name])
            
            self.vehicle_names.append(vehicle_name)
            self.vehicle_configs[vehicle_name] = vehicle_config
            
            # Set default classes if not specified
            if vehicle_config.get("anomaly_classes") == "all":
                self.vehicle_configs[vehicle_name]["anomaly_classes"] = list(range(0, 19))
            if vehicle_config.get("diagnostics_classes") == "all":
                self.vehicle_configs[vehicle_name]["diagnostics_classes"] = list(range(1, 15))

    def _convert_to_json_serializable(self, obj):
        """Convert OmegaConf objects to JSON serializable Python objects"""
        if isinstance(obj, (ListConfig, list)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (DictConfig, dict)):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            # Handle other OmegaConf objects by converting to dict first
            try:
                return self._convert_to_json_serializable(OmegaConf.to_container(obj))
            except:
                return str(obj)
        else:
            # Handle primitive types
            return obj

    def start_all_producers(self):
        """Start all producers using HTTP API"""
        results = []
        for producer_name, vehicle_name in zip(self.producers.keys(), self.vehicle_names):
            result = self.start_producer(producer_name, self.producers[producer_name], self.vehicle_configs[vehicle_name])
            results.append(result)
        return "All producers started!", results

    def start_producer(self, producer_name, producer_container, vehicle_config):
        """Start producer using HTTP API instead of command execution"""
        try:
            # Get container IP
            container_ip = self.containers_ips.get(producer_name)
            if not container_ip:
                return f"Failed to start producer {producer_name}: Container IP not found"
            
            # Build candidate URLs: prefer DNS name when running inside the Docker network,
            # but fall back to direct container IP when running on host
            hostname_url = f"http://{producer_name}:5000"
            ip_url = f"http://{container_ip}:5000"

            # Pick the first URL whose health endpoint looks like the producer API (not the dashboard)
            api_url = None
            for candidate in (hostname_url, ip_url):
                if self._wait_for_health(candidate, overall_timeout_seconds=20):
                    api_url = candidate
                    break
            if not api_url:
                error_msg = f"Failed to start producer {producer_name}: API not healthy at {hostname_url} or {ip_url}"
                logging.getLogger("PRODUCER_MANAGER").error(error_msg)
                return error_msg
            
            
            # Step 1: Configure the producer
            config_data = self._build_config_data(vehicle_config, producer_name.split('_')[0])
            
            # Test JSON serialization before sending
            try:
                import json
                json.dumps(config_data)
            except TypeError as e:
                error_msg = f"JSON serialization failed for {producer_name}: {e}"
                logging.getLogger("PRODUCER_MANAGER").error(error_msg)
                # Log the problematic data structure
                logging.getLogger("PRODUCER_MANAGER").error(f"Config data keys: {list(config_data.keys())}")
                for key, value in config_data.items():
                    try:
                        json.dumps(value)
                    except TypeError:
                        logging.getLogger("PRODUCER_MANAGER").error(f"Non-serializable key: {key}, type: {type(value)}, value: {value}")
                return error_msg
            
            config_response = self.http.post(
                f"{api_url}/configure",
                json=config_data,
                timeout=30
            )
            config_response.raise_for_status()
            
            # Step 2: Start the producer
            start_response = self.http.post(
                f"{api_url}/start",
                json={},
                timeout=30
            )
            start_response.raise_for_status()
            
            # Step 3: Verify it's running
            status_response = self.http.get(f"{api_url}/status", timeout=10)
            status_response.raise_for_status()
            
            logging.getLogger("PRODUCER_MANAGER").info(f"Producer {producer_name} started successfully")
            return f"Producer {producer_name} started successfully"
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to start producer {producer_name}: {e}"
            logging.getLogger("PRODUCER_MANAGER").error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Failed to start producer {producer_name}: {e}"
            logging.getLogger("PRODUCER_MANAGER").error(error_msg)
            return error_msg

    def stop_producer(self, producer_name, producer_container):
        """Stop producer using HTTP API"""
        try:
            container_ip = self.containers_ips.get(producer_name)
            if not container_ip:
                return f"Failed to stop producer {producer_name}: Container IP not found"
            
            api_url = f"http://{container_ip}:5000"
            response = self.http.post(f"{api_url}/stop", json={}, timeout=30)
            response.raise_for_status()
            
            return f"Producer {producer_name} stopped successfully"
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to stop producer {producer_name}: {e}"
            logging.getLogger("PRODUCER_MANAGER").error(error_msg)
            return error_msg

    def get_producer_status(self, producer_name, producer_container):
        """Get producer status via HTTP API"""
        try:
            container_ip = self.containers_ips.get(producer_name)
            if not container_ip:
                return {"error": "Container IP not found"}
            
            api_url = f"http://{container_ip}:5000"
            response = self.http.get(f"{api_url}/status", timeout=10)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def update_producer_config(self, producer_name, producer_container, new_config):
        """Update producer configuration via HTTP API"""
        try:
            container_ip = self.containers_ips.get(producer_name)
            if not container_ip:
                return f"Failed to update producer {producer_name}: Container IP not found"
            
            api_url = f"http://{container_ip}:5000"
            response = self.http.put(
                f"{api_url}/config",
                json=new_config,
                timeout=30
            )
            response.raise_for_status()
            
            return f"Producer {producer_name} configuration updated successfully"
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to update producer {producer_name}: {e}"
            logging.getLogger("PRODUCER_MANAGER").error(error_msg)
            return error_msg

    def _build_config_data(self, vehicle_config, vehicle_name):
        """Build configuration data for HTTP API"""
        # Convert all OmegaConf objects to JSON serializable Python objects
        vehicle_config = self._convert_to_json_serializable(vehicle_config)
        probe_metrics = self._convert_to_json_serializable(self.probe_metrics)
        attack_config = self._convert_to_json_serializable(self.attack_config)
        
        # Debug logging to identify problematic values
        logging.getLogger("PRODUCER_MANAGER").debug(f"Vehicle config type: {type(vehicle_config)}")
        logging.getLogger("PRODUCER_MANAGER").debug(f"Probe metrics type: {type(probe_metrics)}")
        logging.getLogger("PRODUCER_MANAGER").debug(f"Attack config type: {type(attack_config)}")
        
        config_data = {
            'vehicle_name': vehicle_name,
            'kafka_broker': vehicle_config.get('kafka_broker', 'kafka:9092'),
            'logging_level': self.logging_level,
            'manager_port': self.manager_port,
            'mode': self.mode,
            
            # Network configuration
            'target_ip': attack_config.get('target_ip', '172.18.0.4'),
            'target_port': attack_config.get('target_port', 80),
            'bot_port': attack_config.get('bot_port', 5002),
            
            # Timing parameters
            'probe_frequency_seconds': vehicle_config.get('probe_frequency_seconds', 2),
            'ping_thread_timeout': vehicle_config.get('ping_thread_timeout', 5),
            'ping_host': vehicle_config.get('ping_host', 'www.google.com'),
            
            # Attack parameters
            'duration': attack_config.get('duration', 0),
            'packet_size': attack_config.get('packet_size', 1024),
            'delay': attack_config.get('delay', 0.001),
            
            # Data generation parameters
            'mu_anomalies': vehicle_config.get('mu_anomalies', 157),
            'mu_normal': vehicle_config.get('mu_normal', 115),
            'alpha': vehicle_config.get('alpha', 0.2),
            'beta': vehicle_config.get('beta', 1.9),
            'time_emulation': vehicle_config.get('time_emulation', False),
            
            # Probe metrics
            'probe_metrics': probe_metrics,
            
            # Anomaly and diagnostics classes
            'anomaly_classes': vehicle_config.get('anomaly_classes', list(range(0, 19))),
            'diagnostics_classes': vehicle_config.get('diagnostics_classes', list(range(0, 15)))
        }
        
        # Final conversion to ensure everything is JSON serializable
        config_data = self._convert_to_json_serializable(config_data)
        
        return config_data

    def stop_all_producers(self):
        """Stop all producers using HTTP API"""
        results = []
        for producer_name in self.producers.keys():
            result = self.stop_producer(producer_name, self.producers[producer_name])
            results.append(result)
        return "All producers stopped!", results

    def get_all_producer_statuses(self):
        """Get status of all producers"""
        statuses = {}
        for producer_name, producer_container in self.producers.items():
            status = self.get_producer_status(producer_name, producer_container)
            statuses[producer_name] = status
        return statuses

    def _wait_for_health(self, api_url, overall_timeout_seconds=60, poll_interval_seconds=2):
        """Poll the producer /health endpoint until healthy or timeout.
        Returns True if healthy, False otherwise.
        """
        logger = logging.getLogger("PRODUCER_MANAGER")
        deadline = time.time() + overall_timeout_seconds
        while time.time() < deadline:
            try:
                resp = self.http.get(f"{api_url}/health", timeout=5)
                if resp.ok:
                    # Validate it's the producer health, not dashboard's
                    try:
                        data = resp.json()
                        # Producer health has keys like 'running' or 'config_loaded' or 'vehicle'
                        if isinstance(data, dict) and ("running" in data or "config_loaded" in data or "vehicle" in data):
                            logger.info(f"Health check passed for {api_url}: {data}")
                            return True
                        else:
                            logger.debug(f"Health check response doesn't look like producer API: {data}")
                    except Exception as e:
                        logger.debug(f"Failed to parse health response from {api_url}: {e}")
                else:
                    logger.debug(f"Health check failed with status {resp.status_code} for {api_url}")
            except requests.exceptions.RequestException as e:
                logger.debug(f"Health check request failed for {api_url}: {e}")
            time.sleep(poll_interval_seconds)
        logger.error(f"Health check timed out or not producer API for {api_url}")
        return False