#!/usr/bin/env python3
"""
Simple API client to test producer HTTP endpoints
"""
import requests
import json
import time

class ProducerAPIClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def configure(self, config_data):
        """Configure the producer"""
        try:
            response = requests.post(
                f"{self.base_url}/configure",
                json=config_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Configuration failed: {e}")
            return None
    
    def start(self):
        """Start the producer"""
        try:
            response = requests.post(f"{self.base_url}/start", json={}, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Start failed: {e}")
            return None
    
    def stop(self):
        """Stop the producer"""
        try:
            response = requests.post(f"{self.base_url}/stop", json={}, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Stop failed: {e}")
            return None
    
    def status(self):
        """Get producer status"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Status check failed: {e}")
            return None
    
    def health(self):
        """Health check"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def get_config(self):
        """Get current configuration"""
        try:
            response = requests.get(f"{self.base_url}/config", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Get config failed: {e}")
            return None
    
    def update_config(self, updates):
        """Update configuration"""
        try:
            response = requests.put(
                f"{self.base_url}/config",
                json=updates,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Update config failed: {e}")
            return None

def test_api():
    """Test the API endpoints"""
    client = ProducerAPIClient()
    
    print("🧪 Testing Producer API...")
    
    # Test 1: Health check
    print("\n1. Health check...")
    health = client.health()
    if health:
        print(f"Health: {health}")
    else:
        print("Health check failed")
        return False
    
    # Test 2: Configure
    print("\n2. Configure producer...")
    config_data = {
        'vehicle_name': 'test_vehicle',
        'kafka_broker': 'kafka:9092',
        'logging_level': 'INFO',
        'manager_port': 5000,
        'mode': 'OF',
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
    
    config_result = client.configure(config_data)
    if config_result:
        print(f"Configured: {config_result['status']}")
    else:
        print("Configuration failed")
        return False
    
    # Test 3: Get configuration
    print("\n3. Get configuration...")
    current_config = client.get_config()
    if current_config:
        print(f"Current config: {len(current_config)} parameters")
        print(f"   Vehicle: {current_config.get('vehicle_name')}")
        print(f"   mu_anomalies: {current_config.get('mu_anomalies')}")
    else:
        print("Get config failed")
        return False
    
    # Test 4: Update configuration
    print("\n4. Update configuration...")
    updates = {
        'mu_anomalies': 200,
        'probe_frequency_seconds': 3
    }
    update_result = client.update_config(updates)
    if update_result:
        print(f"Updated: {update_result['status']}")
    else:
        print("Update failed")
        return False
    
    # Test 5: Start producer
    print("\n5. Start producer...")
    start_result = client.start()
    if start_result:
        print(f"Started: {start_result['status']}")
        print(f"   Vehicle: {start_result['vehicle']}")
    else:
        print("Start failed")
        return False
    
    # Test 6: Check status
    print("\n6. Check status...")
    time.sleep(2)  # Give it time to start
    status = client.status()
    if status:
        print(f"Status: {status['running']}")
        print(f"   Records produced: {status.get('records_produced', 0)}")
        print(f"   Under attack: {status.get('under_attack', False)}")
    else:
        print("Status check failed")
        return False
    
    # Test 7: Stop producer
    print("\n7. Stop producer...")
    stop_result = client.stop()
    if stop_result:
        print(f"Stopped: {stop_result['status']}")
    else:
        print("Stop failed")
        return False
    
    print("\n🎉 All API tests passed!")
    return True

if __name__ == "__main__":
    success = test_api()
    exit(0 if success else 1)
