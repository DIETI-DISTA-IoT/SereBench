import docker
import logging
from OpenFAIR.producer_manager import ProducerManager
from OpenFAIR.consumer_manager import ConsumerManager
from OpenFAIR.attack_agent import AttackAgent
from OpenFAIR.dash_monitor import DashBoardMonitor
import threading
import subprocess
import time
import signal
from confluent_kafka import SerializingProducer
from confluent_kafka.serialization import StringSerializer
import json
import requests
import os
import platform
import socket


WANDBER_COMMAND = "python wandber.py"
FL_COMMAND = "python federated_learning.py"
SM_COMMAND = "python security_manager.py"
ATTACK_COMMAND = "python attack.py"
HEALTHY = "HEALTHY"
INFECTED = "INFECTED"


class ContainerManager:
    
    def __init__(self, cfg):
        self.logger = logging.getLogger("CONTAINER_MANAGER")
        self.logger.setLevel(cfg.logging_level.upper())
        # Connect to the Docker daemon
        self.client = docker.from_env()
        self.wandber = {
            'container': None,
            'thread': None}
        self.federated_learner = {
            'container': None,
            'thread': None
        }
        self.security_manager = {
            'container': None,
            'thread': None
        }
        self.producers = {}
        self.consumers = {}
        self.containers_ips = {}
        self.cfg = cfg
        self.vehicle_names = []
        for vehicle in cfg.vehicles:
            if type(vehicle) == str:
                vehicle_name = vehicle
            else:
                vehicle_name = list(vehicle.keys())[0]
            self.vehicle_names.append(vehicle_name) 

        self.vehicle_status_dict = self.init_vehicle_status_dict()
        self.last_attack_started_at = {vehicle_name: time.time() for vehicle_name in self.vehicle_names}
        self.refresh_containers()
        self.host_ip = self.get_my_ip()
        self.logger.info(f"Host IP: {self.host_ip}")
        self.attack_agent = AttackAgent(self, cfg)
        self.start_dashboard_monitor()
        if cfg.dashboard.proxy:
            self.logger.info("Proxy mode enabled")
            self.proxy_configuration()


    def proxy_configuration(self):
        # get the value of the no_proxy env var:
        no_proxy = os.environ.get('no_proxy', '')
        for node_ip in self.containers_ips.values():
            if node_ip and node_ip not in no_proxy:
                no_proxy += f",{node_ip}"
        os.environ['no_proxy'] = no_proxy


    def start_dashboard_monitor(self):
        # Start the dashboard monitor
        conf_prod = {
            'bootstrap.servers': self.cfg.dashboard.kafka_broker_url,
            'key.serializer': StringSerializer('utf_8'),
            'value.serializer': lambda x, ctx: json.dumps(x).encode('utf-8')
        }
        self.producer = SerializingProducer(conf_prod)
        signal.signal(signal.SIGINT, lambda sig, frame: self.signal_handler(sig, frame))
        self.monitor = DashBoardMonitor(self.logger, self.cfg)
        self.monitor_thread = threading.Thread(target=self.health_probes_thread)
        self.monitor_thread.daemon = True
        self.monitor_alive = True
        self.monitor_thread.start()
        

    def start_automatic_attacks(self):
        self.logger.info("Starting automatic Attack Agent")
        self.attack_agent.alive = True
        self.attack_agent.thread.start()
        return "Automatic Attack Agent started!"


    def stop_automatic_attacks(self):
        self.logger.info("Stopping automatic Attack Agent")
        self.attack_agent.alive = False
        if self.attack_agent.thread.is_alive():
            self.attack_agent.thread.join(1)
        self.attack_agent.stop_all_attacks()
        self.logger.info("Attack Agent stopped correctly.")
        return "Automatic Attack Agent stopped!"

    def produce_message(self, data, topic_name):

        try:
            self.producer.produce(topic=topic_name, value=data)  # Send the message to Kafka
            self.logger.debug(f"sent health probes.")
        except Exception as e:
            self.logger.info(f"Error while producing message to {topic_name} : {e}")


    def signal_handler(self, sig, frame):
        self.monitor_alive = False
        self.monitor_thread.join(1)
        self.logger.info(f"Dashboard monitor stopped correctly.")
        if self.attack_agent.alive:
            self.stop_automatic_attacks()
        self.producer.flush()
        self.logger.info(f"Exiting...")
        exit(0)


    def health_probes_thread(self):
        self.logger.info(f"Starting thread for dashboard health probes")
        while self.monitor_alive:
            health_dict = self.monitor.probe_health()
            self.produce_message(
                data=health_dict, 
                topic_name=f"DASHBOARD_PROBES")
            time.sleep(self.cfg.dashboard.probe_frequency_seconds)


    def get_my_ip(self):
        """Get the host IP address, works on both Windows and Linux"""
        try:
            # Try to get IP from environment variable first
            host_ip = os.getenv('HOST_IP')
            if host_ip:
                return host_ip
            
            # Detect OS and use appropriate method
            if platform.system() == "Windows":
                # Windows method: use socket to get local IP
                try:
                    # Connect to a remote address to determine local IP
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                    return local_ip
                except Exception:
                    # Fallback: get hostname and resolve
                    hostname = socket.gethostname()
                    return socket.gethostbyname(hostname)
            else:
                # Linux/Unix method: use hostname command
                cmd = "hostname -I | cut -d' ' -f1"
                return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE).stdout.decode().strip()
        except Exception as e:
            self.logger.warning(f"Could not determine host IP: {e}")
            # Fallback to localhost
            return "127.0.0.1"


    def init_vehicle_status_dict(self):
        vehicle_status_dict = {}
        for vehicle_name in self.vehicle_names:
            vehicle_status_dict[vehicle_name] = HEALTHY
        self.logger.info("Vehicle State Dictionary:")
        for vehicle, state in vehicle_status_dict.items():
            self.logger.info(f"  {vehicle}: {state}")  

        return vehicle_status_dict


    def create_vehicles(self):

        for vehicle_name in self.vehicle_names:

            self.create_producer(vehicle_name)
            self.create_consumer(vehicle_name)

        self.refresh_containers()
        return "Vehicles created!"


    def delete_vehicles(self):
        self.logger.info("Deleting vehicles")
        for producer in self.producers.values():
            producer.stop()
            producer.remove()
        for consumer in self.consumers.values():
            consumer.stop()
            consumer.remove()
        return "Vehicles deleted!"


    def create_producer(self, vehicle_name):
        container_name = f"{vehicle_name}_producer"
        env_vars = {
            "VEHICLE_NAME": vehicle_name,
            "HOST_IP": self.host_ip,
        }
        cpu_period = int(self.producer_manager.vehicle_configs[vehicle_name]['cpu_period'])
        cpu_quota = int(self.producer_manager.vehicle_configs[vehicle_name]['cpu_quota'])
        cpuset_cpus = str(self.producer_manager.vehicle_configs[vehicle_name]['cpu_cores'])

        self.client.containers.run(
            image="open_fair-producer",
            name=container_name,
            detach=True,
            network="trains_network",
            environment=env_vars,
            cpu_period=cpu_period,
            cpu_quota=cpu_quota,
            cpuset_cpus=cpuset_cpus
            # rely on image CMD to start the app
        )
        

    def create_consumer(self, vehicle_name):
        container_name = f"{vehicle_name}_consumer"
        env_vars = {
            "VEHICLE_NAME": vehicle_name,
            "HOST_IP": self.host_ip,
        }
        cpu_period = int(self.consumer_manager.consumer_configs[vehicle_name]['cpu_period'])
        cpu_quota = int(self.consumer_manager.consumer_configs[vehicle_name]['cpu_quota'])
        cpuset_cpus = str(self.consumer_manager.consumer_configs[vehicle_name]['cpu_cores'])

        self.client.containers.run(
            image="open_fair-consumer",
            name=container_name,
            detach=True,
            network="trains_network",
            environment=env_vars,
            cpu_period=cpu_period,
            cpu_quota=cpu_quota,
            cpuset_cpus=cpuset_cpus
            # rely on image CMD to start the app
        )


    def refresh_containers(self):     

        for container in self.client.containers.list():
            container_info = self.client.api.inspect_container(container.id)
            # Extract the IP address of the container from its network settings
            container_img_name = container_info['Config']['Image']
            # Prefer the compose network name 'trains_network'
            networks = container_info['NetworkSettings']['Networks']
            network_name = 'trains_network' if 'trains_network' in networks else next(iter(networks.keys()), None)
            container_ip = networks[network_name]['IPAddress'] if network_name else None
            self.logger.info(f'Found {container.name} container with ip {container_ip}')
            # Classify by container name for reliability
            if 'producer' in container.name:
                self.producers[container.name] = container
            elif 'consumer' in container.name:
                self.consumers[container.name] = container
            elif 'wandber' in container.name:
                self.wandber['container'] = container
                self.federated_learner['container'] = container
                self.security_manager['container'] = container

            self.containers_ips[container.name] = container_ip 
        

        self.producer_manager = ProducerManager(self.cfg, self.producers, self.containers_ips)
        self.consumer_manager = ConsumerManager(self.cfg, self.consumers)
            
    
    def produce_all(self):
        message, results = self.producer_manager.start_all_producers()
        # Log individual results
        for result in results:
            self.logger.info(result)
        return message


    def stop_producing_all(self):
        message, results = self.producer_manager.stop_all_producers()
        # Log individual results
        for result in results:
            self.logger.info(result)
        return message


    def consume_all(self):
        message, results = self.consumer_manager.start_all_consumers()
        for result in results:
            self.logger.info(result)
        return message


    def stop_consuming_all(self):
        self.consumer_manager.stop_all_consumers()
        return "All consumers stopped!"


    def stop_wandb(self):
        self.wandber['container']
        try:
            # Try to find and kill the process
            pid_result = self.wandber['container'].exec_run(f"pgrep -f '{WANDBER_COMMAND}'")
            pid = pid_result[1].decode().strip()
            
            if pid:
                self.wandber['container'].exec_run(f"kill -SIGINT {pid}")
                self.logger.info(f"Stopping wandber...")
                return "Stopping wandber..."
            else:
                self.logger.info(f"No running process found for wandber")
                return "No running process found for wandber"
        except Exception as e:
            self.logger.error(f"Error stopping wandber: {e}")
            return "Error stopping wandber"


    def start_wandb(self):

        start_command = f"python wandber.py " + \
            f" --logging_level={self.cfg.logging_level} " + \
            f" --project_name={self.cfg.wandb.project_name} " + \
            f" --run_name={self.cfg.wandb.run_name} " + \
            f" --kafka_broker_url={self.cfg.wandb.kafka_broker_url} " + \
            f" --kafka_consumer_group_id={self.cfg.wandb.kafka_consumer_group_id} " + \
            f" --kafka_auto_offset_reset={self.cfg.wandb.kafka_auto_offset_reset} " + \
            f" --kafka_topic_update_interval_secs={self.cfg.kafka_topic_update_interval_secs}"
                
        if self.cfg.wandb.online:
            start_command += " --online"
        
        def run_wandber(self):
            return_tuple = self.wandber['container'].exec_run(
                 start_command,
                 tty=True,
                 stream=True,
                 stdin=True)
            for line in return_tuple[1]:
                self.logger.info(line.decode().strip())
        
        thread = threading.Thread(target=run_wandber, args=(self,))
        thread.start()
        return "Wandber consumer started!"


    def start_federated_learning(self):

        start_command = FL_COMMAND + \
            f" --logging_level={self.cfg.logging_level} " + \
            f" --project_name={self.cfg.wandb.project_name} " + \
            f" --run_name={self.cfg.wandb.run_name} " + \
            f" --kafka_broker_url={self.cfg.wandb.kafka_broker_url} " + \
            f" --kafka_consumer_group_id={self.cfg.wandb.kafka_consumer_group_id} " + \
            f" --kafka_auto_offset_reset={self.cfg.wandb.kafka_auto_offset_reset} " + \
            f" --kafka_topic_update_interval_secs={self.cfg.kafka_topic_update_interval_secs}" +\
            f" --aggregation_strategy={self.cfg.federated_learning.aggregation_strategy}" +\
            f" --initialization_strategy={self.cfg.federated_learning.initialization_strategy}" +\
            f" --aggregation_interval_secs={self.cfg.federated_learning.aggregation_interval_secs}" +\
            f" --weights_buffer_size={self.cfg.federated_learning.weights_buffer_size}" +\
            f" --output_dim={self.cfg.anomaly_detection.output_dim}" + \
            f" --h_dim={self.cfg.anomaly_detection.h_dim}" + \
            f" --num_layers={self.cfg.anomaly_detection.num_layers}" +\
            f" --input_dim={self.cfg.anomaly_detection.input_dim}" +\
            " --probe_metrics=" + ",".join(map(str,self.cfg.security_manager.probe_metrics)) + \
            " --mode=" + str(self.cfg.mode)
        
        if self.cfg.wandb.online:
            start_command += " --online"
        if self.cfg.anomaly_detection.layer_norm:
            start_command += " --layer_norm"
        
        def run_federated_learning(self):
            return_tuple = self.wandber['container'].exec_run(
                 start_command,
                 tty=True,
                 stream=True,
                 stdin=True)
            for line in return_tuple[1]:
                self.logger.info(line.decode().strip())
        
        thread = threading.Thread(target=run_federated_learning, args=(self,))
        thread.start()
        return "Federated learning started!"
    

    def stop_federated_learning(self):
        try:
            # Try to find and kill the process
            pid_result = self.federated_learner['container'].exec_run(f"pgrep -f '{FL_COMMAND}'")
            pid = pid_result[1].decode().strip()
            
            if pid:
                self.federated_learner['container'].exec_run(f"kill -SIGINT {pid}")
                m = "Stopping FL..."
                self.logger.info(m)
                return m
            else:
                m = "No running process found for federated learning"
                self.logger.info(m)
                return m
        except Exception as e:
            m = f"Error stopping federated learning: {e}"
            self.logger.error(m)
            return m

        
    def start_security_manager(self):

        assert len(self.vehicle_names) > 0, "No vehicles found. Please create vehicles first."
        vehicle_param_str = self.vehicle_names[0]
        for vehicle in self.vehicle_names[1:]:
            vehicle_param_str += f"\ {vehicle}"


        start_command = SM_COMMAND + \
            f" --logging_level={self.cfg.logging_level} " + \
            f" --kafka_broker_url={self.cfg.wandb.kafka_broker_url} " + \
            f" --kafka_consumer_group_id={self.cfg.wandb.kafka_consumer_group_id} " + \
            f" --kafka_auto_offset_reset={self.cfg.wandb.kafka_auto_offset_reset} " + \
            f" --kafka_topic_update_interval_secs={self.cfg.kafka_topic_update_interval_secs}" +\
            f" --initialization_strategy={self.cfg.security_manager.initialization_strategy}" +\
            f" --buffer_size={self.cfg.security_manager.buffer_size}" +\
            f" --batch_size={self.cfg.security_manager.batch_size}" +\
            f" --learning_rate={self.cfg.security_manager.learning_rate}" +\
            f" --epoch_size={self.cfg.security_manager.epoch_size}" +\
            f" --training_freq_seconds={self.cfg.security_manager.training_freq_seconds}" +\
            f" --save_model_freq_epochs={self.cfg.security_manager.save_model_freq_epochs}" +\
            f" --model_saving_path={self.cfg.security_manager.model_saving_path}" + \
            f" --vehicle_names={vehicle_param_str}" + \
            f" --initialization_strategy={self.cfg.security_manager.initialization_strategy}" + \
            f" --input_dim={len(self.cfg.security_manager.probe_metrics)}" + \
            f" --output_dim={self.cfg.security_manager.output_dim}" + \
            f" --h_dim={self.cfg.security_manager.hidden_dim}" + \
            f" --num_layers={self.cfg.security_manager.num_layers}" + \
            f" --dropout={self.cfg.security_manager.dropout}" + \
            f" --optimizer={self.cfg.security_manager.optimizer}" + \
            f" --manager_port={self.cfg.dashboard.port}" + \
            f" --sm_port={self.cfg.security_manager.sm_port}" +\
            f" --true_positive_reward={self.cfg.security_manager.true_positive_reward}" + \
            f" --false_positive_reward={self.cfg.security_manager.false_positive_reward}" + \
            f" --true_negative_reward={self.cfg.security_manager.true_negative_reward}" + \
            f" --false_negative_reward={self.cfg.security_manager.false_negative_reward}"
            
        
        if self.cfg.security_manager.mitigation:
            start_command += f" --mitigation"
        if self.cfg.security_manager.layer_norm:
            start_command += f" --layer_norm"
        
        def run_security_manager(self):
            return_tuple = self.wandber['container'].exec_run(
                 start_command,
                 tty=True,
                 stream=True,
                 stdin=True)
            for line in return_tuple[1]:
                self.logger.info(line.decode().strip())
        
        thread = threading.Thread(target=run_security_manager, args=(self,))
        thread.start()
        return "Security Manager started!", 200
    

    def stop_security_manager(self):
        try:
            # Try to find and kill the process
            pid_result = self.security_manager['container'].exec_run(f"pgrep -f '{SM_COMMAND}'")
            pid = pid_result[1].decode().strip()
            
            if pid:
                self.security_manager['container'].exec_run(f"kill -SIGINT {pid}")
                m = "Stopping SM..."
                self.logger.info(m)
                return m
            else:
                m = "No running process found for security manager"
                self.logger.info(m)
                return m
        except Exception as e:
            m = f"Error stopping security manager: {e}"
            self.logger.error(m)
            return m
        
    
    def start_attack_from_vehicle(self, vehicle_name, origin):

        assert f"{vehicle_name}_producer" in self.producers
        
        self.logger.info(f"[BOTMASTER] Starting Attack from  {vehicle_name}")
        bot_ip = self.containers_ips[f'{vehicle_name}_producer']
        bot_port = self.cfg.attack.bot_port
        url = f"http://{bot_ip}:{bot_port}/start-attack"
        response = requests.post(url, json={})
        preamble = "Automatic" if origin == "AI" else "Manual"
        if response.status_code == 200:
            self.vehicle_status_dict[vehicle_name] = INFECTED
            self.last_attack_started_at[vehicle_name] = time.time()
            m = f"[BOTMASTER] - {preamble} attack at {vehicle_name} started successfully."
            self.logger.info(m)
        else:
            m = f"[BOTMASTER] - Failed to start {preamble} attack at {vehicle_name}. Status code: {response.status_code}"
            self.logger.error(m)
        return m, response.status_code
    

    def stop_attack_from_vehicle(self, vehicle_name, origin):

        assert f"{vehicle_name}_producer" in self.producers
        reactive_mitigation_time = None

        self.logger.info(f"[ATO] Healing attack on {vehicle_name}...")
        bot_ip = self.containers_ips[f'{vehicle_name}_producer']
        bot_port = self.cfg.attack.bot_port
        url = f"http://{bot_ip}:{bot_port}/stop-attack"
        response = requests.post(url, json={})
        adverb = "automatically" if origin == "AI" else "manually"
        if response.status_code == 200:
            m = f"[ATO] Attack at {vehicle_name} stopped {adverb}."
            self.logger.info(m)
            self.vehicle_status_dict[f"{vehicle_name}"] = HEALTHY
            reactive_mitigation_time = time.time() - self.last_attack_started_at[vehicle_name]
        else:
            m = f"[ATO] Failed to stop attack at {vehicle_name} {adverb}. Status code: {response.status_code}"
            self.logger.error(m)

        return {"message" :m, "mitigation_time": reactive_mitigation_time}


    def start_preconf_attack(self):
        for attacking_vehicle_name in self.cfg.attack.preconf_attacking_vehicles:
            self.start_attack_from_vehicle(attacking_vehicle_name, origin="MANUAL")
        return "Preconfigured attack started!"
    

    def stop_preconf_attack(self):
        for attacking_vehicle_name in self.cfg.attack.preconf_attacking_vehicles:
            self.stop_attack_from_vehicle(attacking_vehicle_name, "MANUALLY")
        return "Preconfigured attack stopped!"
    

    def get_vehicle_status(self, vehicle_name):
        return self.vehicle_status_dict[vehicle_name]
    

    def start_mitigation(self):
        security_manager_ip = self.containers_ips['wandber']
        mitigation_service_port = self.cfg.security_manager.sm_port
        url = f"http://{security_manager_ip}:{mitigation_service_port}/start-mitigation"
        response = requests.post(url, json={})
        
        if response.status_code == 200:
            m = "Mitigation started successfully."
            self.logger.info(m)
        else:
            m = f"Failed to start mitigation. Status code: {response.status_code}"
            self.logger.error(m)
        return m, response.status_code


    def stop_mitigation(self):
        security_manager_ip = self.containers_ips['wandber']
        mitigation_service_port = self.cfg.security_manager.sm_port
        url = f"http://{security_manager_ip}:{mitigation_service_port}/stop-mitigation"
        response = requests.post(url, json={})
        
        if response.status_code == 200:
            m = "Mitigation stopped successfully."
            self.logger.info(m)
        else:
            m = f"Failed to stop mitigation. Status code: {response.status_code}"
            self.logger.error(m)
        return m, response.status_code

