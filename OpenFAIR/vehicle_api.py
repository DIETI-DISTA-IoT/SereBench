from container_api import ContainerAPI
import threading
import time

class VehicleAPI(ContainerAPI):

    def __init__(self, vehicle_id, port = 5000):

        super().__init__(container_type = "vehicle", container_name = vehicle_id, port = port)
        self.vehicle_id = vehicle_id

        self.is_producing = False
        self.is_consuming = False
        self.is_attacking = False

        self.producer_thread = None
        self.consumer_thread = None
        self.attack_thread = None

    def get_detailed_status(self):

        return {

            "vehicle_id": self.vehicle_id,
            "is_producing": self.is_producing,
            "is_consuming": self.is_consuming,
            "is_attacking": self.is_attacking
        }

    def handle_start(self, data):

        """ Start the vehicle - can specify if starting producer, consumer or both """
        start_producer = data.get('producer', True)
        start_consumer = data.get('consumer', True)

        result = {"message": f"Vehicle {self.vehicle_id} started"}

        if start_producer and not self.is_producing:

            self.start_producer()
            result["producer"] = "started"

        if start_consumer and not self.is_consuming:

            self.start_consumer()
            result["consumer"] = "started"

        return result

    def handle_stop(self, data):

        """ Stop the vehicle - can specify if stopping producer, consumer or both """
        stop_producer = data.get('producer', True)
        stop_consumer = data.get('consumer', True)

        result = {"message": f"Vehicle {self.vehicle_id} stopped"}

        if stop_producer and self.is_producing:

            self.stop_producer()
            result["producer"] = "stopped"

        if stop_consumer and self.is_consuming:

            self.stop_consumer()
            result["consumer"] = "stopped"

    def handle_command(self, command, params):

        """ Handle vehicle-specific command """
        if command == "start_attack":

            target = params.get('target')
            if not target:
                return {"status": "error", "message": "Target is required for attack"}

            self.start_attack(target)

            return {"message": f"Started attack on {target}"}

        elif command == "stop_attack":

            self.stop_attack()
            return {"message": "Stopped attack"}

        else:
            return {"status": "error", "message": f"Unknown command: {command}"}, 400

    def start_producer(self):

        """ Start the data producer """
        self.is_producing = True
        self.producer_thread = threading.Thread(target = self._produce_data)
        self.producer_thread.daemon = True
        self.producer_thread.start()
        self.logger.info(f"Started producer for vehicle {self.vehicle_id}")

    def stop_producer(self):

        """ Stop the data producer """
        self.is_producing = False

        if self.producer_thread:
                self.producer_thread.join(timeout = 2.0)

        self.logger.info(f"Stopped producer for vehicle {self.vehicle_id}")

    def start_consumer(self):

        """ Start the data consumer """
        self.is_consuming = True
        self.consumer_thread = threading.Thread(target = self._consume_data)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()
        self.logger.info(f"Started consumer for vehicle {self.vehicle_id}")

    def stop_consumer(self):

        """ Stop the data consumer """
        self.is_consuming = False

        if self.consumer_thread:
            self.consumer_thread.join(timeout = 2.0)

        self.logger.info(f"Stopped consumer for vehicle {self.vehicle_id}")

    def start_attack(self, target):

        """ Start an attack on a target """
        self.is_attacking = True
        self.attack_thread = threading.Thread(target = self._perform_attack, args = (target,))
        self.attack_thread.daemon = True
        self.attack_thread.start()
        self.logger.info(f"Vehicle {self.vehicle_id} started attack on {target}")

    def stop_attack(self):

        """ Stop ongoing attack """
        self.is_attacking = False

        if self.attack_thread:
            self.attack_thread.join(timeout = 2.0)

        self.logger.info(f"Vehicle {self.vehicle_id} stopped attack")

    def _produce_data(self):

        """ Simulates producing data to Kafka """
        while self.is_producing:

            # TODO: Implement actual data production logic here
            self.logger.debug(f"Vehicle {self.vehicle_id} producing data")
            time.sleep(1)

    def _consume_data(self):

        """ Simulates consuming data from Kafka """
        while self.is_consuming:

            # TODO: Implement actual data consumption logic here
            self.logger.debug(f"Vehicle {self.vehicle_id} consuming data")
            time.sleep(1)

    def _perform_attack(self, target):

        """ Simulates performing an attack on a target """
        while self.is_attacking:

            # TODO: Implement actual attack logic here
            self.logger.debug(f"Vehicle {self.vehicle_id} attacking {target}")
            time.sleep(1)

# Example usage
if __name__ == "__main__":

    import os
    vehicle_id = os.environ.get("VEHICLE_ID", "e700_4801")
    port = int(os.environ.get("PORT", 5000))

    vehicle_api = VehicleAPI(vehicle_id = vehicle_id, port = port)
    vehicle_api.run()
