import requests
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional

class ContainerAPIClient:

    def __init__(self, timeout = 5):

        """
        Initialize the container API client

        Args:
            timeout (int): Default request timeout in seconds
        """
        self.timeout = timeout
        self.logger = logging.getLogger("ContainerAPIClient")
        self.containers = {} # Map of container_name -> container_info

    def register_container(self, name: str, host: str, port: int, container_type: str):

        """
        Register a container for communication

        Args:
            name (str): Container name/ID
            host (str): Container hostname or IP
            port (int): Container port
            container_type (str): Type of container
        """

        self.containers[name] = {

            "name": name,
            "host": host,
            "port": port,
            "type": container_type,
            "url": f"http://{host}:{port}"
        }

        self.logger.info(f"Registered container {name} at {host}:{port}")

    def get_container_url(self, name: str) -> str:

        """
        Get the base URL for a container

        Args:
            name (str): Container name/ID

        Returns:
            str: Container base URL

        Raises:
            ValueError: If container not found
        """

        if name not in self.containers:
            raise ValueError(f"Container '{name}' not registered")

        return self.containers[name]["url"]

    def get_status(self, name: str) -> Dict:

        """
        Get the base URL for a container

        Args:
            name (str): Container name/ID

        Returns:
            dict: Container status information
        """

        url = f"{self.get_container_url(name)}/status"
        response = requests.get(url, timeout = self.timeout)
        response.raise_for_status()

        return response.json()

    def get_all_statuses(self) -> Dict[str, Dict]:

        """
        Get status of all registered containers

        Returns:
            dict: Map of container name to status
        """

        results = {}

        def _get_status(name):

            try:
                return name, self.get_status(name)

            except Exception as e:

                self.logger.error(f"Error getting status for {name}: {str(e)}")
                return name, {"status": "error", "message": str(e)}

        with ThreadPoolExecutor(max_workers = 10) as executor:
            for name, status in executor.map(_get_status, self.containers.keys()):
                results[name] = status

        return results

    def start_container(self, name: str, params: Optional[Dict] = None) -> Dict:

        """
        Start a container

        Args:
            name (str): Container name/ID
            params (dict. optional): Start parameters

        Returns:
            dict: Response data
        """

        url = f"{self.get_container_url(name)}/start"
        response = requests.post(url, json = params or {}, timeout = self.timeout)
        response.raise_for_status()

        return response.json()

    def stop_container(self, name: str, params: Optional[Dict] = None) -> Dict:

        """
        Stop a container

        Args:
            name (str): Container name/ID
            params (dict, optional): Stop parameters

        Returns:
            dict: Response data
        """

        url = f"{self.get_container_url(name)}/stop"
        response = requests.post(url, json = params or {}, timeout = self.timeout)
        response.raise_for_status()

        return response.json()

    def send_command(self, name: str, command: str, params: Optional[Dict] = None) -> Dict:

        """
        Send a command to a container

        Args:
            name (str): Container name/ID
            command (str): Command to execute
            params (dict, optional): Command parameters

        Returns:
            dict: Response data
        """

        url = f"{self.get_container_url(name)}/command"
        payload = {

            "command": command,
            "params": params or {}
        }
        response = requests.post(url, json = payload, timeout = self.timeout)
        response.raise_for_status()

        return response.json()

    def batch_command(self, container_names: List[str], command: str, params: Optional[Dict] = None) -> Dict[str, Dict]:

        """
        Send the same command to multiple containers

        Args:
            container_names (list): List of container names
            command (str): Command to execute
            params (dict, optional): Command parameters

        Returns:
            dict: Map of container name to response
        """

        results = {}

        def _send_command(name):

            try:
                return name, self.send_command(name, command, params)

            except Exception as e:

                self.logger.error(f"Error sending command to {name}: {str(e)}")
                return name, {"status": "error", "message": str(e)}

        with ThreadPoolExecutor(max_workers = 10) as executor:
            for name, result in executor.map(_send_command, container_names):
                results[name] = result

        return results

    def get_containers_by_type(self, container_type: str) -> List[str]:

        """
        Get names of all containers of a specific type

        Args:
            container_type (str): Container type to filter by

        Returns:
            list: List of container names
        """

        return [name for name, info in self.containers.items()
                if info[type] == container_type]