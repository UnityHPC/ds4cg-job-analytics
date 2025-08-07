from abc import ABC, abstractmethod
import json
import os
from requests_cache import CachedSession
from pathlib import Path
from .enum_constants import NodeInfoKeyEnum


class RemoteConfigFetcher(ABC):
    """Class to fetch and parse partition information from a remote JSON file."""

    session = CachedSession('fetch_cache', expire_after=60)  # Cache for 60 seconds

    @property
    @abstractmethod
    def url(self) -> str:
        """URL of the remote JSON file to fetch."""
        pass

    @property
    @abstractmethod
    def local_path(self) -> Path:
        """Path where the local JSON file will be saved or read from."""
        pass

    @property
    @abstractmethod
    def info_name(self) -> str:
        """Type of information being fetched (e.g., 'partition')."""
        pass

    def _validate_info(self, info: list[dict]) -> bool:
        """Validate that the fetched information is a list of dictionaries.
        
        Args:
            info (list[dict]): The fetched information to validate.
            
        Raises:
                ValueError: If the fetched data is not in the expected format.

        Returns:
                bool: True if the data is valid, False otherwise.
        """
        if not isinstance(info, list):
            raise ValueError(f"Expected a list of dictionaries, got {type(info)} instead.")
        if not all(isinstance(item, dict) for item in info):
            raise ValueError("All items in the list must be dictionaries.")
        return True

    def get_info(self, offline: bool = False) -> list[dict]:
        """Fetch and save information from the remote JSON file, or read from local file if fetch fails.

        Args:
            offline (bool): If True, skip fetching from remote and only read from local file.

        Raises:
            FileNotFoundError: If the local file does not exist and the fetch fails.
            ValueError: If the fetched data is not in the expected format.

        Returns:
            list[dict]: Parsed JSON data.
        """
        if not offline:
            try:
                response = self.session.get(self.url, timeout=10)
                if response.status_code == 200:
                    remote_info = response.json()
                    if not self._validate_info(remote_info):
                        raise ValueError(f"Invalid {self.info_name} information format.")
                    if os.getenv("RUN_ENV") != "TEST":
                        # Ensure directory exists
                        self.local_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(self.local_path, "w") as f:
                            json.dump(remote_info, f, indent=2)
                        print(f"Fetched and saved {self.local_path.name} from remote URL.")
                    return remote_info
                else:
                    print(
                        f"Failed to retrieve {self.info_name} information. "
                        f"Status code: {response.status_code}\n"
                        f"URL: {self.url}\n"
                        f"Response: {response.text}"
                    )
            except Exception as e:
                print(f"An error occurred while fetching {self.info_name} information: {e}")

        # Fallback: read from local file if available
        try:
            with open(self.local_path) as file:
                local_info = json.load(file)
                if not self._validate_info(local_info):
                    raise ValueError(f"Invalid {self.info_name} information format in local file.")
            print(f"Loaded {self.local_path.name} from local file.")
            return local_info
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{self.local_path.name} not found locally at {self.local_path}. "
                "Please ensure the file exists."
            ) from e


class PartitionInfoFetcher(RemoteConfigFetcher):
    """Class to fetch and parse partition information from a remote JSON file."""

    @property
    def url(self) -> str:
        """URL of the remote JSON file to fetch."""
        return "https://gitlab.rc.umass.edu/unity/education/documentation/unity-website/-/raw/main/data/partition_info.json"

    @property
    def local_path(self) -> Path:
        """Path where the local JSON file will be saved or read from."""
        return Path(__file__).parent / "snapshots/partition_info.json"

    @property
    def info_name(self) -> str:
        """Type of information being fetched."""
        return "partition"
    
    def _validate_info(self, info: list[dict]) -> bool:
        """Validate the fetched partition information.

        This function checks that each partition info dictionary contains 'name' and 'type' keys.
        It also applies the base class validation.

        Args:
            info (list[dict]): The fetched partition information to validate.
        
        Raises:
            ValueError: If the fetched data does not contain the expected keys.
            
        Returns:
            bool: True if the data is valid, False otherwise."""

        if not all("name" in p and "type" in p for p in info):
            raise ValueError("Each partition info dictionary must contain 'name' and 'type' keys.")
        return super()._validate_info(info)


class NodeInfoFetcher(RemoteConfigFetcher):
    """Class to fetch and parse node information from a remote JSON file."""

    @property
    def url(self) -> str:
        """URL of the remote JSON file to fetch."""
        return "https://gitlab.rc.umass.edu/unity/education/documentation/unity-website/-/raw/main/data/node_info.json"

    @property
    def local_path(self) -> Path:
        """Path where the local JSON file will be saved or read from."""
        return Path(__file__).parent / "snapshots/node_info.json"

    @property
    def info_name(self) -> str:
        """Type of information being fetched."""
        return "node"
    
    def _validate_info(self, info: list[dict]) -> bool:
        """Validate the fetched node information.

        This function checks that each node info dictionary contains the keys listed in NodeInfoKeyEnum.
        It also applies the base class validation.

        Args:
            info (list[dict]): The fetched node information to validate.

        Raises:
            ValueError: If the fetched data does not contain the expected keys.
            
        Returns:
            bool: True if the data is valid, False otherwise."""

        for p in info:
            missing_keys = [key.value for key in NodeInfoKeyEnum.__members__.values() if key.value not in p]
            if missing_keys:
                raise ValueError(
                    "Each node info dictionary must contain the keys listed in NodeInfoKeyEnum."
                    f" Missing keys: {missing_keys}"
                    f"{' in ' + p['nodes_folded'] if 'nodes_folded' in p else ''}"
                )
        return super()._validate_info(info)
    
    def get_node_memory(self, node_name: str, offline: bool = False) -> int:
        """Get the total CPU memory of a node in GiB.

        Args:
            node_name (str): The name of the node.
            offline (bool): If True, skip fetching from remote and only read from local file.

        Raises:
            ValueError: If the node is not found in the node configuration file.
            ValueError: If there is an error calculating the node memory.

        Returns:
            int: Total memory of the node in GiB, or None if not found.
        """
        nodes_info = self.get_info(offline)
        try:
            # see if that name exists in the nodes info file
            for node_batch in nodes_info:
                for name in node_batch['nodes']:
                    if name == node_name:
                        return node_batch["ram"]
            # If the node is not found, raise an exception
            raise ValueError(f"Node '{node_name}' not found in node configuration file {self.local_path}.")
        except Exception as e:
            raise ValueError(f"Error calculating node memory from {self.local_path}: {e}") from e
