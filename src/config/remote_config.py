from abc import ABC, abstractmethod
import json
import os
from requests_cache import CachedSession
from pathlib import Path


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

    def get_info(self, offline: bool = False) -> dict:
        """Fetch and save information from the remote JSON file, or read from local file if fetch fails.

        Args:
            offline (bool): If True, skip fetching from remote and only read from local file.

        Raises:
            FileNotFoundError: If the local file does not exist and the fetch fails.

        Returns:
            dict: Parsed JSON data.
        """
        if not offline:
            try:
                response = self.session.get(self.url, timeout=10)
                if response.status_code == 200:
                    remote_info = response.json()
                    if not os.getenv("PYTEST_VERSION"):
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