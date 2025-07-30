import json
import requests
from pathlib import Path


class PartitionInfoFetcher:
    """Class to fetch and parse partition information from a remote JSON file."""

    url = "https://gitlab.rc.umass.edu/unity/education/documentation/unity-website/-/raw/main/data/partition_info.json"
    local_path = Path("./snapshots/partition_info.json")

    @classmethod
    def get_partition_info(cls) -> dict:
        """Fetch and save partition information from the JSON file on GitLab, or read from local file if fetch fails.

        Raises:
            FileNotFoundError: If the local file does not exist and the fetch fails.

        Returns:
            dict: Parsed JSON data containing partition information.
        """
        partition_info = None
        try:
            response = requests.get(cls.url, timeout=10)
            if response.status_code == 200:
                partition_info = response.json()
                # Ensure directory exists
                cls.local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cls.local_path, "w") as f:
                    json.dump(partition_info, f, indent=2)
                print("Fetched and saved partition_info.json from GitLab.")
                return partition_info
            else:
                print(
                    f"Failed to retrieve partition information. "
                    f"Status code: {response.status_code}\n"
                    f"URL: {cls.url}\n"
                    f"Response: {response.text}"
                )
        except Exception as e:
            print(f"An error occurred while fetching partition information: {e}")

        # Fallback: read from local file if available
        try:
            with open(cls.local_path) as file:
                partition_info = json.load(file)
            print("Loaded partition_info.json from local file.")
            return partition_info
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"partition_info.json not found locally at {cls.local_path}. "
                "Please ensure the file exists."
            ) from e
