import shutil
import tempfile
from collections.abc import Generator

import pandas as pd
import pytest

from src.database import DatabaseConnection
from .mock_data.convert_csv_to_db import convert_csv_to_db
from src.config.enum_constants import QOSEnum, AdminPartitionEnum, AdminsAccountEnum, PartitionTypeEnum, StatusEnum
from src.config.remote_config import PartitionInfoFetcher


def preprocess_mock_data(
    db_path: str,
    table_name: str = "Jobs",
    min_elapsed_seconds: int = 0,
    include_cpu_only_jobs: bool = False,
    include_custom_qos_jobs: bool = False,
    include_failed_cancelled_jobs: bool = False,
) -> pd.DataFrame:
    """
    Helper function to filter job records from database based on various criteria.

    This function applies the same filtering logic as the preprocessing pipeline
    to create a ground truth dataset for testing purposes. It filters out jobs
    based on elapsed time, account type, partition type, QOS values, and status.

    Args:
        db_path (str): Path to the DuckDB database file.
        table_name (str, optional): Name of the table to query. Defaults to "Jobs".
        min_elapsed_seconds (int, optional): Minimum elapsed time in seconds to filter jobs.
            Jobs with elapsed time below this threshold are excluded. Defaults to 0.
        include_cpu_only_jobs (bool, optional): If True, include jobs that run on CPU-only
            partitions. If False, only include jobs from GPU partitions. Defaults to False.
        include_custom_qos_jobs (bool, optional): If True, include jobs with custom QOS values
            (not in the standard QOS enum). If False, only include jobs with standard QOS.
            Defaults to False.
        include_failed_cancelled_jobs (bool, optional): If True, include jobs with FAILED
            or CANCELLED status. If False, exclude these jobs. Defaults to False.

    Returns:
        pd.DataFrame: Filtered DataFrame containing job records that meet the specified criteria.

    Raises:
        Exception: If there's an error during database operations or query execution.

    Note:
        This function is used in tests to create expected results for comparison with
        the actual pipeline output. It excludes jobs with:
        - Root account
        - Building partition
        - Updates QOS
        - Smaller elapsed time than min_elapsed_seconds
        And applies additional filters based on the provided parameters.
    """
    qos_values = "(" + ",".join(f"'{obj.value}'" for obj in QOSEnum) + ")"

    # get cpu partition list
    partition_info = PartitionInfoFetcher().get_info()
    gpu_partitions = [p["name"] for p in partition_info if p["type"] == PartitionTypeEnum.GPU.value]
    gpu_partitions_str = "(" + ",".join(f"'{partition_name}'" for partition_name in gpu_partitions) + ")"
    mem_db = None
    try:
        mem_db = DatabaseConnection(
            db_path
        )  # with read_only = True as we don't expect to write into database directly from tests

        conditions_arr = [
            f"Elapsed >= {min_elapsed_seconds}",
            f"Account != '{AdminsAccountEnum.ROOT.value}'",
            f"Partition != '{AdminPartitionEnum.BUILDING.value}'",
            f"QOS != '{QOSEnum.UPDATES.value}'",
        ]
        if not include_custom_qos_jobs:
            conditions_arr.append(f"QOS in {qos_values}")
        if not include_cpu_only_jobs:
            conditions_arr.append(f"Partition IN {gpu_partitions_str}")
        if not include_failed_cancelled_jobs:
            conditions_arr.append(f"Status != '{StatusEnum.FAILED.value}'")
            conditions_arr.append(f"Status != '{StatusEnum.CANCELLED.value}'")

        query = f"SELECT * FROM {table_name} WHERE {' AND '.join(conditions_arr)}"
        return mem_db.fetch_query(query=query)
    except Exception as e:
        raise Exception("Exception at helper_filter_irrelevant_records") from e
    finally:
        if mem_db is not None:
            mem_db.disconnect()


# Get path to the temporary mock database file
@pytest.fixture(scope="module")
def mock_data_path(request: pytest.FixtureRequest) -> Generator[str]:
    try:
        is_new_format = request.param
        temp_db_dir = tempfile.mkdtemp()
        temp_db_path = f"{temp_db_dir}/mock_new_format.db" if is_new_format else f"{temp_db_dir}/mock.db"
        csv_path = "tests/mock_data/mock_new_format.csv" if is_new_format else "tests/mock_data/mock.csv"
        convert_csv_to_db(csv_path, temp_db_path, new_format=is_new_format)
        yield temp_db_path
    finally:
        shutil.rmtree(temp_db_dir)


# load mock database as a Dataframe
@pytest.fixture(scope="module")
def mock_data_frame(mock_data_path: str) -> Generator[pd.DataFrame]:
    mem_db = None
    try:
        mem_db = DatabaseConnection(
            mock_data_path
        )  # with read_only = True as we don't expect to write into database directly from tests
        yield mem_db.fetch_all_jobs()
    except Exception as e:
        raise Exception("Exception at mock_data_frame") from e
    finally:
        if mem_db is not None:
            mem_db.disconnect()
