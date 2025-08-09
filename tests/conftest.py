import pytest
from src.database import DatabaseConnection
from .mock_data.convert_csv_to_db import convert_csv_to_db
import tempfile
import shutil
from src.config.enum_constants import QOSEnum, AdminPartitionEnum, AdminsAccountEnum, PartitionTypeEnum, StatusEnum
from src.config.remote_config import PartitionInfoFetcher
import pandas as pd


def helper_filter_irrelevant_records(
    db_path: str,
    table_name: str = "Jobs",
    min_elapsed_seconds: int = 0,
    include_cpu_only_jobs: bool = False,
    include_custom_qos: bool = False,
    include_failed_cancelled_jobs: bool = False,
) -> pd.DataFrame:
    qos_values = "(" + ",".join(f"'{obj.value}'" for obj in QOSEnum) + ")"
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
        if not include_custom_qos:
            conditions_arr.append(f"QOS in {qos_values}")
        if not include_cpu_only_jobs:
            conditions_arr.append(f"Partition IN {gpu_partitions_str}")
        if not include_failed_cancelled_jobs:
            conditions_arr.append(f"Status != '{StatusEnum.FAILED.value}'")
            conditions_arr.append(f"Status != '{StatusEnum.CANCELLED.value}'")

        query = f"SELECT * FROM {table_name} WHERE {' AND '.join(conditions_arr)}"
        print(f"Query: {query}")
        return mem_db.fetch_query(query=query)
    except Exception as e:
        raise Exception("Exception at helper_filter_irrelevant_records") from e


# fixture for returning path to temporary db
@pytest.fixture(scope="module")
def mock_data_path():
    try:
        temp_db_dir = tempfile.mkdtemp()
        temp_db_path = f"{temp_db_dir}/mock.db"
        convert_csv_to_db("tests/mock_data/mock.csv", temp_db_path)
        yield temp_db_path
    finally:
        shutil.rmtree(temp_db_dir)


# fixture for returning mock_data_frame
@pytest.fixture(scope="module")
def mock_data_frame(mock_data_path):
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
            mem_db._disconnect()
            del mem_db
