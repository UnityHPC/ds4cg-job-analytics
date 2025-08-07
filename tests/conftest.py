import pytest
from src.database import DatabaseConnection
from .mock_data.convert_csv_to_db import convert_csv_to_db
import tempfile
import shutil
from src.config.enum_constants import QOSEnum, AdminPartitionEnum, AdminsAccountEnum, PartitionTypeEnum
from src.config.remote_config import PartitionInfoFetcher
import pandas as pd


def helper_filter_irrelevant_records(
    input_df: pd.DataFrame,
    min_elapsed_seconds: int,
    include_cpu_only_jobs: bool = False,
    include_custom_qos: bool = False,
) -> pd.DataFrame:
    """
    Private function to help generate expected ground truth dataframe for test.

    Given a ground truth dataframe, this will create a new dataframe without records meeting the following criteria:
    - QOS is updates
    - Account is root
    - Partition is building
    - Elasped time is less than min_elapsed

    Args:
        input_df (pd.DataFrame): Input dataframe to filter. Note that the Elapsed field should be in unit seconds.
        min_elapsed_seconds (int): Minimum elapsed time in seconds.
        include_cpu_only_jobs (bool): Whether to include jobs that do not use GPUs (CPU-only jobs). Default is False.
        include_custom_qos (bool): condition on whether to include records with customized QOS values or not


    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    qos_values = set([member.value for member in QOSEnum])

    # TODO(Tan): Update implementation to use the same logic as preprocess_data
    mask = pd.Series([True] * len(input_df), index=input_df.index)

    mask &= input_df["Elapsed"] >= min_elapsed_seconds
    mask &= input_df["Account"] != AdminsAccountEnum.ROOT.value
    mask &= input_df["Partition"] != AdminPartitionEnum.BUILDING.value
    mask &= (input_df["QOS"] != QOSEnum.UPDATES.value) & (include_custom_qos | input_df["QOS"].isin(qos_values))
    # Filter out jobs whose partition type is not 'gpu', unless include_cpu_only_jobs is True.
    partition_info = PartitionInfoFetcher().get_info()
    gpu_partitions = [p["name"] for p in partition_info if p["type"] == PartitionTypeEnum.GPU.value]
    mask &= input_df["Partition"].isin(gpu_partitions) | include_cpu_only_jobs

    return input_df[mask].copy()


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
