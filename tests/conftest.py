import pytest
from src.database import DatabaseConnection
from .mock_data.convert_csv_to_db import convert_csv_to_db
import tempfile
import shutil
from src.config.enum_constants import (
    QOSEnum,
    AdminPartitionEnum,
    AdminsAccountEnum,
)
import pandas as pd


def helper_filter_irrelevant_records(
    input_df: pd.DataFrame, min_elapsed_seconds: int, include_custom_qos=False
) -> pd.DataFrame:
    """
    Helper function to make basic filtering on a ground truth dataframe, intended to use for test only.

    Given a ground truth dataframe, this will create a new dataframe without records meeting the following criteria:
    - QOS is updates, or contains custom values if include_custom_qos is False
    - Account is root
    - Partition is building
    - Elasped time is less than min_elapsed

    Args:
        input_df (pd.DataFrame): Input dataframe to filter. Note that the Elapsed field should be in unit seconds.
        min_elapsed_seconds (int): Minimum elapsed time in seconds.
        include_custom_qos (bool): condition on whether to include records with customized QOS values or not

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    qos_values = [e.value for e in QOSEnum]
    res = input_df[
        (input_df["Elapsed"] >= min_elapsed_seconds)
        & (input_df["Account"] != AdminsAccountEnum.ROOT.value)
        & (input_df["Partition"] != AdminPartitionEnum.BUILDING.value)
        & (input_df["QOS"] != QOSEnum.UPDATES.value)
        & ((input_df["QOS"].isin(qos_values)) | include_custom_qos)
    ]
    return res


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
        mem_db = DatabaseConnection(mock_data_path, read_only=False)
        yield mem_db.fetch_all_jobs()
    except Exception as e:
        raise Exception("Exception at mock_data_frame") from e
    finally:
        if mem_db is not None:
            mem_db._disconnect()
            del mem_db
