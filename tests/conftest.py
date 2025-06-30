import pytest
from src.database import DatabaseConnection
from .mock_data.convert_csv_to_db import convert_csv_to_db
import tempfile
import shutil
from src.config.enum_constants import (
    QOSEnum,
    PartitionEnum,
    AdminsAccountEnum,
)
import pandas as pd


def helper_filter_irrelevant_records(input_df: pd.DataFrame, min_elapsed_seconds: int) -> pd.DataFrame:
    """
    Given a ground truth dataframe, this will create a new dataframe without records meeting the following criteria:
    - QOS is updates
    - Account is root
    - Partition is building
    - Elasped time is less than min_elapsed

    Args:
        input_df (pd.DataFrame): Input dataframe to filter. Note that the Elapsed field should be in unit seconds.
        min_elapsed_seconds (int): Minimum elapsed time in seconds.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """

    res = input_df[
        (input_df["Elapsed"] >= min_elapsed_seconds)
        & (input_df["Account"] != AdminsAccountEnum.ROOT.value)
        & (input_df["Partition"] != PartitionEnum.BUILDING.value)
        & (input_df["QOS"] != QOSEnum.UPDATES.value)
    ]
    return res


@pytest.fixture(scope="module")
def mock_data():
    temp_db_dir = tempfile.mkdtemp()
    try:
        temp_db_path = f"{temp_db_dir}/mock.db"
        convert_csv_to_db("tests/mock_data/mock.csv", temp_db_path)
        db = DatabaseConnection(temp_db_path)
        yield db.fetch_all(), temp_db_path
    except Exception as e:
        raise Exception("Exception at mock_data_frame") from e
    finally:
        shutil.rmtree(temp_db_dir)
