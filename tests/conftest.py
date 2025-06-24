import pytest
import pandas as pd
import numpy as np
from src.database import DatabaseConnection
from mockData.convert_csv_to_db import convert_csv_to_db

# @pytest.fixture
# def load_mock_data_1():
#     db = DatabaseConnection("tests/mockData/mock1.db")
#     return db.fetch_all()


# @pytest.fixture
# def load_mock_data_2():
#     db = DatabaseConnection("tests/mockData/mock2.db")
#     return db.fetch_all()


@pytest.fixture
def load_mock_data():
    convert_csv_to_db("tests/mockData/mock_data.csv", "tests/mockData/mock_data.db")
    db = DatabaseConnection("tests/mockData/mock_data.db")
    return db.fetch_all()


@pytest.fixture
def small_sample_data():
    data = {
        "JobName": ["job1", "job2", "job3", "job4"],
        "UUID": ["123456789", "123456789", "123456789", "123456789"],
        "ArrayID": [np.nan, 1, 2, np.nan],
        "Interactive": [np.nan, "Matlab", np.nan, "Matlab"],
        "Constraints": [np.nan, np.array(["some constraints"]), np.nan, np.array(["some constraints"])],
        "GPUType": [np.nan, np.array(["v100"]), np.nan, np.array(["v100"])],
        "GPUs": [np.nan, 1, np.nan, 4],
    }
    df = pd.DataFrame(data)
    return df
