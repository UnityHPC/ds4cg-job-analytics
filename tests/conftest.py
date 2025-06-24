import pytest
import pandas as pd
import numpy as np
from src.database import DatabaseConnection


# @pytest.fixture(scope="session")
# def load_big_data():
#     db = DatabaseConnection("data/slurm_data_small.db")
#     return db.fetch_all()


# loading small local data before every test modules run
@pytest.fixture
def load_mock_data_1():
    db = DatabaseConnection("tests/mockData/mock1.db")
    return db.fetch_all()


@pytest.fixture
def load_mock_data_2():
    db = DatabaseConnection("tests/mockData/mock2.db")
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
