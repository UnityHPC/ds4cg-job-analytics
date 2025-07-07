import pytest
<<<<<<< HEAD
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add src to sys.path for module imports
sys.path.append(str(Path.cwd() / "src"))
from database.DatabaseConnection import DatabaseConnection


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
=======
from src.database import DatabaseConnection
from .mock_data.convert_csv_to_db import convert_csv_to_db
import tempfile
import shutil


@pytest.fixture(scope="module")
def mock_data_frame():
    temp_db_dir = tempfile.mkdtemp()
    db = None
    try:
        temp_db_path = f"{temp_db_dir}/mock.db"
        convert_csv_to_db("tests/mock_data/mock.csv", temp_db_path)
        db = DatabaseConnection(temp_db_path)
        yield db.fetch_all_jobs()
    except Exception as e:
        raise Exception("Exception at mock_data_frame") from e
    finally:
        if db is not None and db.is_connected():
            db.connection.close()
            shutil.rmtree(temp_db_dir)
>>>>>>> fix/vram-calculations
