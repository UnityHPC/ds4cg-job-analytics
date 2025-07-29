import pytest
from src.database import DatabaseConnection
from .mock_data.convert_csv_to_db import convert_csv_to_db
import tempfile
import shutil
import os


@pytest.fixture(scope="module")
def mock_data_frame():
    temp_db_dir = tempfile.mkdtemp()
    db = None
    try:
        temp_db_path = f"{temp_db_dir}/mock.db"
        convert_csv_to_db("tests/mock_data/mock.csv", temp_db_path)
        mem_db = DatabaseConnection(temp_db_path)
        yield mem_db.fetch_all_jobs()
    except Exception as e:
        raise Exception("Exception at mock_data_frame") from e
    finally:
        del db
        print("removing tree")
        print(temp_db_dir)
        shutil.rmtree(temp_db_dir)
        # print("removed tree")
        # if os.path.exists(temp_db_dir):
        #     raise Exception("Temporary directory was not removed properly")
