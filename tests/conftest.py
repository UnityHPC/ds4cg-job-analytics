import pytest
from src.database import DatabaseConnection
from .mock_data.convert_csv_to_db import convert_csv_to_db
import tempfile
import shutil


@pytest.fixture(scope="module")
def mock_data_frame():
    temp_db_dir = tempfile.mkdtemp()
    try:
        temp_db_path = f"{temp_db_dir}/mock.db"
        convert_csv_to_db("tests/mock_data/mock.csv", temp_db_path)
        db = DatabaseConnection(temp_db_path)
        yield db.fetch_all()
    except Exception as e:
        raise Exception("Exception at mock_data_frame") from e
    finally:
        shutil.rmtree(temp_db_dir)
