import pytest
from src.database import DatabaseConnection
from .mock_data.convert_csv_to_db import convert_csv_to_db
import os
import tempfile


# TODO: use temp file to store the mock data db
@pytest.fixture(scope="module")
def mock_data_frame():
    fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(temp_db_path)
    try:
        convert_csv_to_db("tests/mock_data/mock.csv", temp_db_path)
        db = DatabaseConnection(temp_db_path)
        yield db.fetch_all()
    except Exception as e:
        raise Exception("Exception at mock_data_frame") from e
    finally:
        os.remove(temp_db_path)
