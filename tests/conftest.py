import pytest
from src.database import DatabaseConnection
from .mockData.convert_csv_to_db import convert_csv_to_db
import os


# TODO: no camel case for directory name
# TODO: use temp file to store the mock data db
@pytest.fixture(scope="module")
def mock_data_frame():
    convert_csv_to_db("tests/mockData/mock.csv", "tests/mockData/mock.db")
    db = DatabaseConnection("tests/mockData/mock.db")
    yield db.fetch_all()
    os.remove("tests/mockData/mock.db")
