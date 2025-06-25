import pytest
from src.database import DatabaseConnection
from .mockData.convert_csv_to_db import convert_csv_to_db
import os
# @pytest.fixture
# def load_mock_data_1():
#     db = DatabaseConnection("tests/mockData/mock1.db")
#     return db.fetch_all()


# @pytest.fixture
# def load_mock_data_2():
#     db = DatabaseConnection("tests/mockData/mock2.db")
#     return db.fetch_all()


@pytest.fixture(scope="module")
def load_mock_data():
    convert_csv_to_db("tests/mockData/mock.csv", "tests/mockData/mock.db")
    db = DatabaseConnection("tests/mockData/mock.db")
    yield db.fetch_all()
    os.remove("tests/mockData/mock.db")
