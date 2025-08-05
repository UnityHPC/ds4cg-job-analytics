import pytest
import tempfile
import shutil
from src.database import DatabaseConnection
from .mock_data.convert_csv_to_db import convert_csv_to_db


@pytest.fixture(scope="module")
def mock_data_frame(request):
    temp_db_dir = tempfile.mkdtemp()
    mem_db = None
    try:
        is_new_format = request.param
        temp_db_path = f"{temp_db_dir}/mock_new_format.db" if is_new_format else f"{temp_db_dir}/mock.db"
        csv_path = "tests/mock_data/mock_new_format.csv" if is_new_format else "tests/mock_data/mock.csv"
        convert_csv_to_db(csv_path, temp_db_path, new_format=is_new_format)
        mem_db = DatabaseConnection(temp_db_path)
        yield mem_db.fetch_all_jobs()
    except Exception as e:
        raise Exception("Exception at mock_data_frame") from e
    finally:
        if mem_db is not None:
            mem_db.disconnect()
            del mem_db
        shutil.rmtree(temp_db_dir)
