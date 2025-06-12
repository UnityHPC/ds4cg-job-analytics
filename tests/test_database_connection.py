import pytest 
from src.analytics.DatabaseConnection import DatabaseConnection
import pandas as pd
def createDBConnection(path) -> DatabaseConnection:
    return DatabaseConnection(path)

#@pytest.fixture
def test_db_connection(path):
    db_conn = createDBConnection(path)
    assert db_conn.is_connected() == True
    assert isinstance(db_conn.as_dataframe(), pd.DataFrame)
    db_conn.disconnect()
    assert db_conn.is_connected() == False
    assert db_conn.get_connection_info() == "No active connection"
