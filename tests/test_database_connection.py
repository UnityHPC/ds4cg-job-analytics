import pytest 
from analytics.DatabaseConnection import DatabaseConnection
import pandas as pd
import os 
def createDBConnection() -> DatabaseConnection:
    return DatabaseConnection()

@pytest.fixture
def test_db_connection():
    db_conn = createDBConnection()
    assert db_conn.is_connected() == True
    assert isinstance(db_conn.as_dataframe(), pd.DataFrame)
    db_conn.disconnect()
    assert db_conn.is_connected() == False
    assert db_conn.get_connection_info() == "No active connection"
