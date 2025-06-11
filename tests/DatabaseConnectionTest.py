import pytest 
from analytics.DatabaseConnection import DatabaseConnection
import pandas as pd
def createDBConnection(path: str = "C:/Users/Nitya Karthik A/ds4cg-job-analytics/data/slurm_data_small.db") -> DatabaseConnection:
    return DatabaseConnection(path)

def test_db_connection(path: str = "C:/Users/Nitya Karthik A/ds4cg-job-analytics/data/slurm_data_small.db"):
    db_conn = createDBConnection(path)
    assert db_conn.is_connected() == True
    assert isinstance(db_conn.as_dataframe(), pd.DataFrame)
    db_conn.disconnect()
    assert db_conn.is_connected() == False
    assert db_conn.get_connection_info() == "No active connection"
