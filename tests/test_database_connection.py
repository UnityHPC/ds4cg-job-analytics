import pytest
import duckdb
import pandas as pd
from src.database.DatabaseConnection import DatabaseConnection


@pytest.fixture
def in_memory_db():
    # Create a database connection object using :memory:
    db = DatabaseConnection(db_url=":memory:")
    
    # Define schema
    schema_sql = """
    CREATE TABLE Jobs (
        UUID VARCHAR,
        JobID INTEGER,
        ArrayID INTEGER,
        JobName VARCHAR,
        IsArray BOOLEAN,
        Interactive VARCHAR,
        Preempted BOOLEAN,
        Account VARCHAR,
        User VARCHAR,
        Constraints VARCHAR[],
        QOS VARCHAR,
        Status VARCHAR,
        ExitCode VARCHAR,
        SubmitTime TIMESTAMP,
        StartTime TIMESTAMP,
        EndTime TIMESTAMP,
        Elapsed INTEGER,
        TimeLimit INTEGER,
        Partition VARCHAR,
        Nodes VARCHAR,
        NodeList VARCHAR[],
        CPUs SMALLINT,
        Memory BIGINT,
        GPUs SMALLINT,
        GPUType VARCHAR[],
        GPUMemUsage BIGINT,
        GPUComputeUsage FLOAT,
        CPUMemUsage BIGINT,
        CPUComputeUsage FLOAT
    );
    """
    db.connection.execute(schema_sql)

    # Insert sample row
    insert_sql = """
    INSERT INTO Jobs VALUES (
        'abc-123', 101, NULL, 'train_model', FALSE, 'yes', FALSE,
        'projectX', 'alice', ['A100'], 'normal', 'COMPLETED', '0:0',
        CURRENT_TIMESTAMP - INTERVAL 3 HOUR,
        CURRENT_TIMESTAMP - INTERVAL 2 HOUR,
        CURRENT_TIMESTAMP - INTERVAL 1 HOUR,
        3600, 7200, 'gpu', 'node001', ['node001'],
        16, 64000000000, 1, ['A100'], 12000000000, 0.85, 32000000000, 0.75
    );
    """
    db.connection.execute(insert_sql)

    return db


def test_fetch_all_returns_correct_data(in_memory_db):
    df = in_memory_db.fetch_all()

    # Check that one row was returned
    assert len(df) == 1

    # Check specific values
    assert df.iloc[0]["JobID"] == 101
    assert df.iloc[0]["User"] == "alice"
    assert df.iloc[0]["GPUs"] == 1
    assert df.iloc[0]["Status"] == "COMPLETED"

def test_fetch_selected_columns_with_filter(in_memory_db):
    # Perform a custom query using the connection
    query = """
        SELECT JobID, User
        FROM Jobs
        WHERE Status = 'COMPLETED'
    """
    df = in_memory_db.connection.execute(query).fetchdf()

    # Assertions
    assert len(df) == 1
    assert list(df.columns) == ["JobID", "User"]
    assert df.iloc[0]["JobID"] == 101
    assert df.iloc[0]["User"] == "alice"
