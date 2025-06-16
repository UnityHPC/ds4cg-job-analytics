import pytest
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
        Memory INTEGER,
        GPUs SMALLINT,
        GPUType VARCHAR[],
        GPUMemUsage FLOAT,
        GPUComputeUsage FLOAT,
        CPUMemUsage FLOAT,
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
        16, 640, 1, ['A100'], 120, 0.85, 320, 0.75
    );
    INSERT INTO Jobs VALUES (
        'xyz-215', 102, NULL, 'train_model', FALSE, 'yes', FALSE,
        'projectX', 'bob', ['A100'], 'normal', 'FAILED', '0:0',
        CURRENT_TIMESTAMP - INTERVAL 3 HOUR,
        CURRENT_TIMESTAMP - INTERVAL 2 HOUR,
        CURRENT_TIMESTAMP - INTERVAL 1 HOUR,
        3600, 7200, 'gpu', 'node001', ['node001'],
        16, 640, 1, ['M40'], 120, 0.85, 320, 0.75
    );

    INSERT INTO Jobs VALUES (
        'xyz-217', 103, NULL, 'train_model', FALSE, 'yes', FALSE,
        'projectX', 'chris', ['A100'], 'normal', 'OUT_OF_MEMORY', '0:0',
        CURRENT_TIMESTAMP - INTERVAL 3 HOUR,
        CURRENT_TIMESTAMP - INTERVAL 2 HOUR,
        CURRENT_TIMESTAMP - INTERVAL 1 HOUR,
        3600, 7200, 'gpu', 'node001', ['node001'],
        16, 640, 1, ['M40'], 120, 0.85, 320, 0.75
    );
    """
    db.connection.execute(insert_sql)

    return db


def test_connection_established(in_memory_db):
    # Check if the connection is established
    assert in_memory_db.is_connected() is True


def test_fetch_all_returns_correct_data(in_memory_db):
    df = in_memory_db.fetch_all()

    assert len(df) == 3

    # Check specific values
    assert df.iloc[0]["JobID"] == 101
    assert df.iloc[0]["User"] == "alice"
    assert df.iloc[0]["GPUs"] == 1
    assert df.iloc[0]["Status"] == "COMPLETED"

    assert df.iloc[1]["JobID"] == 102
    assert df.iloc[1]["User"] == "bob"
    assert df.iloc[1]["GPUs"] == 1
    assert df.iloc[1]["Status"] == "FAILED"

    assert df.iloc[2]["JobID"] == 103
    assert df.iloc[2]["User"] == "chris"
    assert df.iloc[2]["GPUs"] == 1
    assert df.iloc[2]["Status"] == "OUT_OF_MEMORY"


def test_fetch_selected_columns_with_filter(in_memory_db):
    query = """
        SELECT JobID, User
        FROM Jobs
        WHERE Status = 'COMPLETED'
    """
    df = in_memory_db.connection.execute(query).fetchdf()

    assert len(df) == 1
    assert list(df.columns) == ["JobID", "User"]
    assert df.iloc[0]["JobID"] == 101
    assert df.iloc[0]["User"] == "alice"


def test_fetch_with_filtering_multiple_conditions(in_memory_db):
    query = """
        SELECT JobID, User
        FROM Jobs
        WHERE Status = 'COMPLETED' AND GPUs = 1
    """
    df = in_memory_db.connection.execute(query).fetchdf()

    assert len(df) == 1
    assert list(df.columns) == ["JobID", "User"]
    assert df.iloc[0]["JobID"] == 101
    assert df.iloc[0]["User"] == "alice"

def test_fetch_all_column_names(in_memory_db):
    column_names = in_memory_db.fetch_all_column_names()

    assert len(column_names) == 29 

    assert "JobID" in column_names
    assert "User" in column_names
    assert "Status" in column_names
    assert "GPUs" in column_names
    
def test_fetch_query_with_invalid_column(in_memory_db):
    query = """
        SELECT GPUMetrics
        FROM Jobs
    """
    with pytest.raises(Exception) as exc_info:
        in_memory_db.fetch_query(query)
    msg = str(exc_info.value)
    assert "Invalid query or column names" in msg
