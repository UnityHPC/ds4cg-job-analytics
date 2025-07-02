import pytest
from src.database.database_connection import DatabaseConnection
import tempfile
import os


@pytest.fixture
def temp_file_db():
    """Create a temporary file-based database for testing."""
    fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(temp_db_path)
    print(f"Database file will be created at: {temp_db_path}")

    mem_db = DatabaseConnection(db_url=temp_db_path)
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
    mem_db.connection.execute(schema_sql)
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
    mem_db.connection.execute(insert_sql)

    yield mem_db

    os.remove(temp_db_path)


def test_connection_established(temp_file_db):
    assert temp_file_db.is_connected() is True


def test_fetch_all_returns_correct_data(temp_file_db):
    mock_jobs_df = temp_file_db.fetch_all_jobs()

    assert len(mock_jobs_df) == 3

    assert mock_jobs_df.iloc[0]["JobID"] == 101
    assert mock_jobs_df.iloc[0]["User"] == "alice"
    assert mock_jobs_df.iloc[0]["GPUs"] == 1
    assert mock_jobs_df.iloc[0]["Status"] == "COMPLETED"

    assert mock_jobs_df.iloc[1]["JobID"] == 102
    assert mock_jobs_df.iloc[1]["User"] == "bob"
    assert mock_jobs_df.iloc[1]["GPUs"] == 1
    assert mock_jobs_df.iloc[1]["Status"] == "FAILED"

    assert mock_jobs_df.iloc[2]["JobID"] == 103
    assert mock_jobs_df.iloc[2]["User"] == "chris"
    assert mock_jobs_df.iloc[2]["GPUs"] == 1
    assert mock_jobs_df.iloc[2]["Status"] == "OUT_OF_MEMORY"


def test_fetch_selected_columns_with_filter(temp_file_db):
    query = """
        SELECT JobID, User
        FROM Jobs
        WHERE Status = 'COMPLETED'
    """
    mock_jobs_df = temp_file_db.connection.execute(query).fetchdf()

    assert len(mock_jobs_df) == 1
    assert list(mock_jobs_df.columns) == ["JobID", "User"]
    assert mock_jobs_df.iloc[0]["JobID"] == 101
    assert mock_jobs_df.iloc[0]["User"] == "alice"


def test_fetch_with_filtering_multiple_conditions(temp_file_db):
    query = """
        SELECT JobID, User
        FROM Jobs
        WHERE Status = 'COMPLETED' AND GPUs = 1
    """
    mock_jobs_df = temp_file_db.connection.execute(query).fetchdf()

    assert len(mock_jobs_df) == 1
    assert list(mock_jobs_df.columns) == ["JobID", "User"]
    assert mock_jobs_df.iloc[0]["JobID"] == 101
    assert mock_jobs_df.iloc[0]["User"] == "alice"


def test_fetch_all_column_names(temp_file_db):
    column_names = temp_file_db.fetch_all_column_names()

    assert len(column_names) == 29

    assert "JobID" in column_names
    assert "User" in column_names
    assert "Status" in column_names
    assert "GPUs" in column_names


def test_fetch_query_with_invalid_column(temp_file_db):
    query = """
        SELECT GPUMetrics
        FROM Jobs
    """

    with pytest.raises(Exception) as exc_info:
        temp_file_db.fetch_query(query)
    msg = str(exc_info.value)
    assert "This query does not match the database schema." in msg
