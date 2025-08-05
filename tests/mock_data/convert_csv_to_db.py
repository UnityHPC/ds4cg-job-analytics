import duckdb
import pandas as pd


def convert_csv_to_db(path_to_csv: str, path_to_db: str):
    """
    Function to convert csv to duckDB database, following the schema provided by Unity.

    This function is intended to be used on csv files that follow the Unity schema only.

    Args:
        path_to_csv (str): Path to the CSV file containing job data.
        path_to_db (str): Path to the DuckDB database file where the data will be stored.

    Returns:
        None: The function creates a DuckDB database and populates it with data from the CSV
    """
    conn = None
    try:
        conn = duckdb.connect(path_to_db)
        df_mock = pd.read_csv(path_to_csv)

        # Convert some columns in csv to correct data types as specified in the Unity schema
        for col in ["SubmitTime", "StartTime", "EndTime"]:
            df_mock[col] = pd.to_datetime(df_mock[col], format="%m/%d/%y %H:%M")
            df_mock[col] = df_mock[col].astype("datetime64[ns]")

        for col in ["GPUMemUsage", "GPUComputeUsage", "CPUMemUsage", "CPUComputeUsage"]:
            df_mock[col] = pd.to_numeric(df_mock[col], errors="coerce")
            df_mock[col] = df_mock[col].astype("float64")

        conn.execute(
            """
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
                SubmitTime TIMESTAMP_NS,
                StartTime TIMESTAMP_NS,
                EndTime TIMESTAMP_NS,
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
            );"""
        )
        conn.register("df_view", df_mock)
        conn.execute("INSERT INTO Jobs SELECT * FROM df_view")
    except Exception as e:
        raise e
    finally:
        if conn is not None:
            conn.close()