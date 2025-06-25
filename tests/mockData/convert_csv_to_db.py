import duckdb
import pandas as pd


# Use this script to convert csv into a duckdb database
def convert_csv_to_db(path_to_csv: str, path_to_db: str):
    conn = duckdb.connect(path_to_db)
    df_mock = pd.read_csv(path_to_csv)

    # TODO: add comment to explain these type casting do
    for col in ["SubmitTime", "StartTime", "EndTime"]:
        df_mock[col] = pd.to_datetime(df_mock[col], format="%m/%d/%y %H:%M")
        df_mock[col] = df_mock[col].astype("datetime64[ns]")

    for col in ["CPUMemUsage", "GPUComputeUsage", "CPUMemUsage", "CPUComputeUsage"]:
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
            GPUMemUsage BIGINT,
            GPUComputeUsage FLOAT,
            CPUMemUsage BIGINT,
            CPUComputeUsage FLOAT
        );"""
    )
    conn.register("df_view", df_mock)
    conn.execute("INSERT INTO Jobs SELECT * FROM df_view")
    conn.close()
