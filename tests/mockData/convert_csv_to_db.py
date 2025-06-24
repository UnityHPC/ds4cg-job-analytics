import duckdb
import pandas as pd


# TODO: clean private data in csv
# TODO: resetup the conftest such that it will generate a mock db and then will destroy it after test
# TODO: readjust the test properly bc you're having new data, it is best to come up with some automatic way of testing instead of harcode values
# Use this script to convert csv into a duckdb database
def convert_csv_to_db(path_to_csv: str, path_to_db: str):
    conn = duckdb.connect(path_to_db)
    df = pd.read_csv(path_to_csv)
    for col in ["SubmitTime", "StartTime", "EndTime"]:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%y %H:%M")
        df[col] = df[col].astype("datetime64[ns]")

    for col in ["CPUMemUsage", "GPUComputeUsage", "CPUMemUsage", "CPUComputeUsage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].astype("float64")

    conn.execute("""
    DROP TABLE IF EXISTS Jobs;
""")
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
    conn.register("df_view", df)
    conn.execute("INSERT INTO Jobs SELECT * FROM df_view")
    conn.close()
