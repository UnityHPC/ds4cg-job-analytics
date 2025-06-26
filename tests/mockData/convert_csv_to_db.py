import duckdb
import pandas as pd


# Use this script to convert csv into a duckdb database
def connect(path_to_csv: str, path_to_db: str):
    # Connect to DuckDB (creates file if it doesn't exist)
    conn = duckdb.connect(path_to_db)
    df = pd.read_csv(path_to_csv)
    for col in ["SubmitTime", "StartTime", "EndTime"]:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%y %H:%M")
        df[col] = df[col].astype("datetime64[ns]")

    for col in ["CPUMemUsage", "GPUComputeUsage", "CPUMemUsage", "CPUComputeUsage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].astype("float64")

    # csv_path = os.path.abspath("mock1.csv")
    # Create table directly from CSV with automatic schema detection
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


if __name__ == "__main__":
    connect("tests/mockData/mock2.csv", "tests/mockData/mock2.db")
# obj = duckdb.connect("tests/mockData/mock2.db")
# df = obj.query("SELECT * FROM Jobs").to_df()
# print(df["CPUMemUsage"][4])
# obj.close()
#     connect()
#     obj = duckdb.connect("mock2.db")
#     df = obj.query("select * from Jobs").to_df()
#     ground_truth = pd.read_csv("tests/mockData/mock2.csv")
