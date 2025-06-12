import duckdb 


class DatabaseConnection:
    def __init__(self, db_url: str ):
        self.db_url = db_url
        self.connection = None
        self.connect()
        

    def connect(self):
        # Simulate a database connection
        self.connection = duckdb.connect(self.db_url)
        print(f"Connected to {self.db_url}")
        return self.connection
    def as_dataframe(self):
        """Return the dataframe containing job metrics."""
        return self.df
    def disconnect(self):
        # Simulate closing the database connection
        self.connection.close()
        self.connection = None

    def is_connected(self) -> bool:
        return self.connection is not None

    def get_connection_info(self) -> str:
        return self.connection if self.is_connected() else "No active connection"
    def fetch_all(self):
        """Fetch all data from the Jobs table."""
        if self.is_connected():
            return self.con.query(
            "select GPUs, GPUMemUsage, GPUComputeUsage, GPUType, Elapsed, "
            "StartTime,"
            "StartTime-SubmitTime as Queued, TimeLimit, Interactive, "
            "IsArray, JobID, ArrayID, Status, Constraints, Partition, User, Account from Jobs "
            f"where GPUs > 0  and GPUType is not null "
            " and Status != 'CANCELLED' and Status != 'FAILED'"
        ).to_df()
        else:
            raise Exception("No active database connection.")
    def fetch_query(self, query: str):
        """Fetch data based on a custom query."""
        if self.is_connected():
            return self.connection.query(query).to_df()
        else:
            raise Exception("No active database connection.")


