import duckdb 


class DatabaseConnection:
    def __init__(self, db_url: str ):
        self.db_url = db_url
        self.connection = None
        self.connection = self.connect()
        

    def connect(self):
        # Simulate a database connection
        self.connection = duckdb.connect(self.db_url)
        print(f"Connected to {self.db_url}")
        return self.connection
    
    def disconnect(self):
        # Simulate closing the database connection
        self.connection.close()
        self.connection = None

    def is_connected(self) -> bool:
        return self.connection is not None

    def get_connection_info(self) -> str:
        return self.connection if self.is_connected() else "No active connection"
    def fetch_all(self):
        if self.is_connected():
            query = """
                SELECT UUID, JobID, ArrayID, JobName, IsArray, Interactive,
                       Preempted, Account, User, Constraints, QOS, Status,
                       ExitCode, SubmitTime, StartTime, EndTime, Elapsed,
                       TimeLimit, Partition, Nodes, NodeList, CPUs, Memory,
                       GPUs, GPUType, GPUMemUsage, GPUComputeUsage,
                       CPUMemUsage, CPUComputeUsage
                FROM Jobs
            """
            return self.connection.execute(query).fetchdf()
        else:
            raise Exception("Not connected")
    def fetch_query(self, query: str):
        """Fetch data based on a custom query."""
        if self.is_connected():
            return self.connection.query(query).to_df()
        else:
            raise Exception("No active database connection.")
