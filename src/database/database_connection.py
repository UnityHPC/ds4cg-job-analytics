import duckdb


class DatabaseConnection:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.connection = self.connect()

    def connect(self):
        self.connection = duckdb.connect(self.db_url)
        print(f"Connected to {self.db_url}")
        return self.connection

    def disconnect(self):
        self.connection.close()

    def is_connected(self) -> bool:
        return self.connection is not None

    def get_connection_info(self) -> str:
        return self.connection if self.is_connected() else "No active connection"

    def fetch_all_column_names(self, table_name: str = "Jobs"):
        """Fetch all column names from the Jobs table."""
        if self.is_connected():
            query = f"SELECT * FROM {table_name} LIMIT 0"
            return self.connection.execute(query).df().columns.tolist()
        else:
            raise Exception("Not connected")

    def fetch_all(self, table_name="Jobs"):
        """Fetch all data from the specified table. Table name is set to Jobs but can be changed accordingly."""
        if self.is_connected():
            query = f"SELECT * FROM {table_name}"
            return self.connection.execute(query).fetchdf()
        else:
            raise Exception("Not connected")

    def fetch_query(self, query: str):
        """
        Fetch data based on a custom query.
        
        Args:            query (str): The SQL query to execute.

        Raises:
            Exception: If the query does not match the database schema or if there is no active connection

        Returns:
            pd.DataFrame: The result of the query as a pandas DataFrame.
        """
        if self.is_connected():
            try:
                return self.connection.query(query).to_df()
            except duckdb.BinderException as e:
                valid_columns = self.fetch_all_column_names()
                raise Exception(
                    f"This query does not match the database schema. Valid columns are: {valid_columns}."
                ) from e
        else:
            raise Exception("No active database connection.")
