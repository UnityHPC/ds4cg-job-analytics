import duckdb
import os


class DatabaseConnection:
    """
    A class to manage database connections using DuckDB.

    This class provides methods to establish a connection to a DuckDB database using a given URL. It can be used to interact with the database and perform operations.

    """
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.connection = self._connect()
        print(f"Connected to {self.db_url}")

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Establish a connection to the DuckDB database.

        Returns: duckdb.DuckDBPyConnection: The connection object to the DuckDB database.
        """
        self.connection = duckdb.connect(self.db_url)
        return self.connection

    def _disconnect(self):
        """Safely closes the active database connection"""
        self.connection.close()

    def __del__(self):
        """Ensure the connection is closed when the object is deleted."""
        if self.is_connected():
            self._disconnect()
            if not os.getenv("PYTEST_VERSION"):
                print(f"Disconnected from {self.db_url}")

    def is_connected(self) -> bool:
        """
        Checks if the database connection is active.

        Returns: bool: True if the connection is active, False otherwise
        """
        return self.connection is not None

    def get_connection_info(self) -> str:
        """
        Retrieves all information about the connection.

        Returns: str:
        """
        return self.connection if self.is_connected() else "No active connection"

    def fetch_all_column_names(self, table_name: str = "Jobs"):
        """Fetch all column names from the any table. By default, it fetches from a table named 'Jobs'.
        
        Args: table_name (str): The name of the table to fetch all the column names from. Defaults to "Jobs"      
        """
        if self.is_connected():
            query = f"SELECT * FROM {table_name} LIMIT 0"
            return self.connection.execute(query).df().columns.tolist()
        else:
            raise Exception("Not connected")

    def fetch_all_jobs(self, table_name="Jobs"):
        """Fetch all data from the specified table. Table name is set to Jobs but can be changed accordingly.

        args:
            table_name (str): The name of the table to fetch data from. Defaults to "Jobs".

        Raises:
            Exception: If the connection is not active.
            
        Returns:
            pd.DataFrame: A pandas DataFrame containing all rows from the specified table.
        """
        if self.is_connected():
            query = f"SELECT * FROM {table_name}"
            return self.connection.execute(query).fetchdf()
        else:
            raise Exception("Not connected")

    def fetch_query(self, query: str):
        """
        Fetch data based on a custom query.

        Args:
            query (str): The SQL query to execute.

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
