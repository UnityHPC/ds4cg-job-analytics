import duckdb


class DatabaseConnection:
    """
    A class to manage database connections using DuckDB.

    This class provides methods to establish a connection to a DuckDB database using a given URL. It can be used to interact with the database and perform operations.

    """
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.connection = self.connect()

    def connect(self):
        """
        Creates a database connection and returns the connection as an object. 

        Returns: duckdb.DuckDBPyConnection:  The established connection object
        """
        self.connection = duckdb.connect(self.db_url)
        print(f"Connected to {self.db_url}")
        return self.connection

    def disconnect(self):
        """
        Safely closes the database connection.

        """
        self.connection.close()

    def is_connected(self) -> bool:
        """
        Checks if it is connected by checking if the connection is not none.

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
        """Fetch all column names from the Jobs table.
        
        Args: table_name 

        Returns: None
        
        """
        if self.is_connected():
            query = f"SELECT * FROM {table_name} LIMIT 0"
            return self.connection.execute(query).df().columns.tolist()
        else:
            raise Exception("Not connected")

    def fetch_all(self, table_name="Jobs"):
        """Fetch all data from the specified table. Table name is set to Jobs but can be changed accordingly.
        
        
        Args: table_name


        Returns: pd.DataFrame 

        
        """
        if self.is_connected():
            query = f"SELECT * FROM {table_name}"
            return self.connection.execute(query).fetchdf()
        else:
            raise Exception("Not connected")

    def fetch_query(self, query: str):
        """
        Fetch data based on a custom query. If the query is invalid due to column names,
        raise an exception with valid column names.


        Args: str 

        Returns: DataFrame 
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
