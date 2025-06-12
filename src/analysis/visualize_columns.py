import pandas as pd

"""
Visualization utilities for processed GPU job data.

Provides a function to visualize and summarize each column of a DataFrame, including appropriate
plots and statistics for numeric and categorical columns. Designed to work with data loaded via
GPUMetrics or similar classes.
"""


class DataVisualizer:
    """A class for visualizing and summarizing columns of processed data from a database or DataFrame."""

    def __init__(
        self, db_path: str = "../../data/slurm_data_small.db", table: str = "jobs", df: pd.DataFrame = None
    ) -> None:
        """Initialize the DataVisualizer.

        Args:
            db_path (str, optional): Path to the DuckDB database file. If provided, will connect to DB.
            table (str, optional): Table name to load from the database (used with db_path).
            df (pd.DataFrame, optional): DataFrame to visualize directly. If provided, DB is ignored.
        """
        self.df = None
        self.con = None
        # If a DataFrame is provided, use it directly
        if df is not None:
            self.df = df.copy()
        # Otherwise, connect to DuckDB and load the specified table
        elif db_path is not None and table is not None:
            import duckdb

            self.con = duckdb.connect(db_path)
            self.df = self.con.execute(f"SELECT * FROM {table}").df()
        else:
            raise ValueError("Must provide either a DataFrame or both db_path and table name.")

    def visualize_columns(self, columns=None, sample_size: int = 1000) -> None:
        """Visualize and summarize specified columns of the data.

        Args:
            columns (list[str], optional): List of columns to visualize. If None, visualize all columns.
            sample_size (int, optional): Number of rows to sample for visualization (default 1000).

        Returns:
            None: Displays plots and prints statistics for each column.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = self.df.copy()
        # If specific columns are provided, select them
        if columns is not None:
            df = df[columns]
        # Sample the data if it's large
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        # Loop through each column for visualization
        for col in df.columns:
            print(f"\nColumn: {col}")
            print(df[col].describe(include="all"))  # Print summary statistics
            plt.figure(figsize=(7, 4))
            # Numeric columns: bar plot for low cardinality, histogram otherwise
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < 20:
                    sns.countplot(x=col, data=df)
                    plt.title(f"Bar plot of {col}")
                else:
                    sns.histplot(df[col].dropna(), kde=True, bins=30)
                    plt.title(f"Histogram of {col}")
            # Categorical columns: bar plot of top 20 categories
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
                top_cats = df[col].value_counts().nlargest(20)
                sns.barplot(x=top_cats.index, y=top_cats.values)
                plt.title(f"Top categories in {col}")
                plt.xticks(rotation=45, ha="right")
            else:
                # Unsupported column types
                print("(Unsupported column type for visualization)")
                plt.close()
                continue
            plt.tight_layout()
            plt.show()
