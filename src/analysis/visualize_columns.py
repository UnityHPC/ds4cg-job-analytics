import pandas as pd
from pathlib import Path
import os

"""
Visualization utilities for pre-processed Unity job data.

Provides a function to visualize and summarize each column of a DataFrame, including appropriate
plots and statistics for numeric and categorical columns.
"""


class DataVisualizer:
    """A class for visualizing and summarizing columns of pre-processed data in a DataFrame."""

    def __init__(
        self, df: pd.DataFrame
    ) -> None:
        """Initialize the DataVisualizer.
        Args:
            df (pd.DataFrame): DataFrame to visualize.
        """
        self.df = None
        if df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Must provide a DataFrame.")
        
    def validate_dataframe(self) -> pd.DataFrame:
        """Validate that the DataFrame is not empty and has columns.

        Returns:
            pd.DataFrame: The validated DataFrame.
        Raises:
            ValueError: If the DataFrame is empty or has no columns.
        """
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty or not provided.")
        if self.df.columns.empty:
            raise ValueError("DataFrame has no columns.")
        return self.df
        
    def validate_sample_size(self, sample_size: int | None) -> int | None:
        """Validate the sample size for visualization.

        Parameters:
            sample_size (int): The number of rows to sample for visualization.
        Returns:
            int or None: Validated sample size.
        Raises:
            ValueError: If sample_size is provided but is not a positive integer.
        """
        if sample_size is not None and (not isinstance(sample_size, int) or sample_size <= 0):
            raise ValueError("Sample size must be a positive integer.")
        return sample_size
    
    def validate_random_seed(self, random_seed: int | None) -> int | None:
        """Validate the random seed for reproducibility.

        Parameters:
            random_seed (int): The random seed to use for sampling.
        Returns:
            int or None: Validated random seed.
        Raises:
            ValueError: If random_seed is provided but is not an integer.
        """
        if not isinstance(random_seed, int):
            raise ValueError("Random seed must be an integer.")
        return random_seed
        
    def validate_columns(self, columns: list[str]) -> list[str]:
        """Validate the provided columns against the DataFrame.

        Parameters:
            columns (list[str]): List of column names to validate.

        Raises:
            ValueError: If any column is not present in the DataFrame.
        """
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise ValueError("Columns must be a list of strings.")
        if self.df is not None and not all(col in self.df.columns for col in columns):
            raise ValueError("One or more specified columns are not present in the DataFrame.")
        return columns

    def validate_output_dir(self, output_dir_path: Path | None) -> Path | None:
        """Validate the output directory for saving plots.

        Parameters:
            output_dir_path (Path): Directory to save plots.
        Returns:
            Path or None: Validated output directory path.
        Raises:
            ValueError: If output_dir_path is provided but is not a valid directory.
        """
        if output_dir_path is not None:
            if not isinstance(output_dir_path, Path):
                raise ValueError("Output directory must be a Path object.")
            if not output_dir_path.is_dir():
                raise ValueError(f"Output directory {output_dir_path} does not exist or is not a directory.")
            if not os.access(output_dir_path, os.W_OK):
                raise PermissionError(f"Output directory {output_dir_path} is not writable.")
        return output_dir_path

    def visualize_columns(
        self,
        columns=None,
        sample_size: int | None = None,
        random_seed: int | None = None,
        output_dir_path: Path | None = None,
        summary_file_name: str = "columns_stats_summary.txt",
        figsize: tuple[int, int] = (7, 4),
    ) -> None:
        """Visualize and summarize specified columns of the data.

        Args:
            columns (list[str], optional): Columns to visualize. If None, all columns are used.
            sample_size (int, optional): Number of rows to sample. If None, uses all rows.
            random_seed (int, optional): Seed for reproducible sampling.
            output_dir_path (Path, optional): Directory to save plots. If None, plots are displayed.
            summary_file_name (str): Name of the text file to save column summaries in the output directory.
            figsize (tuple[int, int]): Size of the figure for plots.
        Raises:
            ValueError: On invalid DataFrame, sample size, random seed, or columns.

        Returns:
            None
        """
        df = self.validate_dataframe()
        self.validate_sample_size(sample_size)
        self.validate_random_seed(random_seed)
        self.validate_columns(columns if columns is not None else df.columns.tolist())
        self.validate_output_dir(output_dir_path)
        
        df = df.copy()

        # If specific columns are provided, select them
        if columns is not None:
            df = df[columns]

        # Sample the data if sample_size is specified
        if sample_size is not None:
            if len(df) < sample_size:
                raise ValueError(
                    f"Sample size {sample_size} is larger than the DataFrame size {len(df)}."
                )
            df = df.sample(sample_size, random_state=random_seed)


        # plt.figure(figsize=figsize)

        if output_dir_path is not None:
            # create text file to save column summary statistics
            summary_file = output_dir_path / summary_file_name
            if summary_file.exists():
                summary_file = summary_file.with_name(summary_file.stem + "_new.txt")
                print(f"Summary file already exists. Saving as {summary_file.name}")
            
            summary_lines = ["Column Summary Statistics\n", "=" * 30 + "\n"]
            for col in df.columns:
                summary_lines.append(f"\nColumn: {col}\n")
                summary_lines.append(str(df[col].describe(include="all")) + "\n")
            
            with open(summary_file, "w", encoding="utf-8") as f:
                f.writelines(summary_lines)
        else:
            for col in df.columns:
                print("\n" + "=" * 50)
                print(f"Column: {col}")
                print("-" * 50)
                print(df[col].describe(include="all"))
                print("=" * 50 + "\n")
                print(df[col].describe())

        # # Loop through each column for visualization
        # for col in df.columns:
        #     plt.figure(figsize=figsize)
        #     col_data = df[col]
        #     col_type = col_data.dtype

        #     # UUID and JobID: treat as categorical if low cardinality, else skip plot
        #     if col in ["UUID", "JobID", "ArrayID"]:
        #         if col_data.nunique() < 30:
        #             sns.countplot(x=col, data=df)
        #             plt.title(f"Bar plot of {col}")
        #             plt.xticks(rotation=45, ha="right")
        #         else:
        #             plt.close()
        #             continue

        #     # Boolean columns
        #     elif col_type == bool or col in ["IsArray", "Preempted"]:
        #         sns.countplot(x=col, data=df)
        #         plt.title(f"Bar plot of {col}")
        #         plt.xticks(rotation=45, ha="right")

        #     # Timestamps: plot histogram of times and durations if possible
        #     elif "Time" in col or col_type.name.startswith("datetime"):
        #         if pd.api.types.is_datetime64_any_dtype(col_data):
        #             sns.histplot(col_data.dropna(), bins=30, kde=False)
        #             plt.title(f"Histogram of {col}")
        #             plt.xticks(rotation=45, ha="right")
        #         else:
        #             plt.close()
        #             continue

        #     # Numeric columns: histogram, and boxplot if enough data
        #     elif pd.api.types.is_numeric_dtype(col_data):
        #         sns.histplot(col_data.dropna(), bins=30, kde=True)
        #         plt.title(f"Histogram of {col}")
        #         plt.tight_layout()
        #         if output_dir_path is not None:
        #             plt.savefig(output_dir_path / f"{col}_hist.png")
        #             plt.close()
        #             plt.figure(figsize=figsize)
        #         if col_data.count() > 10:
        #             sns.boxplot(x=col_data.dropna())
        #             plt.title(f"Boxplot of {col}")
        #         else:
        #             plt.close()
        #             continue

        #     # Categorical columns: bar plot of top 20 categories
        #     elif pd.api.types.is_categorical_dtype(col_data) or col_type == object:
        #         # For array-like columns, join lists to string for plotting
        #         if col_data.apply(lambda x: isinstance(x, (list, tuple))).any():
        #             flat = col_data.dropna().explode()
        #             top_cats = flat.value_counts().nlargest(20)
        #             sns.barplot(x=top_cats.index, y=top_cats.values)
        #             plt.title(f"Top 20 values in {col} (exploded)")
        #             plt.xticks(rotation=45, ha="right")
        #         else:
        #             top_cats = col_data.value_counts().nlargest(20)
        #             sns.barplot(x=top_cats.index, y=top_cats.values)
        #             plt.title(f"Top 20 values in {col}")
        #             plt.xticks(rotation=45, ha="right")

        #     else:
        #         # Unsupported column types
        #         print(f"(Unsupported column type for visualization: {col})")
        #         plt.close()
        #         continue

        #     plt.tight_layout()
        #     if output_dir_path is not None:
        #         plt.savefig(output_dir_path / f"{col}.png")
        #         plt.close()
        #     else:
        #         plt.show()

        # # Loop through each column for visualization
        # for col in df.columns:
        #     print(f"\nColumn: {col}")
        #     print(df[col].describe(include="all"))  # Print summary statistics
        #     plt.figure(figsize=(7, 4))
        #     # Numeric columns: bar plot for low cardinality, histogram otherwise
        #     if pd.api.types.is_numeric_dtype(df[col]):
        #         if df[col].nunique() < 20:
        #             sns.countplot(x=col, data=df)
        #             plt.title(f"Bar plot of {col}")
        #         else:
        #             sns.histplot(df[col].dropna(), kde=True, bins=30)
        #             plt.title(f"Histogram of {col}")
        #     # Categorical columns: bar plot of top 20 categories
        #     elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
        #         top_cats = df[col].value_counts().nlargest(20)
        #         sns.barplot(x=top_cats.index, y=top_cats.values)
        #         plt.title(f"Top categories in {col}")
        #         plt.xticks(rotation=45, ha="right")
        #     else:
        #         # Unsupported column types
        #         print("(Unsupported column type for visualization)")
        #         plt.close()
        #         continue
        #     plt.tight_layout()
        #     plt.show()
