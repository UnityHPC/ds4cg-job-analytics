import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import pandas as pd
from pydantic import BaseModel

KwargsModelType = TypeVar("KwargsModelType", bound=BaseModel)


class DataVisualizer(Generic[KwargsModelType], ABC):
    """
    Base class for visualizing and summarizing pre-processed data.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the DataVisualizer.

        Args:
            df (pd.DataFrame): DataFrame to visualize.

        Returns:
            None

        Raises:
            ValueError: If no DataFrame is provided.
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

    @staticmethod
    def anonymize_str_column(column: pd.Series, prefix: str) -> pd.Series:
        """Anonymize a DataFrame column by replacing its values with a unique identifier.

        Args:
            column (pd.Series): The column to anonymize.
            prefix (str): The prefix to add to the anonymized values.

        Returns:
            pd.Series: The anonymized column.
        """
        return prefix + column.rank(method="dense").astype(int).astype(str).str.zfill(2)

    @staticmethod
    def validate_sampling_arguments(sample_size: int | None, random_seed: int | None) -> tuple[int | None, int | None]:
        """Validate the sample size and random seed for visualization.

        Args:
            sample_size (int): The number of rows to sample for visualization.
            random_seed (int): The random seed for reproducibility.

        Returns:
            int or None: Validated sample size.

        Raises:
            ValueError: If sample_size is provided but is not a positive integer.
        """
        if sample_size is not None and (not isinstance(sample_size, int) or sample_size <= 0):
            raise ValueError("Sample size must be a positive integer.")
        if sample_size is not None and (not isinstance(random_seed, int)):
            raise ValueError("Random seed must be an integer.")
        return sample_size, random_seed

    @staticmethod
    def validate_figsize(figsize: tuple[float | int, float | int]) -> tuple[float | int, float | int]:
        """Validate the figure size for visualization.

        Args:
            figsize (tuple[float | int, float | int]): Size of the figure.

        Raises:

            TypeError: If figsize is not a tuple of two numbers (float or int).

        Returns:
            tuple[float | int, float | int]: Validated figure size.
        """

        if not (
            isinstance(figsize, tuple) and len(figsize) == 2 and all(isinstance(x, (float, int)) for x in figsize)
        ):
            raise TypeError("'figsize' must be a tuple of two numbers (float or int)")
        return figsize

    @staticmethod
    def validate_output_dir(output_dir_path: Path | None) -> Path | None:
        """Validate the output directory for saving plots.

        Args:
            output_dir_path (Path): Directory to save plots.

        Returns:
            Path or None: Validated output directory path.

        Raises:
            ValueError: If output_dir_path is provided but is not a valid directory.
            PermissionError: If output_dir_path is not writable.
        """
        if output_dir_path is not None:
            if not isinstance(output_dir_path, Path):
                raise ValueError("Output directory must be a Path object.")
            if not output_dir_path.is_dir():
                raise ValueError(f"Output directory {output_dir_path} does not exist or is not a directory.")
            if not os.access(output_dir_path, os.W_OK):
                raise PermissionError(f"Output directory {output_dir_path} is not writable.")
        return output_dir_path

    @staticmethod
    def _generate_summary_stats(
        jobs_df: pd.DataFrame, validated_output_dir_path: Path | None, summary_file_name: str
    ) -> None:
        """Generate summary statistics for each column.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            validated_output_dir_path (Path | None): The directory to save the summary file.
            summary_file_name (str): The name of the summary file. If it already exists, it's overwritten.

        Returns:
            None
        """
        if validated_output_dir_path is not None:
            # create text file to save column summary statistics
            summary_file = validated_output_dir_path / summary_file_name
            if summary_file.exists():
                print(f"Summary file already exists. Overwriting {summary_file.name}")

            summary_lines = ["Column Summary Statistics\n", "=" * 30 + "\n"]
            for col in jobs_df.columns:
                summary_lines.append(f"\nColumn: {col}\n")
                summary_lines.append(str(jobs_df[col].describe(include="all")) + "\n")

            with open(summary_file, "w", encoding="utf-8") as f:
                f.writelines(summary_lines)
        else:
            for col in jobs_df.columns:
                print("\n" + "=" * 50)
                print(f"Column: {col}")
                print("-" * 50)
                print(jobs_df[col].describe(include="all"))

    @staticmethod
    def pie_chart_autopct_func(p: float, threshold_pct: int = 5) -> str:
        """Format the percentage for pie chart labels.

        Args:
            p (float): The percentage value to format.
            threshold_pct (int): The threshold percentage below which labels are not shown.

        Returns:
            str: Formatted percentage string or empty string if below threshold.
        """
        return f"{p:.1f}%" if p >= threshold_pct else ""

    @staticmethod
    def validate_column_argument(column: str | None, df: pd.DataFrame) -> str | None:
        """Validate the provided column against the DataFrame.

        Args:
            column (str | None): Column name to validate.
            df (pd.DataFrame): The DataFrame to check against.

        Raises:
            TypeError: If 'column' is not a string or None.
            ValueError: If the column is not present in the DataFrame.

        Returns:
            str | None: Validated column name.
        """
        if column is not None and not isinstance(column, str):
            raise TypeError("'column' must be a string or None")

        if column is not None and df is not None and column not in df.columns:
            raise ValueError("The specified column is not present in the DataFrame.")

        return column

    @staticmethod
    def validate_columns(columns: list[str] | None, df: pd.DataFrame) -> list[str] | None:
        """Validate the provided columns against the DataFrame.

        Args:
            columns (list[str]): List of column names to validate.
            df (pd.DataFrame): The DataFrame to check against.

        Raises:
            TypeError: If 'columns' is not a list of strings or None.
            ValueError: If any column is not present in the DataFrame or if 'columns' is an empty list.

        Returns:
            list[str]: Validated list of column names.
        """
        if columns is not None and (not all(isinstance(x, str) for x in columns)):
            raise TypeError("'columns' must be a list of strings or None")

        if columns is not None and len(columns) == 0:
            raise ValueError("'columns' cannot be an empty list. 'columns' must be a list of strings or None")

        if columns is not None and df is not None and not all(col in df.columns for col in columns):
            raise ValueError("One or more specified columns are not present in the DataFrame.")
        return columns

    @abstractmethod
    def validate_visualize_kwargs(
        self,
        kwargs: dict[str, Any],
        validated_jobs_df: pd.DataFrame,
        kwargs_model: type[KwargsModelType],
    ) -> KwargsModelType:
        """
        Validate the keyword arguments for the visualize method.

        Args:
            kwargs (dict[str, Any]): The keyword arguments to validate.
            validated_jobs_df (pd.DataFrame): The validated DataFrame to use for visualization.
            kwargs_model (type[KwargsModelType]): The Pydantic model to validate against.

        Returns:
            KwargsModelType: The validated keyword arguments as a BaseModel.
        """
        pass

    @abstractmethod
    def visualize(self, output_dir_path: Path | None = None, **kwargs: dict[str, Any]) -> None:
        """Visualize the data in the DataFrame.

        Args:
            output_dir_path (Path | None): Directory to save plots and summaries.
            **kwargs: Additional keyword arguments for flexibility, such as columns to visualize,
                      sample size, random seed, summary file name, and figure size.

        Raises:
            ValueError: If any of the provided arguments are invalid.

        Returns:
            None
        """
        pass
