import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import seaborn as sns
from pathlib import Path
import os
import numpy as np
from matplotlib.gridspec import GridSpec

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

        Args:
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

        Args:
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

        Args:
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

        Args:
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

        # Generate summary statistics for each column
        if output_dir_path is not None:
            # create text file to save column summary statistics
            summary_file = output_dir_path / summary_file_name
            if summary_file.exists():
                print(f"Summary file already exists. Overwriting {summary_file.name}")
            
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

        # Generate visualizations for each column
        for col in df.columns:
            plt.figure(figsize=figsize)
            col_data = df[col]
            col_type = col_data.dtype

            if col in ["UUID"]:
                # Skip UUID column as it is not useful for visualization
                plt.close()
                continue
            
            # JobID: treat as categorical if low cardinality, else skip plot
            if col in ["JobID", "ArrayID"]:
                if col_data.nunique() < 30:
                    sns.countplot(x=col, data=df)
                    plt.title(f"Bar plot of {col}")
                    plt.xticks(rotation=45, ha="right")
                else:
                    plt.close()
                    continue

            # Boolean columns
            elif col_type is bool or col in ["IsArray", "Preempted"]:
                plt.figure(figsize=(3,7))
                ax = sns.countplot(x=col, stat="percent", data=df)
                if isinstance(ax.containers[0], BarContainer):
                    # The heights are already in percent (0-100) due to stat="percent"
                    ax.bar_label(ax.containers[0], labels=[f"{h.get_height():.1f}%" for h in ax.containers[0]])
                plt.title(f"Bar plot of {col}")
                plt.xticks(rotation=45, ha="right")
                if output_dir_path is not None:
                    plt.savefig(output_dir_path / f"{col}_barplot.png")
                plt.show()

            # Timestamps: plot histogram of times and durations if possible
            elif col in ["StartTime"]:
                if not pd.api.types.is_datetime64_any_dtype(col_data):
                    # Convert to datetime if not already
                    col_data = pd.to_datetime(col_data, errors="coerce")  # invalid timestamps will be NaT

                col_data = col_data.dropna().sort_values()
                if col_data.empty:
                    print(f"No valid timestamps in {col}. Skipping visualization.")
                    plt.close()
                    continue

                min_time = col_data.min()
                max_time = col_data.max()
                total_days = (max_time - min_time).days + 1

                plt.figure(figsize=(7, 7))
                # If jobs span more than 2 days, plot jobs per day
                if total_days > 2:
                    # Group by date, count jobs per day
                    jobs_per_day = col_data.dt.floor('D').value_counts().sort_index()
                    # Trim days at start/end with zero jobs
                    jobs_per_day = jobs_per_day[jobs_per_day > 0]
                    # A line plot is often better for time series to show trends over days
                    plt.plot(
                        jobs_per_day.index,
                        np.asarray(jobs_per_day.values, dtype=int),
                        marker='o',
                        linestyle='-'
                    )
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
                    plt.xlabel("Date")
                    plt.ylabel("Number of jobs")
                    plt.title(f"Jobs per day for {col}")
                    plt.xticks(rotation=45, ha="right")
                    # Annotate points above the line with a vertical offset for readability
                    ylim = ax.get_ylim()
                    y_range = ylim[1] - ylim[0]
                    offset = max(5, 0.04 * y_range)
                    for x, y in zip(jobs_per_day.index, jobs_per_day.values, strict=True):
                        if y > 0:
                            label_y = y + offset
                            # Calculate the annotation box height in data coordinates
                            # so the box does not go beyond the top spine
                            ann = ax.annotate(
                                f"{int(y)}",
                                xy=(x, label_y),
                                xytext=(0, 0),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                                clip_on=True
                            )
                            # Adjust annotation if it goes out of bounds
                            bbox = ann.get_window_extent(renderer=None)
                            inv = ax.transData.inverted()
                            bbox_data = inv.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
                            top_of_box = bbox_data[1][1]
                            if top_of_box > ylim[1]:
                                # Move annotation up so it fits inside the plot
                                delta = top_of_box - ylim[1]
                                ann.set_position((0, -delta * ax.bbox.height / y_range))
                    if output_dir_path is not None:
                        plt.savefig(output_dir_path / f"{col}_days_lineplot.png")
                else:
                    # All jobs within a couple of days: plot by hour
                    jobs_per_hour = col_data.dt.floor('H').value_counts().sort_index()
                    jobs_per_hour = jobs_per_hour[jobs_per_hour > 0]
                    # Use line plot for time series to better show trends over hours
                    plt.plot(
                        jobs_per_hour.index,
                        np.asarray(jobs_per_hour.values, dtype=int),
                        marker='o',
                        linestyle='-'
                    )
                    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
                    plt.xlabel("Hour")
                    plt.ylabel("Number of jobs")
                    plt.title(f"Jobs per hour for {col}")
                    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)

                    # Set x-axis labels: show date at midnight, hour otherwise
                    ax = plt.gca()
                    tick_locs = jobs_per_hour.index
                    tick_labels = []
                    for dt in tick_locs:
                        if dt.hour == 0:
                            tick_labels.append(dt.strftime("%Y-%m-%d %H:00"))
                        else:
                            tick_labels.append(dt.strftime("%H:00"))
                    # Show at most 12 labels to avoid crowding
                    step = max(1, len(tick_labels) // 12)
                    ax.set_xticks(tick_locs[::step])
                    ax.set_xticklabels([tick_labels[i] for i in range(0, len(tick_labels), step)], rotation=45, ha="right")

                    plt.tight_layout()
                    if output_dir_path is not None:
                        plt.savefig(output_dir_path / f"{col}_hourly_lineplot.png")
                plt.show()


             # Interactive    

            # # Numeric columns: histogram, and boxplot if enough data
            # elif pd.api.types.is_numeric_dtype(col_data):
            #     sns.histplot(col_data.dropna(), bins=30, kde=True)
            #     plt.title(f"Histogram of {col}")
            #     plt.tight_layout()
            #     if output_dir_path is not None:
            #         plt.savefig(output_dir_path / f"{col}_hist.png")
            #         plt.close()
            #         plt.figure(figsize=figsize)
            #     if col_data.count() > 10:
            #         sns.boxplot(x=col_data.dropna())
            #         plt.title(f"Boxplot of {col}")
            #     else:
            #         plt.close()
            #         continue

            # # Categorical columns: bar plot of top 20 categories
            # elif pd.api.types.is_categorical_dtype(col_data) or col_type == object:
            #     # For array-like columns, join lists to string for plotting
            #     if col_data.apply(lambda x: isinstance(x, (list, tuple))).any():
            #         flat = col_data.dropna().explode()
            #         top_cats = flat.value_counts().nlargest(20)
            #         sns.barplot(x=top_cats.index, y=top_cats.values)
            #         plt.title(f"Top 20 values in {col} (exploded)")
            #         plt.xticks(rotation=45, ha="right")
            #     else:
            #         top_cats = col_data.value_counts().nlargest(20)
            #         sns.barplot(x=top_cats.index, y=top_cats.values)
            #         plt.title(f"Top 20 values in {col}")
            #         plt.xticks(rotation=45, ha="right")

            # else:
            #     # Unsupported column types
            #     print(f"(Unsupported column type for visualization: {col})")
            #     plt.close()
            #     continue

            # plt.tight_layout()
            # if output_dir_path is not None:
            #     plt.savefig(output_dir_path / f"{col}.png")
            #     plt.close()
            # else:
            #     plt.show()

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
