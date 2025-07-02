from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))  # Adjust path to include src directory
from src.database import DatabaseConnection
from src.preprocess.preprocess import preprocess_data
from src.analysis.visualize_columns import DataVisualizer


class PairPlotVisualizer(DataVisualizer):
    """
    A class for visualizing pairs of columns in a DataFrame, inheriting from DataVisualizer.
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    @staticmethod
    def load_and_preprocess_data(db_path=None, table_name="Jobs") -> pd.DataFrame:
        """
        Load data from the database and preprocess it.

        Args:
            db_path (str or Path, optional): Path to the DuckDB database. Defaults to 'data/slurm_data.db'.
            table_name (str, optional): Table name to query. Defaults to 'Jobs'.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        if db_path is None:
            db_path = Path(__file__).resolve().parents[2] / "data" / "slurm_data.db"
        db = DatabaseConnection(db_path)
        raw_data = db.connection.execute(f"SELECT * FROM {table_name}").fetchdf()
        return preprocess_data(raw_data)

    def plot_pairs(self, pairs=None, hue=None, sample=1000, plot_kws=None, output_dir_path=None):
        """
        Plot selected pairs of columns from the DataFrame using seaborn scatterplots.

        Args:
            pairs (list of tuple, optional): List of (x, y) column pairs to plot. If None, uses recommended pairs.
            hue (str, optional): Column name for color grouping.
            sample (int, optional): Number of rows to sample for plotting (for large datasets).
            plot_kws (dict, optional): Additional keyword arguments for seaborn.scatterplot.
            output_dir_path (Path, optional): Directory to save plots. If None, plots are shown but not saved.

        Returns:
            None: Shows or saves the plots.
        """
        self.validate_dataframe()
        if pairs is None:
            pairs = [
                ("GPUMemUsage", "GPUComputeUsage"),
                ("Elapsed", "GPUMemUsage"),
                ("Elapsed", "GPUComputeUsage"),
                ("CPUMemUsage", "CPUComputeUsage"),
                ("GPUMemUsage", "CPUMemUsage"),
                ("GPUMemUsage", "CPUs"),
                ("GPUs", "GPUMemUsage"),
                ("GPUs", "Elapsed"),
                ("GPUs", "GPUComputeUsage"),
                ("GPUComputeUsage", "CPUComputeUsage"),
            ]
        if sample and len(self.df) > sample:
            self.df = self.df.sample(sample, random_state=42)
        if plot_kws is None:
            plot_kws = dict(alpha=0.6)
        n = len(pairs)
        ncols = 2
        nrows = (n + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        axes = axes.flatten()
        for i, (x, y) in enumerate(pairs):
            ax = axes[i]
            sns.scatterplot(data=self.df, x=x, y=y, hue=hue, ax=ax, **plot_kws)
            ax.set_title(f"{y} vs. {x}")
            if output_dir_path:
                output_dir_path = self.validate_output_dir(output_dir_path)
                plot_path = output_dir_path / f"{x}_vs_{y}.png"
                fig.savefig(plot_path)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        if not output_dir_path:
            plt.show()

    def plot_time_columns(self, time_columns=None, output_dir_path=None):
        """
        Plot time-related columns using line plots or histograms.

        Args:
            time_columns (list of str, optional): List of time-related columns to plot.
            Defaults to ['Elapsed', 'TimeLimit', 'StartTime', 'SubmitTime'].
            output_dir_path (Path, optional): Directory to save plots. If None, plots are shown but not saved.

        Returns:
            None: Shows or saves the plots.
        """
        self.validate_dataframe()
        if time_columns is None:
            time_columns = ['Elapsed', 'TimeLimit', 'StartTime', 'SubmitTime']

        for col in time_columns:
            if col not in self.df.columns:
                print(f"Skipping column {col}: not found in DataFrame.")
                continue

            col_data = self.df[col].dropna()

            if pd.api.types.is_timedelta64_dtype(col_data):
                col_data = col_data.dt.total_seconds() / 60  # Convert to minutes
                plt.figure(figsize=(10, 6))
                sns.histplot(col_data, bins=30, kde=True, color="blue")
                plt.title(f"Histogram of {col} (minutes)")
                plt.xlabel("Minutes")
                plt.ylabel("Frequency")
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_data = col_data.sort_values()
                plt.figure(figsize=(10, 6))
                plt.plot(col_data, range(len(col_data)), marker="o", linestyle="-")
                plt.title(f"Line Plot of {col} Over Time")
                plt.xlabel("Time")
                plt.ylabel("Count")
            else:
                print(f"Skipping column {col}: unsupported data type.")
                continue

            if output_dir_path:
                output_dir_path = self.validate_output_dir(output_dir_path)
                plot_path = output_dir_path / f"{col}_time_plot.png"
                plt.savefig(plot_path)
            plt.show()

    def plot_time_vs_columns(self, time_column, other_columns=None, output_dir_path=None):
        """
        Plot a time-related column against other specified columns.

        Args:
            time_column (str): The time-related column to plot on the x-axis.
            other_columns (list of str, optional): List of columns to plot against the time column.
                Defaults to ['GPUMemUsage', 'GPUComputeUsage', 'CPUMemUsage', 'CPUComputeUsage'].
            output_dir_path (Path, optional): Directory to save plots. If None, plots are shown but not saved.

        Returns:
            None: Shows or saves the plots.
        """
        self.validate_dataframe()
        if time_column not in self.df.columns:
            print(f"Skipping time column {time_column}: not found in DataFrame.")
            return

        if other_columns is None:
            other_columns = ['GPUMemUsage', 'GPUComputeUsage', 'CPUMemUsage', 'CPUComputeUsage']

        for col in other_columns:
            if col not in self.df.columns:
                print(f"Skipping column {col}: not found in DataFrame.")
                continue

            time_data = self.df[time_column].dropna()
            col_data = self.df[col].dropna()

            if pd.api.types.is_datetime64_any_dtype(time_data):
                time_data = time_data.sort_values()
                col_data = col_data.loc[time_data.index]
                plt.figure(figsize=(10, 6))
                plt.plot(time_data, col_data, marker="o", linestyle="-")
                plt.title(f"{col} vs. {time_column}")
                plt.xlabel(time_column)
                plt.ylabel(col)
            elif pd.api.types.is_timedelta64_dtype(time_data):
                time_data = time_data.dt.total_seconds() / 60  # Convert to minutes
                col_data = col_data.loc[time_data.index]
                plt.figure(figsize=(10, 6))
                plt.plot(time_data, col_data, marker="o", linestyle="-")
                plt.title(f"{col} vs. {time_column} (minutes)")
                plt.xlabel(f"{time_column} (minutes)")
                plt.ylabel(col)
            else:
                print(f"Skipping time column {time_column}: unsupported data type.")
                continue

            if output_dir_path:
                output_dir_path = self.validate_output_dir(output_dir_path)
                plot_path = output_dir_path / f"{col}_vs_{time_column}.png"
                plt.savefig(plot_path)
            plt.show()
