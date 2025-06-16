import duckdb
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def get_db_dataframe(db_path=None, table_name="Jobs"):
    """
    Connect to the DuckDB database and return the jobs table as a pandas DataFrame.

    Args:
        db_path (str or Path, optional): Path to the DuckDB database. Defaults to 'data/slurm_data_small.db'.
        table_name (str, optional): Table name to query. Defaults to 'Jobs'.

    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """
    if db_path is None:
        db_path = Path(__file__).resolve().parents[2] / "data" / "slurm_data_small.db"
    con = duckdb.connect(str(db_path))
    df = con.execute(f"SELECT * FROM {table_name}").df()
    con.close()
    return df


def plot_selected_pairs(df, pairs=None, hue=None, sample=1000, plot_kws=None):
    """
    Plot selected pairs of columns from a DataFrame using seaborn scatterplots.

    Args:
        df (pd.DataFrame): The DataFrame to plot from.
        pairs (list of tuple, optional): List of (x, y) column pairs to plot. If None, uses recommended pairs.
        hue (str, optional): Column name for color grouping.
        sample (int, optional): Number of rows to sample for plotting (for large datasets).
        plot_kws (dict, optional): Additional keyword arguments for seaborn.scatterplot.

    Returns:
        None: Shows the plots.
    """
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
        ]
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)
    if plot_kws is None:
        plot_kws = dict(alpha=0.6)
    n = len(pairs)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.flatten()
    for i, (x, y) in enumerate(pairs):
        ax = axes[i]
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, **plot_kws)
        ax.set_title(f"{y} vs. {x}")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
