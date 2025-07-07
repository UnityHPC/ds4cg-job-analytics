import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_efficiency_curves(
    data: pd.DataFrame,
    efficiency_column: str,
    gpu_hours_column: str,
    thresholds: np.ndarray,
    group_by: str = "",
    filter_by: str = "",
    filter_value: str = "",
    title: str = "",
):
    """
    Plot efficiency curves for jobs and GPU hours below efficiency thresholds.

    Args:
        data: DataFrame with efficiency data
        efficiency_column: Column with efficiency percentages
        gpu_hours_column: Column with GPU hours
        thresholds: Array of efficiency thresholds
        group_by: Column to group data by (optional)
        filter_by: Column to filter data by (optional)
        filter_value: Value to filter by (optional)
        title: Title for the plot (optional)

    Returns:
        tuple: (Figure, Axes)
    """
    if filter_by and filter_value:
        data = data[data[filter_by] == filter_value]

    groups = [None] if group_by == "" else data[group_by].unique()

    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 5 * len(groups)), sharex=True)
    if len(groups) == 1:
        axes = [axes]

    for i, group in enumerate(groups):
        current_ax = axes[i]
        group_data = data if group is None else data[data[group_by] == group]

        job_percentages, gpu_hour_percentages = _calculate_efficiency_curves(
            group_data, efficiency_column, gpu_hours_column, thresholds
        )

        current_ax.plot(thresholds, job_percentages, label="Jobs", marker="o")
        current_ax.plot(thresholds, gpu_hour_percentages, label="GPU Hours", marker="o")
        current_ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="50% Efficiency")

        metric_title = group_by if group_by else "Job Count"
        current_ax.set_title(f"Efficiency Curve - {metric_title}")
        current_ax.set_xlabel("Efficiency Threshold (%)")
        current_ax.set_ylabel("Percentage Below Threshold (%)")
        current_ax.legend()

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    return fig, axes


def _calculate_efficiency_curves(
    data: pd.DataFrame, efficiency_column: str, gpu_hours_column: str, thresholds: np.ndarray
):
    """
    Calculate the percentage of jobs and GPU hours below each efficiency threshold.

    Args:
        data: DataFrame with efficiency data
        efficiency_column: Column with efficiency percentages
        gpu_hours_column: Column with GPU hours
        thresholds: Array of efficiency thresholds

    Returns:
        tuple: (job_percentages, gpu_hour_percentages)
    """
    valid_data = data.dropna(subset=[efficiency_column])
    valid_data[efficiency_column] = np.clip(valid_data[efficiency_column], 0, 100)

    total_jobs = len(valid_data)
    total_gpu_hours = valid_data[gpu_hours_column].sum() if gpu_hours_column in valid_data.columns else total_jobs

    job_percentages = []
    gpu_hour_percentages = []

    if thresholds[-1] < 100:
        thresholds = np.append(thresholds, 100)

    for threshold in thresholds:
        jobs_below = len(valid_data[valid_data[efficiency_column] <= threshold])
        job_pct = (jobs_below / total_jobs) * 100 if total_jobs > 0 else 0
        job_percentages.append(job_pct)

        if gpu_hours_column in valid_data.columns:
            gpu_hours_below = valid_data[valid_data[efficiency_column] <= threshold][gpu_hours_column].sum()
            gpu_hour_pct = (gpu_hours_below / total_gpu_hours) * 100 if total_gpu_hours > 0 else 0
        else:
            gpu_hour_pct = job_pct
        gpu_hour_percentages.append(gpu_hour_pct)

    if job_percentages and job_percentages[-1] < 100:
        job_percentages[-1] = 100.0
    if gpu_hour_percentages and gpu_hour_percentages[-1] < 100:
        gpu_hour_percentages[-1] = 100.0

    return job_percentages, gpu_hour_percentages
