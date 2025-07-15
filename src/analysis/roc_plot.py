"""
Design template for ROC:

Chris's code very useful. Generally, I can follow the format of 2 functions like that.
But I just need to validate it, raise error properly, and make sure it can work with multiple other columns.
Also need to handle the NULL values properly and optimize Chris codes

Actions item:
- Bring the 2 functions in Chris's code to this class DONE
- Refactor the calculate effciency threshold function DONE
- For the plot(), don't focus on group by for now, implement validation, errors raise, null handling for columns DONE
- how can we also allow options for filtering (dynamically, without creating a whole new instance)??
- Extend the plot() to accept some other columns (vram_hours, users)

TODO (Tan): consider writing tests for validate_input function
Low priority:
- explore group by option
- can try to integrate the aggregation metrics (in case we want to group by)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import sys
from .vram_usage import EfficiencyAnalysis

sys.path.append(str(Path(__file__).resolve().parents[2]))  # Adjust path to include src directory
from src.config.enum_constants import ROCProportionMetricsEnum, JobEfficiencyMetricsEnum


class ROCVisualizer(EfficiencyAnalysis):
    """
    A class for visualizing pairs of columns in a DataFrame, inheriting from EfficiencyAnalysis.

    TODO(Tan) : need to implement a validate function that makes sure whatever metrics used is calculated

    """

    def __init__(
        self,
        db_path: str | Path,
        table_name: str = "Jobs",
    ) -> None:
        super().__init__(db_path, table_name)
        # if self.jobs_with_efficiency_metrics is None:
        self.calculate_job_efficiency_metrics(self.jobs_df)

    def _validate_inputs(
        self,
        input_df: pd.DataFrame,
        min_threshold: float,
        max_threshold: float,
        threshold_step: float,
        threshold_metric: JobEfficiencyMetricsEnum,
        proportion_metric: ROCProportionMetricsEnum,
    ) -> None:
        """
        Validate the input metrics to ensure they are present in the DataFrame.

        Args:
            threshold_metric (JobEfficiencyMetricsEnum): The metric used for thresholds.
            proportion_metric (ROCProportionMetricsEnum): The metric for calculating proportions.
            threshold_step (float): Step size for thresholds.
            min_threshold (float): Minimum threshold value.
            max_threshold (float): Maximum threshold value.

        Raises:
            KeyError: If the specified metrics are not found in the DataFrame.
            ValueError: If the threshold step is not a positive number.
        """
        if threshold_metric.value not in input_df.columns:
            raise KeyError(f"Threshold metric '{threshold_metric.value}' not found in DataFrame.")
        if (
            proportion_metric.value != ROCProportionMetricsEnum.JOBS.value
            and proportion_metric.value not in input_df.columns
        ):
            raise KeyError(f"Proportion metric '{proportion_metric.value}' not found in DataFrame.")
        if threshold_step <= 0:
            raise ValueError("Threshold step must be a positive number.")
        if min_threshold > max_threshold:
            raise ValueError("min_threshold cannot be greater than max_threshold.")

    def _roc_calculate_proportion(
        self,
        plot_data_frame: pd.DataFrame,
        proportion_metric: ROCProportionMetricsEnum,
        thresholds_arr: list[float],
        threshold_metric: JobEfficiencyMetricsEnum,
        plot_percentage: bool = True,
    ) -> np.ndarray[float]:
        """
        Calculate the proportion of data that meet the alloc_vram_efficiency threshold for each threshold value.

        For each given threshold, this function will calculate the proportion of data (in terms of the
            specified metric) whose alloc_vram_efficiency is less than or equal to the threshold.

        Args:
            plot_data_frame (pd.DataFrame): DataFrame containing the data to plot.
            proportion_metric (ROCMetricsEnum): The metric to calculate proportions for.
            thresholds_arr (list[float]): List of predefined threshold values.
            threshold_metric (EfficiencyMetricsJobsEnum): The specific efficiency metric used as thresholds_arr.
            plot_percentage (bool): Whether to return the proportion as percentage or as raw value. Defaults to True.
        Returns:
            list[float]: List of proportions corresponding to each threshold.

        Raises:
            KeyError: If the specified threshold_metric or proportion_metric is not found in the DataFrame
        """

        threshold_values = plot_data_frame[threshold_metric.value].to_numpy()
        # temp = np.mean(threshold_values)
        # print(np.mean(threshold_values), np.std(threshold_values))
        # print((threshold_values <= temp).sum())
        if proportion_metric == ROCProportionMetricsEnum.JOBS:
            total_count = len(plot_data_frame)
            if plot_percentage:
                proportions = [
                    (threshold_values <= threshold).sum() / total_count * 100 for threshold in thresholds_arr
                ]
            else:
                proportions = [(threshold_values <= threshold).sum() for threshold in thresholds_arr]
        else:
            metric_values = plot_data_frame[proportion_metric.value].to_numpy()
            total_sum = metric_values.sum()

            if total_sum == 0:
                return np.zeros(len(thresholds_arr))

            proportions = []
            for threshold in thresholds_arr:
                mask = threshold_values <= threshold
                proportion = (
                    metric_values[mask].sum() / total_sum * 100 if plot_percentage else metric_values[mask].sum()
                )
                proportions.append(proportion)
        return np.array(proportions)

    def plot_roc(
        self,
        input_df: pd.DataFrame | None = None,
        title: str | None = None,
        min_threshold: float = 0.0,
        max_threshold: float = 100.0,
        threshold_step: float = 1.0,
        threshold_metric: JobEfficiencyMetricsEnum = JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY,
        proportion_metric: ROCProportionMetricsEnum = ROCProportionMetricsEnum.JOBS,
        plot_percentage: bool = True,
    ) -> tuple[Figure, list[Axes]]:
        """
        Plot the ROC curve based on the specified threshold and proportion metrics.

        Args:
            dataframe (pd.DataFrame | None): The data to plot. If None, uses the instance's job_metrics dataframe.
            title (str | None): Title of the plot. Defaults to None.
            min_threshold (float): Minimum threshold value. Defaults to 0.0.
            max_threshold (float): Maximum threshold value. Defaults to 100.0.
            threshold_step (float): Step size for thresholds. Defaults to 1.0.
            threshold_metric (JobEfficiencyMetricsEnum): Metric used for thresholds. Defaults to ALLOC_VRAM_EFFICIENCY.
            proportion_metric (ROCProportionMetricsEnum): Metric for calculating proportions. Defaults to JOBS.
            plot_percentage (bool): Whether to plot the proportion as a percentage or as raw counts. Defaults to True.

        Returns:
            tuple[Figure, list[Axes], int]: A tuple containing the figure, list of axes.

        Raises:
            KeyError: If the specified threshold_metric or proportion_metric is not found in the DataFrame
            ValueError: If the min_threshold is greater than max_threshold or if the threshold_step is not positive.
            ValueError: If no data is available for the specified threshold metric.
        """

        data = self.jobs_with_efficiency_metrics
        if input_df is not None:
            data = input_df

        # Validate inputs
        self._validate_inputs(
            data,
            min_threshold,
            max_threshold,
            threshold_step,
            threshold_metric,
            proportion_metric,
        )

        plot_data: pd.DataFrame = data[data[threshold_metric.value].notna()].copy()

        if plot_data.empty:
            raise ValueError("No data available for the specified threshold metric.")

        null_data_length = len(data) - len(plot_data)

        thresholds_arr: np.ndarray = np.arange(min_threshold, max_threshold + threshold_step, threshold_step)
        fig, axe = plt.subplots(1, 1, figsize=(16, 6))
        axe_list = [axe]

        # plotting
        proportions_data = self._roc_calculate_proportion(
            plot_data, proportion_metric, thresholds_arr, threshold_metric, plot_percentage
        )
        if title is None:
            title = (
                f"ROC plot for {'proportion' if plot_percentage else 'amounts'} of "
                f"{proportion_metric.value} by threshold {threshold_metric.value}"
            )
        if null_data_length:
            print(f"Amount of entries whose {threshold_metric.value} is null: {null_data_length}")
        y_label = f"{'Percentage' if plot_percentage else 'Count'} of {proportion_metric.value} below threshold"
        axe.set_title(title)
        axe.set_ylabel(y_label)
        axe.set_xlabel(f"Threshold values ({threshold_metric.value})")
        axe.plot(thresholds_arr, proportions_data)
        return fig, axe_list
