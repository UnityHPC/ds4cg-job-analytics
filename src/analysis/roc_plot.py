"""
Design template for ROC:

Chris's code very useful. Generally, I can follow the format of 2 functions like that.
But I just need to validate it, raise error properly, and make sure it can work with multiple other columns.
Also need to handle the NULL values properly and optimize Chris codes

Actions item:
- Bring the 2 functions in Chris's code to this class
- Refactor the calculate effciency threshold function
- For the plot(), don't focus on group by for now, implement validation, errors raise, null handling for job counts + GPU Hours
- Extend the plot() to accept some other columns (vram_hours, users)

Low priority:
- explore group by option
- can try to integrate the aggregation metrics (in case we want to group by)
- how can we also allow options for filtering (dynamically, without creating a whole new instance)??
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
from src.config.enum_constants import ROCProportionMetricsEnum, EfficiencyMetricsJobsEnum


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

    def plot_roc(
        self,
        title: str | None = None,
        threshold_step: float = 1.0,
        max_threshold: float = 100.0,
        threshold_metric: EfficiencyMetricsJobsEnum = EfficiencyMetricsJobsEnum.ALLOC_VRAM_EFFICIENCY.value,
        proportion_metric: ROCProportionMetricsEnum = ROCProportionMetricsEnum.JOB_NUMS,
    ) -> tuple[Figure, list[Axes]]:
        """
        Plots the ROC curve for the number of jobs, job_hours columns by varied efficiency thresholds_arr.

        If metrics argument is not given, will just plot the
        """

        plot_data: pd.DataFrame = self.jobs_with_efficiency_metrics.copy()
        # create threshold range
        thresholds_arr: np.ndarray = np.arange(0, max_threshold + threshold_step, threshold_step)
        fig, axe = plt.subplots(1, 1, fig_size=(16, 6))
        axe_list = [axe]

        # plotting
        proportions_data = self._roc_calculate_proportion(
            plot_data, proportion_metric, thresholds_arr, threshold_metric
        )
        if title is None:
            title = f"ROC plot for proportion of {proportion_metric.value} by threshold {threshold_metric.value}"
        axe.set_title(title)
        axe.set_ylabel(f"Percentage of {proportion_metric.value} below threshold (%)")
        axe.set_xlabel(f"Threshold values ({threshold_metric.value})")
        axe.plot(thresholds_arr, proportions_data)
        return fig, axe_list

    def _roc_calculate_proportion(
        self,
        plot_data_frame: pd.DataFrame,
        proportion_metric: ROCProportionMetricsEnum,
        thresholds_arr: list[float],
        threshold_metric: EfficiencyMetricsJobsEnum,
    ) -> list[float]:
        """
        Calculate the proportion of data that meet the alloc_vram_efficiency threshold for each threshold value.

        For each given threshold, this function will calculate the proportion of data (in terms of the
            specified metric) whose alloc_vram_efficiency is less than or equal to the threshold.

        Args:
            plot_data_frame (pd.DataFrame): DataFrame containing the data to plot.
            proportion_metric (ROCMetricsEnum): The metric to calculate proportions for.
            thresholds_arr (list[float]): List of predefined threshold values.
            threshold_metric (EfficiencyMetricsJobsEnum): The specific efficiency metric used as thresholds_arr.

        Returns:
            list[float]: List of proportions corresponding to each threshold.
        """

        #!Ensure valid threshold
        # TODO (Tan): what if threshold_metric does not exist in plot_data_frame
        plot_data_frame[threshold_metric] = np.clip(plot_data_frame[threshold_metric], 0, 100)
        # Ensure the last threshold is at least 100 to include all jobs
        if thresholds_arr[-1] < 100:
            thresholds_arr = np.append(thresholds_arr, 100)  # add 100 to the thresholds_arr array

        total_data_proportion = None
        if proportion_metric == ROCProportionMetricsEnum.JOB_NUMS:
            total_data_proportion = len(plot_data_frame)
        else:
            total_data_proportion = plot_data_frame[proportion_metric].sum()

        res = []

        # caculate proportion of jobs under each threshold
        for threshold in thresholds_arr:
            data_below = None

            if proportion_metric == ROCProportionMetricsEnum.JOB_NUMS:
                data_below = len(plot_data_frame[threshold_metric] <= threshold)

            else:
                data_below = plot_data_frame[plot_data_frame[threshold_metric] <= threshold].copy().sum()

            data_pct = (data_below / total_data_proportion) * 100 if total_data_proportion > 0 else 0
            res.append(data_pct)

        return res
