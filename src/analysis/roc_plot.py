"""
TODO (Tan): consider writing tests for validate_input function

Note somewhere: if we want to look at a user but with proportion metrics as some jobb metrics (jobs, job_hours),
then we shuould use the multiple line plot job since that still calculate the job metrics but per user/ pi_group.
THe group by thing for user and pi group maybe used for only number. of user/ pi_group metrics to inspect
    aggregated score.

Actions item:
- Add clipping options
- Add the annotation percentage of database, clipping for multiple_line plots
- Update miultiple_line plots to use other functions
- explore group by option + aggregation metrics
- refactor roc_plot to use less arguments, consider printing out statistics info of thresholds
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import sys
from .efficiency_analysis import EfficiencyAnalysis
from typing import Literal

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.enum_constants import ProportionMetricsEnum, JobEfficiencyMetricsEnum
import warnings


class ROCVisualizer(EfficiencyAnalysis):
    """
    A class for visualizing ROC plots, inheriting from EfficiencyAnalysis.
    """

    def __init__(self, jobs_df: pd.DataFrame) -> None:
        super().__init__(jobs_df)

    def _format_number_for_display(self, value: float) -> str:
        """
        Format a number for display, using scientific notation for large values.

        Args:
            value (float): The number to format.

        Returns:
            str: The formatted number as a string.
        """

        if abs(value) >= 1e6:
            return f"{value:.1e}"
        elif abs(value) >= 1000:
            return f"{value:.0f}"
        else:
            return f"{value:.1f}"

    def _helper_filter_invalid_records(
        self, plot_data_frame: pd.DataFrame, threshold_metric: JobEfficiencyMetricsEnum
    ) -> pd.DataFrame:
        """
        Filter out invalid records from the DataFrame based on the threshold metric.

        Currently invalid records are those with NaN or -inf values in the threshold metric column.
        Intended to be used after validating dataframe by _validate_inputs() only.

        Args:
            plot_data_frame (pd.DataFrame): The DataFrame to filter.
            threshold_metric (JobEfficiencyMetricsEnum): The metric used for filtering.

        Returns:
            pd.DataFrame: A new Dataframe that is filtered.

        """
        # contain all metrics we want to filter out invalid record
        metric_to_filters = {
            JobEfficiencyMetricsEnum.VRAM_CONSTRAINT_EFFICIENCY,
            JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY_SCORE,
        }

        # no filter needed
        if threshold_metric not in metric_to_filters:
            return plot_data_frame.copy()
        col = plot_data_frame[threshold_metric.value]
        mask = pd.Series([True] * len(col), index=col.index)
        mask &= col.notna()
        mask &= col != -np.inf

        return plot_data_frame[mask].copy()

    def _validate_and_filter_inputs(
        self,
        input_df: pd.DataFrame,
        min_threshold: float,
        max_threshold: float,
        threshold_step: float,
        threshold_metric: JobEfficiencyMetricsEnum,
        proportion_metric: ProportionMetricsEnum,
    ) -> tuple[pd.DataFrame, float]:
        """
        Validate the input fields and filter out invalid records.

        Args:
            threshold_metric (JobEfficiencyMetricsEnum): The metric used for thresholds.
            proportion_metric (ProportionMetricsEnum): The metric for calculating proportions.
            threshold_step (float): Step size for thresholds.
            min_threshold (float): Minimum threshold value.
            max_threshold (float): Maximum threshold value.

        Returns:
            pd.Dataframe: The filtered dataframe.
            float: The percentage of filtered data over the whole dataset.

        Raises:
            KeyError: If the specified metrics are not found in the DataFrame.
            ValueError: If the threshold step is not a positive number.
            ValueError: If min_threshold is greater than max_threshold.

        Warnings:
            UserWarning: if the filtered dataframe is empty.
            UserWarning: if threshold_step is too small in comparison to the threshold range
                defined by min_threshold and max_threshold.
        """

        # check if provided parameter is valid or not
        if threshold_metric and threshold_metric.value not in input_df.columns:
            raise KeyError(f"Threshold metric '{threshold_metric.value}' not found in DataFrame.")
        if (
            proportion_metric
            and proportion_metric.value != ProportionMetricsEnum.JOBS.value
            and proportion_metric.value not in input_df.columns
        ):
            raise KeyError(f"Proportion metric '{proportion_metric.value}' not found in DataFrame.")
        if threshold_step <= 0:
            raise ValueError("Threshold step must be a positive number.")
        if min_threshold > max_threshold:
            raise ValueError("min_threshold cannot be greater than max_threshold.")

        # handle filtering invalid values
        plot_data: pd.DataFrame = self._helper_filter_invalid_records(input_df, threshold_metric)
        if plot_data.empty:
            warnings.warn("No data available for the specified threshold metric.", stacklevel=2, category=UserWarning)

        # calculate number of filtered records and provide information
        filtered_out_records = len(input_df) - len(plot_data)
        if filtered_out_records:
            print(f"Filtered out {filtered_out_records} invalid records based on {threshold_metric.value} column.")
        if threshold_metric == JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY_SCORE and min_threshold >= 0.0:
            min_threshold = plot_data[threshold_metric.value].min()
            print(f"Setting min_threshold to {min_threshold} based on data.")

        # check if threshold step is not too small in comparison to the range
        distance = max_threshold - min_threshold
        if threshold_step / distance <= 10 ** (-6):
            # raise warning if threshold_step is too small in comparison to min_threshold
            warnings.warn(
                (
                    f"Minimum threshold value is {min_threshold}, but step size is {threshold_step}. "
                    "This may lead to high computational cost and time."
                ),
                stacklevel=2,
                category=UserWarning,
            )

        # calculate percentage of plot_data in comparison to total dataset
        remain_percentage = (len(input_df) - filtered_out_records) / len(input_df) * 100

        return plot_data, remain_percentage

    def _generate_num_marker(
        self,
        axe: Axes,
        thresholds_arr: np.ndarray,
        proportions_data: np.ndarray,
        num_markers: int,
    ) -> None:
        """
        Generate markers on the ROC plot at specified intervals.

        Args:
            axe (Axes): The axes on which to plot the markers.
            thresholds_arr (np.ndarray): Array of threshold values (x-axis).
            proportions_data (np.ndarray): Array of proportion values (y-axis).
            num_markers (int): Number of markers to generate.

        Returns:
            None

        Raises:
            ValueError: if num_markers provided is negative
        """
        if num_markers < 0:
            raise ValueError("Invalid num_marker parameter")
        step = len(thresholds_arr) / num_markers
        marker_indices = [int(i * step) for i in range(num_markers)]
        for idx in marker_indices:
            x_val = thresholds_arr[idx]
            y_val = proportions_data[idx]

            # Add marker
            axe.plot(x_val, y_val, "go", markersize=5, zorder=5)

            # custom format string
            x_formatted = self._format_number_for_display(x_val)
            y_formatted = self._format_number_for_display(y_val)

            # Add text label showing (x, y) values
            axe.annotate(
                f"({x_formatted}, {y_formatted})",
                (x_val, y_val),
                fontsize=10,
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    def _clip_upper_threshold_metric(
        self, clip_threshold_metric: tuple, plot_data_frame: pd.DataFrame, threshold_metric: JobEfficiencyMetricsEnum
    ) -> None:
        """
        Clip the column values down to the given upper bound.

        Args:
            clip_threshold_metric (tuple): A tuple where the first element is a boolean indicating whether to
                clip the threshold metrics, and the second element is the upper value to clip to.
            plot_data_frame (pd.DataFrame): The DataFrame containing the data to plot.
            threshold_metrics (JobEfficiencyMetricsEnum): The metric used for thresholds.

        Raises:
            ValueError: If the clip_threshold_metric parameter is not a tuple containing a boolean and a float.
        """
        if (
            len(clip_threshold_metric) != 2
            or not isinstance(clip_threshold_metric[0], bool)
            or not isinstance(clip_threshold_metric[1], float)
        ):
            raise ValueError(
                "Invalid clip_threshold_metric parameter provided. "
                "Parameter must be a tuple containing a booean value and a numeric values"
            )
        to_clip, upper_bound = clip_threshold_metric
        if to_clip:
            plot_data_frame[threshold_metric.value] = plot_data_frame[threshold_metric.value].clip(upper=upper_bound)

    # TODO (Tan): fixed the vectorized version commented below, currently an issue when run alloc_vram_efficiency_score
    # def _roc_calculate_proportion(
    #     self,
    #     plot_data_frame: pd.DataFrame,
    #     proportion_metric: ProportionMetricsEnum,
    #     thresholds_arr: np.ndarray[float],
    #     threshold_metric: JobEfficiencyMetricsEnum,
    #     plot_percentage: bool = True,
    # ) -> np.ndarray[float]:
    #     """
    #     Calculate the proportion of data that meet the alloc_vram_efficiency threshold for each threshold value.

    #     For each given threshold, this function will calculate the proportion of data (in terms of the
    #         specified metric) whose alloc_vram_efficiency is less than or equal to the threshold.

    #     Args:
    #         plot_data_frame (pd.DataFrame): DataFrame containing the data to plot.
    #         proportion_metric (ROCMetricsEnum): The metric to calculate proportions for.
    #         thresholds_arr (np.ndarray[float]): List of predefined threshold values.
    #         threshold_metric (EfficiencyMetricsJobsEnum): The specific efficiency metric used as thresholds_arr.
    #         plot_percentage (bool): Whether to return the proportion as percentage or as raw value. Defaults to True.
    #     Returns:
    #         np.ndarray[float]: List of proportions corresponding to each threshold.

    #     """
    #     proportions = []
    #     threshold_metric_column = plot_data_frame[threshold_metric.value].to_numpy()
    #     if proportion_metric == ProportionMetricsEnum.JOBS:
    #         comparison_mask = threshold_metric_column[:, np.newaxis] <= thresholds_arr
    #         proportions = comparison_mask.sum(axis=0)
    #         if plot_percentage:
    #             proportions = (proportions / len(threshold_metric_column)) * 100
    #     else:
    #         proportion_metric_column = plot_data_frame[proportion_metric.value].to_numpy()
    #         count_unique_proportion_metric = {ProportionMetricsEnum.USER, ProportionMetricsEnum.PI_GROUP}
    #         # check if we are dealing with USER/ PI_GROUP metrics
    #         if proportion_metric in count_unique_proportion_metric:
    #             total_unique = len(np.unique(proportion_metric_column))
    #             for threshold in thresholds_arr:
    #                 mask = threshold_metric_column <= threshold
    #                 unique_count = len(np.unique(proportion_metric_column[mask]))
    #                 res = unique_count / total_unique * 100 if plot_percentage else unique_count
    #                 proportions.append(res)
    #         else:
    #             total_sum = proportion_metric_column.sum()
    #             if total_sum == 0:
    #                 return np.zeros(len(thresholds_arr))
    #             # broadcast comparison to each threshold to each data in threshold_metric_column
    #             comparison_mask = threshold_metric_column[:, np.newaxis] <= thresholds_arr
    #             weighted_filtered_matrix = proportion_metric_column[:, np.newaxis] * comparison_mask
    #             proportions = weighted_filtered_matrix.sum(axis=0)
    #             if plot_percentage:
    #                 proportions = proportions / total_sum * 100
    #     return np.array(proportions).astype(float)

    def _roc_calculate_proportion(
        self,
        plot_data_frame: pd.DataFrame,
        proportion_metric: ProportionMetricsEnum,
        thresholds_arr: np.ndarray,
        threshold_metric: JobEfficiencyMetricsEnum,
        plot_percentage: bool = True,
    ) -> np.ndarray:
        """
        Calculate the proportion of data that meet the alloc_vram_efficiency threshold for each threshold value.

        For each given threshold, this function will calculate the proportion of data (in terms of the
            specified metric) whose alloc_vram_efficiency is less than or equal to the threshold.

        Args:
            plot_data_frame (pd.DataFrame): DataFrame containing the data to plot.
            proportion_metric (ROCMetricsEnum): The metric to calculate proportions for.
            thresholds_arr (np.ndarray): List of predefined threshold values.
            threshold_metric (EfficiencyMetricsJobsEnum): The specific efficiency metric used as thresholds_arr.
            plot_percentage (bool): Whether to return the proportion as percentage or as raw value. Defaults to True.

        Returns:
            np.ndarray: List of proportions corresponding to each threshold.

        """
        threshold_values = plot_data_frame[threshold_metric.value].to_numpy(dtype=float)
        if proportion_metric == ProportionMetricsEnum.JOBS:
            total_count = len(plot_data_frame)
            if plot_percentage:
                proportions = [
                    (threshold_values <= threshold).sum() / total_count * 100 for threshold in thresholds_arr
                ]
            else:
                proportions = [(threshold_values <= threshold).sum() for threshold in thresholds_arr]
        else:
            proportions = []
            metric_values = plot_data_frame[proportion_metric.value].to_numpy()
            count_unique_proportion_metric = {ProportionMetricsEnum.USER, ProportionMetricsEnum.PI_GROUP}
            # check if we are dealing with USER metrics
            if proportion_metric in count_unique_proportion_metric:
                total_unique = len(np.unique(metric_values))
                for threshold in thresholds_arr:
                    mask = threshold_values <= threshold
                    unique_count = len(np.unique(metric_values[mask]))
                    proportion = unique_count / total_unique * 100 if plot_percentage else unique_count
                    proportions.append(proportion)
            else:
                total_sum = metric_values.sum()

                if total_sum == 0:
                    return np.zeros(len(thresholds_arr), dtype=float)

                for threshold in thresholds_arr:
                    mask = threshold_values <= threshold
                    proportion = (
                        metric_values[mask].sum() / total_sum * 100 if plot_percentage else metric_values[mask].sum()
                    )
                    proportions.append(proportion)
        return np.array(proportions, dtype=float)

    def plot_roc_jobs(
        self,
        title: str | None = None,
        min_threshold: float = 0.0,
        max_threshold: float = 100.0,
        threshold_step: float = 1.0,
        threshold_metric: JobEfficiencyMetricsEnum = JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY,
        proportion_metric: ProportionMetricsEnum = ProportionMetricsEnum.JOBS,
        plot_percentage: bool = True,
        num_markers: int = 10,
        clip_threshold_metric: tuple[bool, float] = (False, 0.0),
    ) -> tuple[Figure, list[Axes]]:
        """
        Plot the ROC curve based on the specified threshold and proportion metrics.

        Before plotting, this will filter out entries whose threshold_metric is NaN.

        In case where threshold_metric is ALLOC_VRAM_EFFICIENCY_SCORE:
            - Filter out entries where the threshold_metric is -inf.
            - If min_threshold is not provided or is 0, set min_threshold to the minimum value of the threshold_metric.

        If plot by percentage, percentage will be based on the filtered data, not the original data.

        If clip_threshold_metrics is True, then threshold will be clipped as following:


        Args:
            title (str or None): Title of the plot. Defaults to None.
            min_threshold (float): Minimum threshold value. Defaults to 0.0.
            max_threshold (float): Maximum threshold value. Defaults to 100.0.
            threshold_step (float): Step size for thresholds. Defaults to 1.0.
            threshold_metric (JobEfficiencyMetricsEnum): Metric used for thresholds. Defaults to ALLOC_VRAM_EFFICIENCY.
            proportion_metric (ProportionMetricsEnum): Metric for calculating proportions. Defaults to JOBS.
            plot_percentage (bool): Whether to plot the proportion as a percentage or as raw counts. Defaults to True.
            clip_metrics (tuple[bool, int]): A tuple where the first element is a boolean indicating whether to
                clip the threshold metrics, and the second element is the upper value to clip to.
                Defaults to (False, 0).

        Returns:
            tuple[Figure, list[Axes]]: A tuple containing the figure, list of axes.

        Raises:
            ValueError: if dataframe jobs_with_efficiency_metrics is not calculated yet.
        """
        if self.jobs_with_efficiency_metrics is None:
            raise ValueError(
                "Attribute jobs_with_efficiency_metrics is not calculated, "
                "use calculate_all_efficiency_metrics() to calculate the dataframe before plotting."
            )
        data = self.jobs_with_efficiency_metrics

        plot_data, remain_percentage = self._validate_and_filter_inputs(
            data,
            min_threshold,
            max_threshold,
            threshold_step,
            threshold_metric,
            proportion_metric,
        )

        # clip threshold_metrics to defined value
        self._clip_upper_threshold_metric(clip_threshold_metric, plot_data, threshold_metric)

        # calculate threshold
        thresholds_arr: np.ndarray = np.arange(min_threshold, max_threshold + threshold_step, threshold_step)
        fig, axe = plt.subplots(1, 1, figsize=(16, 7))
        axe_list = [axe]
        # calculate proportion for each threshold
        proportions_data = self._roc_calculate_proportion(
            plot_data, proportion_metric, thresholds_arr, threshold_metric, plot_percentage
        )

        # Create label with total count and percentage of plot_data for legend
        if proportion_metric == ProportionMetricsEnum.JOBS:
            total_raw_value = len(plot_data)
        elif proportion_metric in {ProportionMetricsEnum.USER, ProportionMetricsEnum.PI_GROUP}:
            total_raw_value = len(np.unique(plot_data[proportion_metric.value]))
        else:
            total_raw_value = plot_data[proportion_metric.value].sum()
        if title is None:
            title = (
                f"ROC plot for {'proportion' if plot_percentage else 'amounts'} of "
                f"{proportion_metric.value} by threshold {threshold_metric.value}"
            )
        title += (
            f"\n Total {proportion_metric.value}: {self._format_number_for_display(total_raw_value)} "
            f"(in {self._format_number_for_display(remain_percentage)}% of dataset)"
        )

        y_label = f"{'Percentage' if plot_percentage else 'Count'} of {proportion_metric.value} below threshold"
        axe.set_title(title)
        axe.set_ylabel(y_label)
        axe.set_xlabel(f"Threshold values ({threshold_metric.value})")
        axe.plot(thresholds_arr, proportions_data)
        axe.legend()

        # # add label to give information about total number of data and percentage to original dataset
        # custom_patch = mpatches.Patch(color="none", label=plot_label)
        # handles, _labels = axe.get_legend_handles_labels()
        # handles.append(custom_patch)
        # axe.legend(handles=handles)

        self._generate_num_marker(axe, thresholds_arr, proportions_data, num_markers)

        return fig, axe_list

    def multiple_line_roc_plot(
        self,
        plot_object_list: list[str],
        object_column_type: Literal[ProportionMetricsEnum.USER, ProportionMetricsEnum.PI_GROUP],
        title: str | None = None,
        min_threshold: float = 0.0,
        max_threshold: float = 100.0,
        threshold_step: float = 1.0,
        threshold_metric: JobEfficiencyMetricsEnum = JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY,
        proportion_metric: Literal[
            ProportionMetricsEnum.JOB_HOURS, ProportionMetricsEnum.JOBS
        ] = ProportionMetricsEnum.JOBS,
        plot_percentage: bool = True,
        clip_threshold_metric: tuple[bool, float] = (False, 0.0),
    ) -> tuple[Figure, list[Axes]]:
        """
        Plot ROC curve for User/ Pi group given threshold_metrics.

        This function will plot an ROC curve for each object, with proportion metrics is count of jobs.

        Before plotting, this will filter out entries whose threshold_metric is NaN.

        In case where threshold_metric is ALLOC_VRAM_EFFICIENCY_SCORE:
            - Filter out entries where the threshold_metric is -inf.
            - If min_threshold is not provided or is 0, set min_threshold to the minimum value of the threshold_metric.

        If plot ```proportion_metric``` by percentage, percentage will be calculated based on the total data per object
        (user/ pi_group) after filtering out invalid data, not by the whole database.


        Args:
            plot_object_list (list[str]): List
                of users/pi_group that we want to plot.
            dataframe (pd.DataFrame or None): The data to plot. If None, uses the instance's job_metrics dataframe.
            title (str or None): Title of the plot. Defaults to None.
            min_threshold (float): Minimum threshold value. Defaults to 0.0.
            max_threshold (float): Maximum threshold value. Defaults to 100.0.
            threshold_step (float): Step size for thresholds. Defaults to 1.0.
            threshold_metric (JobEfficiencyMetricsEnum): Metric used for thresholds. Defaults to ALLOC_VRAM_EFFICIENCY.
            plot_percentage (bool): Whether to plot the proportion as a percentage or as raw counts. Defaults to True.
            clip_metric (tuple[bool, int]): A tuple where the first element is a boolean indicating whether to
                clip the threshold metrics, and the second element is the upper value to clip to.
                Defaults to (False, 0).

        Returns:
            tuple[Figure, list[Axes]]: A tuple containing the figure, list of axes.

        Raises:
            ValueError: if dataframe jobs_with_efficiency_metrics is not calculated yet.
        """
        if self.jobs_with_efficiency_metrics is None:
            raise ValueError(
                "Attribute jobs_with_efficiency_metrics is not calculated, "
                "use calculate_all_efficiency_metrics() to calculate the dataframe before plotting."
            )

        data = self.jobs_with_efficiency_metrics
        plot_data, remain_percentage = self._validate_and_filter_inputs(
            data,
            min_threshold,
            max_threshold,
            threshold_step,
            threshold_metric,
            proportion_metric,
        )

        self._clip_upper_threshold_metric(clip_threshold_metric, plot_data, threshold_metric)

        thresholds_arr: np.ndarray = np.arange(min_threshold, max_threshold + threshold_step, threshold_step)
        fig, axe = plt.subplots(1, 1, figsize=(16, 6))
        for target in plot_object_list:
            filtered = plot_data[plot_data[object_column_type.value] == target].copy()
            proportion_data = self._roc_calculate_proportion(
                filtered, proportion_metric, thresholds_arr, threshold_metric, plot_percentage
            )
            axe.plot(thresholds_arr, proportion_data, label=f"{target}")

        # Create label with total count and percentage of plot_data for title
        if proportion_metric == ProportionMetricsEnum.JOBS:
            total_raw_value = len(plot_data)
        elif proportion_metric in {ProportionMetricsEnum.USER, ProportionMetricsEnum.PI_GROUP}:
            total_raw_value = len(np.unique(plot_data[proportion_metric.value]))
        else:
            total_raw_value = plot_data[proportion_metric.value].sum()

        if title is None:
            title = (
                f"Multple line ROC plot ({object_column_type.value}) for "
                f"{'proportion' if plot_percentage else 'amounts'} of "
                f"{proportion_metric.value} by threshold {threshold_metric.value}"
            )
        title += (
            f"\n Total {proportion_metric.value}: {self._format_number_for_display(total_raw_value)} "
            f"(in {self._format_number_for_display(remain_percentage)}% of dataset)"
        )
        y_label = f"{'Percentage' if plot_percentage else 'Count'} of {proportion_metric.value} below threshold"
        axe.set_title(title)
        axe.set_ylabel(y_label)
        axe.set_xlabel(f"Threshold values ({threshold_metric.value})")
        axe.legend()
        return fig, [axe]
