"""
Design template for ROC:

Chris's code very useful. Generally, I can follow the format of 2 functions like that.
But I just need to validate it, raise error properly, and make sure it can work with multiple other columns.
Also need to handle the NULL values properly and optimize Chris codes

Actions item:
- Bring the 2 functions in Chris's code to this class DONE
- Refactor the calculate effciency threshold function DONE
- For the plot(), don't focus on group by for now, implement validation, errors raise, null handling for columns DONE
- how can we also allow options for filtering  -> allow passing filtered dataframe. DONE
- Extend the plot() to accept some other columns (vram_hours, users) DONE
- Need another function that have multiple plot lines, for example for different users or pi_group

TODO (Tan): consider writing tests for validate_input function
Low priority:
- For each some threshold, can plot the data of y axis on the plot
- explore group by option
- can try to integrate the aggregation metrics (in case we want to group by)
- refactor roc_plot to use less arguments, consider printing out statistics info of thresholds
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
            ValueError: If min_threshold is greater than max_threshold.
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

    # TODO (Tan): fixed the vectorized version commented below, currently an issue when run alloc_vram_efficiency_score
    # def _roc_calculate_proportion(
    #     self,
    #     plot_data_frame: pd.DataFrame,
    #     proportion_metric: ROCProportionMetricsEnum,
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
    #     if proportion_metric == ROCProportionMetricsEnum.JOBS:
    #         comparison_mask = threshold_metric_column[:, np.newaxis] <= thresholds_arr
    #         proportions = comparison_mask.sum(axis=0)
    #         if plot_percentage:
    #             proportions = (proportions / len(threshold_metric_column)) * 100
    #     else:
    #         proportion_metric_column = plot_data_frame[proportion_metric.value].to_numpy()
    #         count_unique_proportion_metric = {ROCProportionMetricsEnum.USER, ROCProportionMetricsEnum.PI_GROUP}
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
        proportion_metric: ROCProportionMetricsEnum,
        thresholds_arr: np.ndarray[float],
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
            thresholds_arr (np.ndarray[float]): List of predefined threshold values.
            threshold_metric (EfficiencyMetricsJobsEnum): The specific efficiency metric used as thresholds_arr.
            plot_percentage (bool): Whether to return the proportion as percentage or as raw value. Defaults to True.
        Returns:
            np.ndarray[float]: List of proportions corresponding to each threshold.

        """
        threshold_values = plot_data_frame[threshold_metric.value].to_numpy()
        if proportion_metric == ROCProportionMetricsEnum.JOBS:
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
            count_unique_proportion_metric = {ROCProportionMetricsEnum.USER, ROCProportionMetricsEnum.PI_GROUP}
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
                    return np.zeros(len(thresholds_arr))

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

        Before plotting, this will filter out entries whose threshold_metric is NaN.
        In case where threshold_metric is ALLOC_VRAM_EFFICIENCY_SCORE:
            - Filter out entries where the threshold_metric is -inf.
            - If min_threshold is not provided or is 0, set min_threshold to the minimum value of the threshold_metric.

        Args:
            dataframe (pd.DataFrame or None): The data to plot. If None, uses the instance's job_metrics dataframe.
            title (str or None): Title of the plot. Defaults to None.
            min_threshold (float): Minimum threshold value. Defaults to 0.0.
            max_threshold (float): Maximum threshold value. Defaults to 100.0.
            threshold_step (float): Step size for thresholds. Defaults to 1.0.
            threshold_metric (JobEfficiencyMetricsEnum): Metric used for thresholds. Defaults to ALLOC_VRAM_EFFICIENCY.
            proportion_metric (ROCProportionMetricsEnum): Metric for calculating proportions. Defaults to JOBS.
            plot_percentage (bool): Whether to plot the proportion as a percentage or as raw counts. Defaults to True.

        Returns:
            tuple[Figure, list[Axes]]: A tuple containing the figure, list of axes.


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
        if null_data_length:
            print(f"Amount of entries whose {threshold_metric.value} is null: {null_data_length}")

        if threshold_metric == JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY_SCORE:
            prev_length = len(plot_data)
            plot_data = plot_data[plot_data[threshold_metric.value] != -np.inf].copy()
            print(f"Amount of entries whose {threshold_metric.value} is -inf : {prev_length - len(plot_data)}")
            if min_threshold >= 0.0:
                min_threshold = plot_data[threshold_metric.value].min()
                print(f"Setting min_threshold to {min_threshold} based on data.")

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
        y_label = f"{'Percentage' if plot_percentage else 'Count'} of {proportion_metric.value} below threshold"
        axe.set_title(title)
        axe.set_ylabel(y_label)
        axe.set_xlabel(f"Threshold values ({threshold_metric.value})")
        axe.plot(thresholds_arr, proportions_data)
        return fig, axe_list

    # def multiple_line_roc_plot(
    #     self,
    #     input_df: pd.DataFrame | None = None,
    #     title: str | None = None,
    #     min_threshold: float = 0.0,
    #     max_threshold: float = 100.0,
    #     threshold_step: float = 1.0,
    #     threshold_metric: JobEfficiencyMetricsEnum = JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY,
    #     proportion_metric: ROCProportionMetricsEnum.USER
    #     | ROCProportionMetricsEnum.PI_GROUP = ROCProportionMetricsEnum.USER,
    #     plot_percentage: bool = True,
    # ) -> None:
    #     """
    #     Plot ROC curve for User/ Pi group given threshold_metrics.

    #     Before plotting, this will filter out entries whose threshold_metric is NaN.
    #     In case where threshold_metric is ALLOC_VRAM_EFFICIENCY_SCORE:
    #         - Filter out entries where the threshold_metric is -inf.
    #         - If min_threshold is not provided or is 0, set min_threshold to the minimum value of the threshold_metric.

    #     Args:
    #         dataframe (pd.DataFrame or None): The data to plot. If None, uses the instance's job_metrics dataframe.
    #         title (str or None): Title of the plot. Defaults to None.
    #         min_threshold (float): Minimum threshold value. Defaults to 0.0.
    #         max_threshold (float): Maximum threshold value. Defaults to 100.0.
    #         threshold_step (float): Step size for thresholds. Defaults to 1.0.
    #         threshold_metric (JobEfficiencyMetricsEnum): Metric used for thresholds. Defaults to ALLOC_VRAM_EFFICIENCY.
    #         proportion_metric (USER or PI_GROUP in ROCProportionMetricsEnum): Metric for calculating proportions.
    #             Defaults to User
    #         plot_percentage (bool): Whether to plot the proportion as a percentage or as raw counts. Defaults to True.

    #     Returns:
    #         tuple[Figure, list[Axes]]: A tuple containing the figure, list of axes.
    #     """
