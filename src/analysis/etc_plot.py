from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import sys
from .efficiency_analysis import EfficiencyAnalysis
from typing import Literal, cast

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.enum_constants import (
    ProportionMetricsEnum,
    JobEfficiencyMetricsEnum,
    UserEfficiencyMetricsEnum,
    PIEfficiencyMetricsEnum,
    ETCPlotTypes,
)
import warnings


class ETCVisualizer(EfficiencyAnalysis):
    """
    A class for visualizing ETC plots, inheriting from EfficiencyAnalysis.
    """

    # map types to the its associate proportion metric
    TYPE_TO_ASSOCIATE_METRIC: dict[ETCPlotTypes, ProportionMetricsEnum] = {
        ETCPlotTypes.JOB: ProportionMetricsEnum.JOBS,
        ETCPlotTypes.USER: ProportionMetricsEnum.USERS,
        ETCPlotTypes.PI_GROUP: ProportionMetricsEnum.PI_GROUPS,
    }

    # For counting user/ pi group that has at least 1 job/user that falls under threshold
    TYPE_TO_COUNT_UNIQUE_METRIC: dict[ETCPlotTypes, set[ProportionMetricsEnum]] = {
        ETCPlotTypes.JOB: {
            ProportionMetricsEnum.USERS,
            ProportionMetricsEnum.PI_GROUPS,
        },  # JOB can have unique count of User/ pi_group that has at least 1 job under thresholds
        ETCPlotTypes.USER: {ProportionMetricsEnum.PI_GROUPS},
        # User can have unique count of pi_group that has at least 1 user under thresholds
        ETCPlotTypes.PI_GROUP: cast(set[ProportionMetricsEnum], {}),
    }

    # metrics that may contain NULL or -inf values
    METRICS_WITH_INVALID_VALUES = {
        JobEfficiencyMetricsEnum.VRAM_CONSTRAINT_EFFICIENCY,
        JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY_SCORE,
        JobEfficiencyMetricsEnum.VRAM_CONSTRAINT_EFFICIENCY_SCORE,
        UserEfficiencyMetricsEnum.EXPECTED_VALUE_VRAM_CONSTRAINTS_EFFICIENCY,
        UserEfficiencyMetricsEnum.EXPECTED_VALUE_ALLOC_VRAM_EFFICIENCY,
        UserEfficiencyMetricsEnum.AVG_ALLOC_VRAM_EFFICIENCY_SCORE,
        UserEfficiencyMetricsEnum.AVG_VRAM_CONSTRAINT_EFFICIENCY_SCORE,
        PIEfficiencyMetricsEnum.EXPECTED_VALUE_VRAM_CONSTRAINTS_EFFICIENCY,
        PIEfficiencyMetricsEnum.AVG_ALLOC_VRAM_EFFICIENCY_SCORE,
        PIEfficiencyMetricsEnum.AVG_VRAM_CONSTRAINT_EFFICIENCY_SCORE,
    }

    USER_TO_JOB_EFFICIENCY_METRIC_MAPPING = {
        UserEfficiencyMetricsEnum.EXPECTED_VALUE_ALLOC_VRAM_EFFICIENCY: (
            JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY
        ),
        UserEfficiencyMetricsEnum.EXPECTED_VALUE_VRAM_CONSTRAINTS_EFFICIENCY: (
            JobEfficiencyMetricsEnum.VRAM_CONSTRAINT_EFFICIENCY
        ),
        UserEfficiencyMetricsEnum.AVG_ALLOC_VRAM_EFFICIENCY_SCORE: (
            JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY_SCORE
        ),
        UserEfficiencyMetricsEnum.AVG_VRAM_CONSTRAINT_EFFICIENCY_SCORE: (
            JobEfficiencyMetricsEnum.VRAM_CONSTRAINT_EFFICIENCY_SCORE
        ),
        UserEfficiencyMetricsEnum.EXPECTED_VALUE_GPU_COUNT: (JobEfficiencyMetricsEnum.GPU_COUNT),
    }

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
        self,
        plot_data_frame: pd.DataFrame,
        threshold_metric: JobEfficiencyMetricsEnum | UserEfficiencyMetricsEnum | PIEfficiencyMetricsEnum,
    ) -> pd.DataFrame:
        """
        Filter out invalid records from the DataFrame based on the threshold metric.

        Currently invalid records are those with NaN or -inf values in the threshold metric column.
        Intended to be used after validating dataframe by _validate_inputs() only.

        Args:
            plot_data_frame (pd.DataFrame): The DataFrame to filter.
            threshold_metric (JobEfficiencyMetricsEnum | UserEfficiencyMetricsEnum): The metric used for filtering.

        Returns:
            pd.DataFrame: A new Dataframe that is filtered.

        """
        # contain all metrics we want to filter out invalid record

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
        threshold_metric: JobEfficiencyMetricsEnum | UserEfficiencyMetricsEnum | PIEfficiencyMetricsEnum,
        proportion_metric: ProportionMetricsEnum,
        plot_type: ETCPlotTypes = ETCPlotTypes.JOB,
    ) -> tuple[pd.DataFrame, float, float | int, float]:
        """
        Validate input parameters and filter invalid records for ETC plot generation.

        This method performs comprehensive validation of ETC plot parameters, filters out records
        with invalid threshold metric values (NaN or -inf), and calculates summary statistics.

        Args:
            input_df (pd.DataFrame): The input DataFrame containing efficiency metrics data.
            min_threshold (float): Minimum threshold value for the ETC curve. For efficiency
                score metrics, this may be automatically adjusted to the data minimum.
            max_threshold (float): Maximum threshold value for the ETC curve.
            threshold_step (float): Step size between threshold values. Must be positive.
            threshold_metric (JobEfficiencyMetricsEnum | UserEfficiencyMetricsEnum | PIEfficiencyMetricsEnum):
                The efficiency metric to use for threshold calculations (x-axis).
            proportion_metric (ProportionMetricsEnum): The metric for calculating proportions (y-axis).
            plot_type (ROCPlotTypes, optional): Type of ETC plot being generated. Defaults to JOB.

        Returns:
            tuple[pd.DataFrame, float, float | int, float]: A tuple containing:
                - Filtered DataFrame with invalid records removed
                - Percentage of data remaining after filtering (0-100)
                - Total value of the proportion metric in the filtered data
                - Adjusted minimum threshold value (may differ from input if auto-adjusted)

        Raises:
            KeyError: If threshold_metric or proportion_metric columns are not found in the DataFrame.
            ValueError: If threshold_step is not positive, min_threshold > max_threshold,
                or threshold_metric and proportion_metric have the same value.

        Warnings:
            UserWarning: Issued when the filtered DataFrame is empty
            UserWarning: threshold_step is extremely small relative to the threshold range
                (may cause performance issues).

        Note:
            For efficiency score metrics (ALLOC_VRAM_EFFICIENCY_SCORE, VRAM_CONSTRAINT_EFFICIENCY_SCORE),
            the min_threshold is automatically set to the minimum data value to handle negative scores.
        """

        # check if provided parameter is valid or not
        if threshold_metric.value == proportion_metric.value:
            # avoid same values for y-axis and x-axis
            raise ValueError("threshold_metric and proportion_metric cannot have the same value.")

        if threshold_metric.value not in input_df.columns:
            raise KeyError(f"Threshold metric '{threshold_metric.value}' not found in DataFrame.")
        if (
            proportion_metric.value != self.TYPE_TO_ASSOCIATE_METRIC[plot_type].value
            and proportion_metric.value not in input_df.columns
        ):
            raise KeyError(f"Proportion metric '{proportion_metric.value}' not found in DataFrame.")
        if threshold_step <= 0:
            raise ValueError("Threshold step must be a positive number.")
        if min_threshold > max_threshold:
            raise ValueError("min_threshold cannot be greater than max_threshold.")

        # for metrics that may reuire taking min of their values as minimum
        manual_min_threshold_setting = {
            JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY_SCORE,
            JobEfficiencyMetricsEnum.VRAM_CONSTRAINT_EFFICIENCY_SCORE,
            UserEfficiencyMetricsEnum.AVG_ALLOC_VRAM_EFFICIENCY_SCORE,
            UserEfficiencyMetricsEnum.AVG_VRAM_CONSTRAINT_EFFICIENCY_SCORE,
            PIEfficiencyMetricsEnum.AVG_ALLOC_VRAM_EFFICIENCY_SCORE,
            PIEfficiencyMetricsEnum.AVG_VRAM_CONSTRAINT_EFFICIENCY_SCORE,
        }

        # handle filtering invalid values
        plot_data: pd.DataFrame = self._helper_filter_invalid_records(input_df, threshold_metric)
        if plot_data.empty:
            warnings.warn("No data available for the specified threshold metric.", stacklevel=2, category=UserWarning)
        # calculate number of filtered records and provide information
        filtered_out_records = len(input_df) - len(plot_data)
        if filtered_out_records:
            print(f"Filtered out {filtered_out_records} invalid records based on {threshold_metric.value} column.")
        if threshold_metric in manual_min_threshold_setting and min_threshold >= 0.0:
            min_threshold = plot_data[threshold_metric.value].min()
            print(f"Setting min_threshold to {min_threshold} based on data.")

        # check if threshold step is not too small in comparison to the range
        distance = max_threshold - min_threshold
        if distance and (threshold_step / distance <= 10 ** (-6)):
            # raise warning if threshold_step is too small in comparison to min_threshold
            warnings.warn(
                (
                    f"Total range of thresholds is {distance}, but step size is {threshold_step}. "
                    "This may lead to high computational cost and time."
                ),
                stacklevel=2,
                category=UserWarning,
            )

        # calculate percentage of plot_data in comparison to total dataset
        remain_percentage = (len(input_df) - filtered_out_records) / len(input_df) * 100

        # caculate the total value of proportion metrics to declare in title
        if proportion_metric == self.TYPE_TO_ASSOCIATE_METRIC[plot_type]:
            total_raw_value = len(plot_data)
        elif proportion_metric in self.TYPE_TO_COUNT_UNIQUE_METRIC[plot_type]:
            total_raw_value = len(np.unique(plot_data[proportion_metric.value]))
        else:
            total_raw_value = plot_data[proportion_metric.value].sum()

        return plot_data, remain_percentage, total_raw_value, min_threshold

    def _generate_num_marker(
        self,
        axe: Axes,
        thresholds_arr: np.ndarray,
        proportions_data: np.ndarray,
        num_markers: int,
    ) -> None:
        """
        Generate markers on the ETC plot at specified intervals.

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
        self,
        clip_threshold_metric: tuple,
        plot_data_frame: pd.DataFrame,
        threshold_metric: JobEfficiencyMetricsEnum | UserEfficiencyMetricsEnum | PIEfficiencyMetricsEnum,
    ) -> None:
        """
        Conditionally clip threshold metric values to a specified upper bound.

        This method modifies the DataFrame in-place by capping threshold metric values at the specified
        upper bound.

        Args:
            clip_threshold_metric (tuple[bool, float]): A 2-element tuple where:
                - First element (bool): Whether to perform clipping
                - Second element (float): Upper bound value for clipping
            plot_data_frame (pd.DataFrame): The DataFrame containing the data to be modified.
                The threshold metric column will be clipped in-place.
            threshold_metric (JobEfficiencyMetricsEnum | UserEfficiencyMetricsEnum | PIEfficiencyMetricsEnum):
                The efficiency metric whose values will be clipped.

        Returns:
            None: The DataFrame is modified in-place.

        Raises:
            ValueError: If clip_threshold_metric is not a 2-element tuple containing a boolean
                and a float, or if the tuple format is invalid.

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

    def _etc_calculate_proportion(
        self,
        plot_data_frame: pd.DataFrame,
        thresholds_arr: np.ndarray,
        proportion_metric: ProportionMetricsEnum,
        threshold_metric: JobEfficiencyMetricsEnum | UserEfficiencyMetricsEnum | PIEfficiencyMetricsEnum,
        plot_percentage: bool = True,
        plot_type: ETCPlotTypes = ETCPlotTypes.JOB,
    ) -> np.ndarray:
        """
        Calculate proportions of data meeting threshold criteria for ETC curve generation.

        Args:
            plot_data_frame (pd.DataFrame): DataFrame containing the efficiency metrics data.
            thresholds_arr (np.ndarray): Array of threshold values to evaluate against.
            proportion_metric (ProportionMetricsEnum): The metric to calculate proportions for
                (e.g., JOBS, VRAM_HOURS, USERS, PI_GROUPS).
            threshold_metric (JobEfficiencyMetricsEnum | UserEfficiencyMetricsEnum | PIEfficiencyMetricsEnum):
                The efficiency metric used as the threshold criteria for filtering.
            plot_percentage (bool, optional): Whether to return proportions as percentages (0-100)
                or raw values. Defaults to True.
            plot_type (ROCPlotTypes, optional): Type of ETC plot being generated, which determines
                how proportion calculations are performed. Defaults to JOB.

        Returns:
            np.ndarray: Array of proportions corresponding to each threshold value. Length matches
                thresholds_arr. Values are percentages if plot_percentage=True, otherwise raw counts/sums.

        Note:
            - For associated metrics (e.g., JOBS for JOB plot type), counts records directly
            - For unique count metrics, counts distinct entities meeting the threshold
            - For sum-based metrics, sums the metric values for records meeting the threshold
            - Returns zeros array if total sum is zero to avoid division by zero
        """

        threshold_values = plot_data_frame[threshold_metric.value].to_numpy(dtype=float)

        # if proportion metric is associated to the plot type
        if proportion_metric == self.TYPE_TO_ASSOCIATE_METRIC[plot_type]:
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
            if proportion_metric in self.TYPE_TO_COUNT_UNIQUE_METRIC[plot_type]:
                total_unique = len(np.unique(metric_values))
                for threshold in thresholds_arr:
                    mask = threshold_values <= threshold
                    unique_count = len(np.unique(metric_values[mask]))
                    proportion = unique_count / total_unique * 100 if plot_percentage else unique_count
                    proportions.append(proportion)
            else:
                # add temporary handle for user graphs to handle vram_hours, job_count, job_hours plotting
                if plot_type == ETCPlotTypes.USER and isinstance(threshold_metric, UserEfficiencyMetricsEnum):
                    proportion_metric_to_new_calculation = {
                        ProportionMetricsEnum.JOB_HOURS: lambda jobs_df: (
                            jobs_df[JobEfficiencyMetricsEnum.JOB_HOURS.value]
                        )
                        .groupby(jobs_df["User"], observed=True)
                        .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
                        .to_numpy(),
                        ProportionMetricsEnum.VRAM_HOURS: lambda jobs_df: (
                            (jobs_df["allocated_vram"] * jobs_df[JobEfficiencyMetricsEnum.JOB_HOURS.value])
                            .groupby(jobs_df["User"], observed=True)
                            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
                            .to_numpy()
                        ),
                        ProportionMetricsEnum.JOBS: lambda jobs_df: (
                            (jobs_df["JobID"])
                            .groupby(jobs_df["User"], observed=True)
                            .apply(lambda series: len(series))
                        ),
                    }
                    # Use the filtered jobs data that excludes inf/NULL threshold_metric values
                    job_eff_column = self.USER_TO_JOB_EFFICIENCY_METRIC_MAPPING[threshold_metric].value
                    filtered_jobs_df = self.jobs_with_efficiency_metrics[
                        (self.jobs_with_efficiency_metrics[job_eff_column].notna())
                        & (self.jobs_with_efficiency_metrics[job_eff_column] != -np.inf)
                    ].copy()
                    new_proportion_metric_col = f"{proportion_metric.value} for {threshold_metric.value}"
                    plot_data_frame.loc[:, new_proportion_metric_col] = proportion_metric_to_new_calculation[
                        proportion_metric
                    ](filtered_jobs_df)
                    metric_values = plot_data_frame[new_proportion_metric_col].to_numpy()
                    print("Reached here")
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

    def plot_etc(
        self,
        plot_type: ETCPlotTypes,
        title: str | None = None,
        threshold_metric: JobEfficiencyMetricsEnum | UserEfficiencyMetricsEnum | PIEfficiencyMetricsEnum | None = None,
        min_threshold: float = 0.0,
        max_threshold: float = 100.0,
        threshold_step: float = 1.0,
        proportion_metric: ProportionMetricsEnum | None = None,
        plot_percentage: bool = True,
        num_markers: int = 10,
        clip_threshold_metric: tuple[bool, float] = (False, 0.0),
    ) -> tuple[Figure, list[Axes]]:
        """
        Generate an ETC (Receiver Operating Characteristic) plot for efficiency analysis.

        This function creates a single-line ETC plot showing the proportion of data below
        various threshold values for the specified efficiency metric. The plot can be
        configured for different analysis levels (jobs, users, or PI groups) and various
        proportion metrics.

        Args:
            plot_type: The type of analysis to perform (JOB, USER, or PI_GROUP).
            title: Custom title for the plot. If None, an automatic title will be
                generated based on the plot parameters. Defaults to None.
            threshold_metric: The efficiency metric to use for threshold analysis.
                If None, defaults are used:
                - JOB: ALLOC_VRAM_EFFICIENCY
                - USER: EXPECTED_VALUE_ALLOC_VRAM_EFFICIENCY
                - PI_GROUP: EXPECTED_VALUE_ALLOC_VRAM_EFFICIENCY
                Defaults to None.
            min_threshold: Minimum threshold value for the analysis. Defaults to 0.0.
            max_threshold: Maximum threshold value for the analysis. Defaults to 100.0.
            threshold_step: Step size between threshold values. Defaults to 1.0.
            proportion_metric: The metric to calculate proportions for. If None,
                defaults to JOBS for all plot types. Defaults to None.
            plot_percentage: If True, plots proportions as percentages. If False,
                plots raw counts. Defaults to True.
            num_markers: Number of markers to display on the plot line. Defaults to 10.
            clip_threshold_metric: A tuple where the first element indicates whether
                to clip the threshold metric values, and the second element is the
                upper bound for clipping. Defaults to (False, 0.0).

        Returns:
            A tuple containing the matplotlib Figure object and a list with a single
            Axes object.

        Raises:
            ValueError: If the required efficiency metrics DataFrame is not calculated.
                Use calculate_all_efficiency_metrics() before calling this function.

        Notes:
            - Invalid records (NaN, -inf values) are automatically filtered out
            - For VRAM efficiency score metrics, -inf values are filtered and min_threshold
              may be adjusted to the minimum valid value
            - The plot includes statistical information in the title showing total values
              and percentage of valid data used
        """
        # Configuration mappings for different plot types
        data_source_map = {
            ETCPlotTypes.JOB: self.jobs_with_efficiency_metrics,
            ETCPlotTypes.USER: self.users_with_efficiency_metrics,
            ETCPlotTypes.PI_GROUP: self.pi_accounts_with_efficiency_metrics,
        }

        default_threshold_metrics: dict[
            ETCPlotTypes, JobEfficiencyMetricsEnum | UserEfficiencyMetricsEnum | PIEfficiencyMetricsEnum
        ] = {
            ETCPlotTypes.JOB: JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY,
            ETCPlotTypes.USER: UserEfficiencyMetricsEnum.EXPECTED_VALUE_ALLOC_VRAM_EFFICIENCY,
            ETCPlotTypes.PI_GROUP: PIEfficiencyMetricsEnum.EXPECTED_VALUE_ALLOC_VRAM_EFFICIENCY,
        }

        default_proportion_metrics: dict[ETCPlotTypes, ProportionMetricsEnum] = {
            ETCPlotTypes.JOB: ProportionMetricsEnum.JOBS,
            ETCPlotTypes.USER: ProportionMetricsEnum.JOBS,
            ETCPlotTypes.PI_GROUP: ProportionMetricsEnum.JOBS,
        }

        attribute_names = {
            ETCPlotTypes.JOB: "jobs_with_efficiency_metrics",
            ETCPlotTypes.USER: "users_with_efficiency_metrics",
            ETCPlotTypes.PI_GROUP: "pi_accounts_with_efficiency_metrics",
        }

        plot_type_labels = {
            ETCPlotTypes.JOB: "Jobs",
            ETCPlotTypes.USER: "Users",
            ETCPlotTypes.PI_GROUP: "PI Group",
        }

        dataset_descriptions = {
            ETCPlotTypes.JOB: "dataset",
            ETCPlotTypes.USER: "aggregated User dataset",
            ETCPlotTypes.PI_GROUP: "aggregated PI Group dataset",
        }

        # Get data source and validate
        data = data_source_map[plot_type]
        if data is None:
            raise ValueError(
                f"Attribute {attribute_names[plot_type]} is not calculated, "
                "use calculate_all_efficiency_metrics() to calculate the dataframe before plotting."
            )

        # Set defaults if not provided
        if threshold_metric is None:
            threshold_metric = default_threshold_metrics[plot_type]
        if proportion_metric is None:
            proportion_metric = default_proportion_metrics[plot_type]

        plot_data, remain_percentage, total_raw_value, min_threshold = self._validate_and_filter_inputs(
            data,
            min_threshold,
            max_threshold,
            threshold_step,
            threshold_metric,
            proportion_metric,
            plot_type=plot_type,
        )

        # Clip threshold metrics to defined value
        self._clip_upper_threshold_metric(clip_threshold_metric, plot_data, threshold_metric)

        # Calculate thresholds
        thresholds_arr: np.ndarray = np.arange(min_threshold, max_threshold + threshold_step, threshold_step)
        fig, axe = plt.subplots(1, 1, figsize=(16, 7))
        axe_list = [axe]

        # Calculate proportion for each threshold
        proportions_data = self._etc_calculate_proportion(
            plot_data, thresholds_arr, proportion_metric, threshold_metric, plot_percentage, plot_type=plot_type
        )

        # Generate title if not provided
        if title is None:
            title = (
                f"ETC plot ({plot_type_labels[plot_type]}) for {'proportion' if plot_percentage else 'amounts'} of "
                f"{proportion_metric.value} by threshold {threshold_metric.value}"
            )
        title += (
            f"\n Total {proportion_metric.value}: {self._format_number_for_display(total_raw_value)} "
            f"(in {self._format_number_for_display(remain_percentage)}% of {dataset_descriptions[plot_type]})"
        )

        # Set up plot
        y_label = f"{'Percentage' if plot_percentage else 'Count'} of {proportion_metric.value} below threshold"
        axe.set_title(title)
        axe.set_ylabel(y_label)
        axe.set_xlabel(f"Threshold values ({threshold_metric.value})")
        axe.plot(thresholds_arr, proportions_data)
        axe.legend()
        y_max = proportions_data.max()
        y_min = proportions_data.min()
        y_range = y_max - y_min
        padding = max(y_range * 0.09, 1.0)  # 9% of range or minimum 1 unit
        axe.set_ylim(top=y_max + padding)

        self._generate_num_marker(axe, thresholds_arr, proportions_data, num_markers)

        return fig, axe_list

    def multiple_line_etc_plot(
        self,
        plot_object_list: list[str],
        object_column_type: Literal[ETCPlotTypes.USER, ETCPlotTypes.PI_GROUP],
        title: str | None = None,
        min_threshold: float = 0.0,
        max_threshold: float = 100.0,
        threshold_step: float = 1.0,
        threshold_metric: JobEfficiencyMetricsEnum = JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY,
        proportion_metric: Literal[
            ProportionMetricsEnum.JOB_HOURS, ProportionMetricsEnum.JOBS, ProportionMetricsEnum.VRAM_HOURS
        ] = ProportionMetricsEnum.JOBS,
        plot_percentage: bool = True,
        clip_threshold_metric: tuple[bool, float] = (False, 0.0),
    ) -> tuple[Figure, list[Axes]]:
        """
        Plot ETC curve for User/ Pi group given threshold_metrics.

        This function will plot an ETC curve for each object, with proportion metrics is count of jobs.

        Before plotting, this will filter out entries whose threshold_metric is NaN.

        In case where threshold_metric is ALLOC_VRAM_EFFICIENCY_SCORE:
            - Filter out entries where the threshold_metric is -inf.
            - If min_threshold is not provided or is 0, set min_threshold to the minimum value of the threshold_metric.

        If plot ```proportion_metric``` by percentage, percentage will be calculated based on the total data per object
        (user/ pi_group) after filtering out invalid data, not by the whole database.


        Args:
            plot_object_list (list[str]): List of users/pi_group that we want to plot.
            object_column_type (Literal): The type of objects to plot (USERS or PI_GROUPS).
            title (str or None): Title of the plot. Defaults to None.
            min_threshold (float): Minimum threshold value. Defaults to 0.0.
            max_threshold (float): Maximum threshold value. Defaults to 100.0.
            threshold_step (float): Step size for thresholds. Defaults to 1.0.
            threshold_metric (JobEfficiencyMetricsEnum): Metric used for thresholds. Defaults to ALLOC_VRAM_EFFICIENCY.
            proportion_metric (Literal): The proportion metric to use (JOB_HOURS, JOBS, or VRAM_HOURS).
                Defaults to JOBS.
            plot_percentage (bool): Whether to plot the proportion as a percentage or as raw counts. Defaults to True.
            clip_threshold_metric (tuple[bool, float]): A tuple where the first element is a boolean
                indicating whether to clip the threshold metrics, and the second element is the upper
                value to clip to. Defaults to (False, 0.0).

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
        plot_data, remain_percentage, total_raw_value, min_threshold = self._validate_and_filter_inputs(
            data,
            min_threshold,
            max_threshold,
            threshold_step,
            threshold_metric,
            proportion_metric,
        )

        self._clip_upper_threshold_metric(clip_threshold_metric, plot_data, threshold_metric)
        y_min, y_max = float("inf"), float("-inf")

        thresholds_arr: np.ndarray = np.arange(min_threshold, max_threshold + threshold_step, threshold_step)
        fig, axe = plt.subplots(1, 1, figsize=(16, 6))
        for target in plot_object_list:
            filtered = plot_data[plot_data[object_column_type.value] == target].copy()
            proportion_data = self._etc_calculate_proportion(
                filtered,
                thresholds_arr,
                proportion_metric,
                threshold_metric,
                plot_percentage,
            )
            y_min = min(y_min, proportion_data.min())
            y_max = max(y_max, proportion_data.max())
            axe.plot(thresholds_arr, proportion_data, label=f"{target}")

        if title is None:
            title = (
                f"Multple line ETC plot ({object_column_type.value}) for "
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
        y_range = y_max - y_min
        padding = max(y_range * 0.09, 1.0)  # 9% of range or minimum 1 unit
        axe.set_ylim(top=y_max + padding)

        return fig, [axe]
