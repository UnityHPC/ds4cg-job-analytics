from src.analysis.efficiency_analysis import EfficiencyAnalysis
from src.config.enum_constants import TimeUnitEnum
import pandas as pd


class FrequencyAnalysis(EfficiencyAnalysis):
    """
    A class for performing frequency analysis on job metrics.

    Inherits from EfficiencyAnalysis to reuse its methods for calculating metrics.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the FrequencyAnalyzer with a DataFrame of jobs.

        Args:
            df (pd.DataFrame): DataFrame containing job data.
        """
        super().__init__(df)

    def prepare_time_series_data(
        self,
        users: list[str],
        metric: str,
        time_unit: TimeUnitEnum | str,
        remove_zero_values: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare time series data for visualization.

        Args:
            users (list[str]): List of usernames.
            metric (str): The metric used to calculate efficiency (e.g., alloc_vram_efficiency or vram_hours).
            time_unit (TimeUnitEnum | str): Time unit for grouping. Can be either the enum or its string value.
            remove_zero_values (bool): Whether to remove zero values.

        Returns:
            pd.DataFrame: A DataFrame with grouped time series data, including additional fields for visualization.
        """
        data = self.jobs_df.copy()

        # Group jobs by the specified time unit
        data = self.group_jobs_by_time(data, time_unit)

        # data["all_time_groups"] = self.trim_zero_job_time_groups(
        #     data["TimeGroup"].unique().tolist(), data["JobID"].value_counts().to_dict(), remove_zero_values
        # )

        # Filter by users
        data = data[data["User"].isin(users)]

        # Aggregate data by time group and user
        grouped_data = (
            data.groupby(["TimeGroup", "User"], observed=True)
            .agg(
                Metric=(metric, "sum"),
                JobCount=("JobID", "count"),
                GPUHours=("job_hours", "sum"),
            )
            .reset_index()
        )

        # Add additional fields for visualization
        grouped_data["TimeGroup_Str"] = grouped_data["TimeGroup"].astype(str)
        grouped_data["TimeGroup_Datetime"] = grouped_data["TimeGroup"].dt.start_time

        # Remove zero values if required
        if remove_zero_values:
            grouped_data = grouped_data[grouped_data["Metric"] > 0]

        return grouped_data

    def group_jobs_by_time(self, data: pd.DataFrame, time_unit: TimeUnitEnum | str) -> pd.DataFrame:
        """
        Group jobs by a specified time unit (Months, Weeks, Days).

        Args:
            data (pd.DataFrame): Jobs DataFrame.
            time_unit (TimeUnitEnum | str): Time unit to group by. Can be either the enum or its string value.

        Returns:
            pd.DataFrame: Grouped jobs DataFrame.

        Raises:
            ValueError: If an invalid time unit is provided.
        """
        # Handle both enum and string inputs
        if isinstance(time_unit, TimeUnitEnum):
            time_unit_value = time_unit.value
        else:
            time_unit_value = time_unit

        if time_unit_value == TimeUnitEnum.MONTHS.value:
            data["TimeGroup"] = pd.to_datetime(data["StartTime"]).dt.to_period("M")
        elif time_unit_value == TimeUnitEnum.WEEKS.value:
            data["TimeGroup"] = pd.to_datetime(data["StartTime"]).dt.to_period("W")
        elif time_unit_value == TimeUnitEnum.DAYS.value:
            data["TimeGroup"] = pd.to_datetime(data["StartTime"]).dt.date
        else:
            raise ValueError(f"Invalid time unit {time_unit_value}. Choose 'Months', 'Weeks', or 'Days'.")

        return data

    def trim_zero_job_time_groups(
        self, all_time_groups: list[str], time_group_job_counts: dict[str, int], remove_zero_values: bool = True
    ) -> list[str]:
        """
        Helper method to trim leading and trailing time groups with zero jobs.

        Args:
            all_time_groups (list): Sorted list of time groups
            time_group_job_counts (dict): Dictionary mapping time groups to their job counts
            remove_zero_values (bool): Whether to trim zero values

        Returns:
            list: Trimmed list of time groups
        """
        if not remove_zero_values or not all_time_groups:
            return all_time_groups

        # Find first non-zero month
        first_non_zero_idx = 0
        while (
            first_non_zero_idx < len(all_time_groups)
            and time_group_job_counts[all_time_groups[first_non_zero_idx]] == 0
        ):
            first_non_zero_idx += 1

        # Find last non-zero month
        last_non_zero_idx = len(all_time_groups) - 1
        while last_non_zero_idx >= 0 and time_group_job_counts[all_time_groups[last_non_zero_idx]] == 0:
            last_non_zero_idx -= 1

        # If we found a valid range
        if first_non_zero_idx <= last_non_zero_idx:
            return all_time_groups[first_non_zero_idx : last_non_zero_idx + 1]
        # If there are no non-zero months, keep at least the first month
        elif len(all_time_groups) > 0:
            return [all_time_groups[0]]
        else:
            return []
