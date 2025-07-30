import datetime

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
        self, users: list[str], metric: str, time_unit: TimeUnitEnum, remove_zero_values: bool = True
    ) -> tuple[list[str], list[str], list[datetime.datetime], dict[str, pd.DataFrame]]:
        """
        Prepare time series data for visualization.

        Args:
            users (list[str]): List of usernames.
            metric (str): The metric used to calculate efficiency (e.g., alloc_vram_efficiency or vram_hours).
            time_unit (TimeUnitEnum): Time unit for grouping (e.g., 'Months', 'Weeks', 'Days').
            remove_zero_values (bool): Whether to remove zero values.

        Returns:
            tuple: (all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict)

        Raises:
            ValueError: If an invalid time unit is provided.
        """

        data = self.jobs_df.copy()
        # Group jobs by the specified time unit
        if time_unit == "Months":
            data["TimeGroup"] = pd.to_datetime(data["StartTime"]).dt.to_period("M")
        elif time_unit == "Weeks":
            data["TimeGroup"] = pd.to_datetime(data["StartTime"]).dt.to_period("W")
        elif time_unit == "Days":
            data["TimeGroup"] = pd.to_datetime(data["StartTime"]).dt.date
        else:
            raise ValueError(f"Invalid time unit {time_unit}. Choose 'Months', 'Weeks', or 'Days'.")

        # Prepare time series data logic
        user_time_groups = set()
        user_time_groups_map = {}
        for user in users:
            user_data = data[data["User"] == user]
            if remove_zero_values:
                user_data = user_data[user_data[metric] > 0]
            user_time_groups_map[user] = set(user_data["TimeGroup"].dropna().unique())
            user_time_groups.update(user_time_groups_map[user])

        # Ensure continuous timeline by filling in missing time periods
        all_time_groups = sorted(user_time_groups)

        # Format time groups as strings and datetime objects
        all_time_groups_str = [str(tg) for tg in all_time_groups]
        all_time_groups_datetime = [pd.to_datetime(str(tg)) for tg in all_time_groups]

        # Process each user's data
        user_dfs_dict = {}
        for user in users:
            user_data = data[data["User"] == user]
            if user_data.empty:
                user_dfs_dict[user] = pd.DataFrame()
                continue

            grouped_efficiency = []
            grouped_hours = []
            grouped_vram_hours = []
            grouped_job_counts = []

            for time_group in all_time_groups:
                group_data = user_data[user_data["TimeGroup"] == time_group]
                user_gpu_hours = group_data["job_hours"].sum() if not group_data.empty else 0
                user_vram_hours = group_data["vram_hours"].sum() if not group_data.empty else 0
                total_gpu_hours = data[data["TimeGroup"] == time_group]["job_hours"].sum()

                if total_gpu_hours > 0 and not group_data.empty:
                    efficiency = (group_data[metric] * user_gpu_hours / total_gpu_hours).mean()
                else:
                    efficiency = 0

                grouped_efficiency.append(efficiency)
                grouped_hours.append(user_gpu_hours)
                grouped_vram_hours.append(user_vram_hours)
                grouped_job_counts.append(group_data["JobID"].count() if not group_data.empty else 0)

            user_df = pd.DataFrame({
                "TimeGroup": all_time_groups,
                "TimeGroup_Str": all_time_groups_str,
                "TimeGroup_Datetime": all_time_groups_datetime,
                "Efficiency": grouped_efficiency,
                "GPU_Hours": grouped_hours,
                "VRAM_Hours": grouped_vram_hours,
                "Job_Count": grouped_job_counts,
            })

            user_dfs_dict[user] = user_df

        return all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict

    def filter_jobs_by_date_range(
        self,
        start_date: str | datetime.datetime | None = None,
        end_date: str | datetime.datetime | None = None,
        days_back: int | None = None,
    ) -> pd.DataFrame:
        """
        Filter jobs based on a specific date range or relative days back.

        Args:
            start_date (str, datetime.datetime): Start date in 'YYYY-MM-DD' format (optional).
            end_date (str, datetime.datetime): End date in 'YYYY-MM-DD' format (optional).
            days_back (int): Number of days back from today to filter jobs (optional).

        Returns:
            pd.DataFrame: Filtered jobs DataFrame.
        """
        data = self.jobs_df.copy()

        if days_back:
            start_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)

        if start_date:
            data = data[data["StartTime"] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data["StartTime"] <= pd.to_datetime(end_date)]

        return data

    def group_jobs_by_time(self, data: pd.DataFrame, time_unit: TimeUnitEnum) -> pd.DataFrame:
        """
        Group jobs by a specified time unit (Months, Weeks, Days).

        Args:
            data (pd.DataFrame): Jobs DataFrame.
            time_unit (TimeUnitEnum): Time unit to group by ('Months', 'Weeks', 'Days').

        Returns:
            pd.DataFrame: Grouped jobs DataFrame.

        Raises:
            ValueError: If an invalid time unit is provided.
        """
        if time_unit == TimeUnitEnum.MONTHS.value:
            data["TimeGroup"] = pd.to_datetime(data["StartTime"]).dt.to_period("M")
        elif time_unit == TimeUnitEnum.WEEKS.value:
            data["TimeGroup"] = pd.to_datetime(data["StartTime"]).dt.to_period("W")
        elif time_unit == TimeUnitEnum.DAYS.value:
            data["TimeGroup"] = pd.to_datetime(data["StartTime"]).dt.date
        else:
            raise ValueError(f"Invalid time unit {time_unit}. Choose 'Months', 'Weeks', or 'Days'.")

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
