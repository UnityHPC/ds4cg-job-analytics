from src.analysis.efficiency_analysis import EfficiencyAnalysis
from src.config.enum_constants import TimeUnitEnum
import pandas as pd


class FrequencyAnalyzer(EfficiencyAnalysis):
    """
    A class for performing frequency analysis on job metrics.
    Inherits from EfficiencyAnalysis to reuse its methods for calculating metrics.
    """

    def __init__(self, jobs_df: pd.DataFrame):
        """
        Initialize the FrequencyAnalyzer with a DataFrame of jobs.

        Args:
            jobs_df (pd.DataFrame): DataFrame containing job data.
        """
        super().__init__(jobs_df)

    def find_inefficient_users(self, alloc_vram_efficiency_filter: int | float | dict | None, min_jobs: int = 5):
        """
        Identify inefficient users based on allocated VRAM efficiency.

        Args:
            alloc_vram_efficiency_filter: Filter for allocated VRAM efficiency.
            min_jobs (int): Minimum number of jobs a user must have to be included.

        Returns:
            pd.DataFrame: DataFrame of inefficient users.
        """
        return self.find_inefficient_users_by_alloc_vram_efficiency(
            alloc_vram_efficiency_filter=alloc_vram_efficiency_filter, min_jobs=min_jobs
        )

    def calculate_users_with_metrics(self):
        """
        Calculate users with efficiency metrics.

        Returns:
            pd.DataFrame: DataFrame of users with efficiency metrics.
        """
        return self.calculate_user_efficiency_metrics()

    def calculate_jobs_with_metrics(self):
        """
        Calculate jobs with efficiency metrics.

        Returns:
            pd.DataFrame: DataFrame of jobs with efficiency metrics.
        """
        return self.calculate_job_efficiency_metrics(self.jobs_df)

    def prepare_time_series_data(self, users, metric, time_unit, remove_zero_values=True):
        """
        Prepare time series data for visualization.

        Args:
            users (list[str]): List of usernames.
            metric (str): The metric used to calculate efficiency (e.g., alloc_vram_efficiency or vram_hours).
            time_unit (str): Time unit for grouping (e.g., 'Months', 'Weeks', 'Days').
            remove_zero_values (bool): Whether to remove zero values.

        Returns:
            tuple: (all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict)
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

        # Prepare time series data logic (moved from TimeSeriesVisualizer)
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

            user_df = pd.DataFrame(
                {
                    "TimeGroup": all_time_groups,
                    "TimeGroup_Str": all_time_groups_str,
                    "TimeGroup_Datetime": all_time_groups_datetime,
                    "Efficiency": grouped_efficiency,
                    "GPU_Hours": grouped_hours,
                    "VRAM_Hours": grouped_vram_hours,
                    "Job_Count": grouped_job_counts,
                }
            )

            user_dfs_dict[user] = user_df

        return all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict

    def filter_jobs_by_date_range(self, start_date=None, end_date=None, days_back=None):
        """
        Filter jobs based on a specific date range or relative days back.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format (optional).
            end_date (str): End date in 'YYYY-MM-DD' format (optional).
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

    def group_jobs_by_time(self, data, time_unit):
        """
        Group jobs by a specified time unit (Months, Weeks, Days).

        Args:
            data (pd.DataFrame): Jobs DataFrame.
            time_unit (str): Time unit to group by ('Months', 'Weeks', 'Days').

        Returns:
            pd.DataFrame: Grouped jobs DataFrame.
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
