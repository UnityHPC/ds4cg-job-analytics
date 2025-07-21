"""
Functions to analyze efficiency of Jobs based on their VRAM usage.

The aim is to identify potential inefficiencies in GPU usage and notify users or PIs about these issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from collections.abc import Callable

from src.preprocess.preprocess import preprocess_data
from src.database import DatabaseConnection
from src.config.constants import DEFAULT_MIN_ELAPSED_SECONDS
from src.config.enum_constants import TimeUnitEnum


def load_jobs_dataframe_from_duckdb(db_path, table_name="Jobs", sample_size=None, random_state=None):
    """
    Connect to the DuckDB slurm_data_small.db and return the jobs table as a pandas DataFrame.

    Args:
        db_path (str or Path): Path to the DuckDB database.
        table_name (str, optional): Table name to query. Defaults to 'Jobs'.

    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """
    if isinstance(db_path, Path):
        db_path = db_path.resolve()
    db = DatabaseConnection(str(db_path))

    jobs_df = db.fetch_all_jobs(table_name=table_name)
    processed_data = preprocess_data(
        jobs_df, min_elapsed_seconds=0, include_failed_cancelled_jobs=False, include_cpu_only_jobs=False
    )
    if sample_size is not None:
        processed_data = processed_data.sample(n=sample_size, random_state=random_state)
    return processed_data


class EfficiencyAnalysis:
    """
    Class to encapsulate the efficiency analysis of jobs based on various metrics.

    It provides methods to load data, analyze workload efficiency, and evaluate CPU-GPU usage patterns.
    """

    def __init__(self, df=None, db_path=None, table_name="Jobs", sample_size=None, random_state=None):
        try:
            if df is not None and not df.empty:
                self.jobs_df = df.copy()
            else:
                self.jobs_df = load_jobs_dataframe_from_duckdb(db_path, table_name, sample_size, random_state)
            self.jobs_w_efficiency_metrics = None
            self.users_w_efficiency_metrics = None
            self.pi_accounts_w_efficiency_metrics = None
            self.analysis_results = None
        except Exception as e:
            raise ValueError(f"Failed to load jobs DataFrame: {e}") from e

    def _apply_numeric_filter(
        self, col: pd.Series, filter: int | list | set | tuple | dict | Callable, inclusive: bool | None
    ):
        """
        Helper to apply a numeric filter to a pandas Series.

        Args:
            col (pd.Series): The column to filter.
            value: The filter value (scalar, list/tuple/set, dict with 'min'/'max', or callable).
            inclusive (bool): Whether min/max are inclusive.

        Returns:
            pd.Series: Boolean mask.
        """
        mask = pd.Series([True] * len(col), index=col.index)
        if filter is not None:
            if callable(filter):
                mask &= col.apply(filter)
            elif isinstance(filter, list | set | tuple):
                mask &= col.isin(filter)
            elif isinstance(filter, dict):
                if "min" in filter:
                    mask &= col.ge(filter["min"]) if inclusive else col.gt(filter["min"])
                if "max" in filter:
                    mask &= col.le(filter["max"]) if inclusive else col.lt(filter["max"])
            else:
                mask &= col.eq(filter)
        return mask

    def filter_jobs_for_analysis(
        self,
        vram_constraint_filter: pd.Int64Dtype | list | set | tuple | dict | Callable | None = None,
        gpu_mem_usage_filter: int | list | set | tuple | dict | Callable | None = None,
        allocated_vram_filter: int | list | set | tuple | dict | Callable | None = None,
        gpu_count_filter: int | list | set | tuple | dict | Callable | None = None,
        elapsed_seconds_min=DEFAULT_MIN_ELAPSED_SECONDS,
    ):
        """
        Filter jobs based on VRAM constraints, GPU allocation, and usage criteria.

        Args:
            vram_constraint_filter:
                - None: no filtering on vram_constraint
                - int or float: select rows where vram_constraint == value
                - list/set/tuple: select rows where vram_constraint is in the list
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
                - pd.NA or <NA>: select rows where vram_constraint is Nullable Int64 (i.e., pd.NA)
                - callable: custom filter function
            gpu_mem_usage_filter:
                - None: no filtering on GPU count
                - int: select rows where GPUs == value
                - list/set/tuple: select rows where GPUs is in the list
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
                - callable: custom filter function
            allocated_vram_filter:
                - Same as above; if dict, must include 'inclusive' (bool)
            gpu_count_filter:
                - Same as above; if dict, must include 'inclusive' (bool)
            elapsed_seconds_min (int): Minimum elapsed time in seconds

        Returns:
            DataFrame: Filtered jobs DataFrame based on the specified criteria.
        """

        mask = pd.Series([True] * len(self.jobs_df), index=self.jobs_df.index)

        # Helper to extract 'inclusive' from dict filter, must be present if dict
        def get_inclusive(filter_val):
            if isinstance(filter_val, dict):
                if "inclusive" not in filter_val or not isinstance(filter_val["inclusive"], bool):
                    raise ValueError("If a filter is a dict, it must include an 'inclusive' boolean key.")
                return filter_val["inclusive"]
            return None

        # vram_constraint
        if vram_constraint_filter is not None:
            col = self.jobs_df["vram_constraint"]
            if callable(vram_constraint_filter):
                mask &= col.apply(vram_constraint_filter)
            elif isinstance(vram_constraint_filter, list | set | tuple):
                mask &= col.isin(vram_constraint_filter)
            elif isinstance(vram_constraint_filter, dict):
                inclusive = get_inclusive(vram_constraint_filter)
                if "min" in vram_constraint_filter:
                    mask &= (
                        col.ge(vram_constraint_filter["min"]) if inclusive else col.gt(vram_constraint_filter["min"])
                    )
                if "max" in vram_constraint_filter:
                    mask &= (
                        col.le(vram_constraint_filter["max"]) if inclusive else col.lt(vram_constraint_filter["max"])
                    )
            elif vram_constraint_filter is pd.NA or (
                isinstance(vram_constraint_filter, float) and np.isnan(vram_constraint_filter)
            ):
                mask &= col.isna()
            else:
                mask &= col.eq(vram_constraint_filter)

        # GPU memory usage filter
        if gpu_mem_usage_filter is not None:
            if isinstance(gpu_mem_usage_filter, dict):
                gpu_mem_usage_inclusive = get_inclusive(gpu_mem_usage_filter)
            else:
                gpu_mem_usage_inclusive = None
            mask &= self._apply_numeric_filter(
                self.jobs_df["GPUMemUsage"], gpu_mem_usage_filter, gpu_mem_usage_inclusive
            )

        # Allocated VRAM filter
        if allocated_vram_filter is not None:
            if isinstance(allocated_vram_filter, dict):
                allocated_vram_inclusive = get_inclusive(allocated_vram_filter)
            else:
                allocated_vram_inclusive = None
            mask &= self._apply_numeric_filter(
                self.jobs_df["allocated_vram"], allocated_vram_filter, allocated_vram_inclusive
            )

        # GPU count filter
        if gpu_count_filter is not None:
            if isinstance(gpu_count_filter, dict):
                gpu_count_inclusive = get_inclusive(gpu_count_filter)
            else:
                gpu_count_inclusive = None
            mask &= self._apply_numeric_filter(self.jobs_df["GPUs"], gpu_count_filter, gpu_count_inclusive)

        return self.jobs_df[mask & (self.jobs_df["Elapsed"].dt.total_seconds() >= elapsed_seconds_min)].copy()

    def calculate_job_efficiency_metrics(
        self,
        filtered_jobs: pd.DataFrame,
    ):
        """
        Calculate jobs efficiency metrics for the filtered jobs DataFrame.

        Refer to the documentation for the definition of the metrics calculated.

        Args:
            filtered_jobs (DataFrame): DataFrame containing jobs to analyze.

        Returns:
            DataFrame: Jobs with efficiency metrics added
        """

        # rename GPUs to gpu_count for clarity
        filtered_jobs = filtered_jobs.rename(columns={"GPUs": "gpu_count"})

        # Calculate job efficiency metrics
        filtered_jobs.loc[:, "job_hours"] = (
            filtered_jobs["Elapsed"].dt.total_seconds() * filtered_jobs["gpu_count"] / 3600
        )
        filtered_jobs.loc[:, "used_vram_gib"] = filtered_jobs["GPUMemUsage"] / (2**30)
        filtered_jobs.loc[:, "alloc_vram_efficiency"] = (
            filtered_jobs["used_vram_gib"] / filtered_jobs["allocated_vram"]
        )
        # TODO (Arda): Clip alloc_vram_efficiency to 1

        # Compute vram_constraint_efficiency, a nullable float. Set to NA if vram_constraint is NA
        filtered_jobs.loc[:, "vram_constraint_efficiency"] = (
            filtered_jobs["used_vram_gib"] / filtered_jobs["vram_constraint"]
        )
        # TODO (Arda): Decide if it should clip vram_constraint_efficiency to 1

        # Calculate job allocated VRAM efficiency score
        # This is a log-transformed score that penalizes low efficiency and longer job_hours
        # TODO (Arda): Decide implementation of alloc_vram_efficiency_score
        filtered_jobs["alloc_vram_efficiency_score"] = (
            np.log(filtered_jobs["alloc_vram_efficiency"]) * filtered_jobs["job_hours"]
        )

        # Calculate weighted vram efficiency per job, normalized by total job_hours for that specific PI
        pi_gpu_hours = filtered_jobs.groupby("Account", observed=True)["job_hours"].transform("sum")
        filtered_jobs.loc[:, "pi_weighted_vram_efficiency"] = (
            filtered_jobs["alloc_vram_efficiency"] * filtered_jobs["job_hours"]
        ) / pi_gpu_hours

        # Add CPU memory metrics if available
        if "CPUMemUsage" in self.jobs_df.columns:
            filtered_jobs.loc[:, "used_cpu_gib"] = filtered_jobs["CPUMemUsage"] / (2**30)

        self.jobs_w_efficiency_metrics = filtered_jobs
        return self.jobs_w_efficiency_metrics

    def calculate_user_efficiency_metrics(self):
        """
        Calculate user efficiency metrics based on job efficiency data.

        Returns:
            pd.DataFrame: DataFrame with users and their average VRAM efficiency
        """
        if self.jobs_w_efficiency_metrics is None:
            raise ValueError(
                "Jobs DataFrame with efficiency metrics is not available. "
                "Please run calculate_job_efficiency_metrics first."
            )

        # Compute user_job_hours_per_job once and reuse for both metrics
        user_job_hours_per_job = self.jobs_w_efficiency_metrics.groupby("User", observed=True)["job_hours"].transform(
            "sum"
        )

        users_w_efficiency_metrics = (
            self.jobs_w_efficiency_metrics.groupby("User", observed=False)
            .agg(
                job_count=("JobID", "count"),
                user_job_hours=("job_hours", "sum"),
                pi_account=("Account", "first"),
            )
            .reset_index()
        )

        self.jobs_w_efficiency_metrics.loc[:, "weighted_alloc_vram_efficiency"] = (
            self.jobs_w_efficiency_metrics["alloc_vram_efficiency"]
            * self.jobs_w_efficiency_metrics["job_hours"]
            / user_job_hours_per_job
        )
        users_w_efficiency_metrics.loc[:, "expected_value_alloc_vram_efficiency"] = (
            self.jobs_w_efficiency_metrics.groupby("User", observed=True)["weighted_alloc_vram_efficiency"]
            .sum()
            .to_numpy()
        )

        self.jobs_w_efficiency_metrics.loc[:, "weighted_gpu_count"] = (
            self.jobs_w_efficiency_metrics["gpu_count"]
            * self.jobs_w_efficiency_metrics["job_hours"]
            / user_job_hours_per_job
        )
        users_w_efficiency_metrics.loc[:, "expected_value_gpu_count"] = (
            self.jobs_w_efficiency_metrics.groupby("User", observed=True)["weighted_gpu_count"].sum().to_numpy()
        )

        # Calculate metric representing the total amount of GPU memory resources a user has been allocated over time.
        # It answers the question: “How much VRAM, and for how long, did this user occupy?”
        users_w_efficiency_metrics.loc[:, "vram_hours"] = (
            self.jobs_w_efficiency_metrics.groupby("User", observed=True)
            .apply(lambda df: (df["allocated_vram"] * df["job_hours"]).sum())
            .to_numpy()
        )

        self.jobs_w_efficiency_metrics = self.jobs_w_efficiency_metrics.drop(
            columns=["weighted_alloc_vram_efficiency", "weighted_gpu_count"]
        )

        self.users_w_efficiency_metrics = users_w_efficiency_metrics
        return self.users_w_efficiency_metrics

    def evaluate_cpu_gpu_usage(self, hours_percentage_threshold=25, vram_efficiency_threshold=0.3):
        """
        Evaluates the efficiency of GPU jobs based on VRAM usage and CPU-GPU balance.

        Args:
            hours_percentage_threshold (float): Threshold for high waste GPU hours as a percentage of total GPU hours
            vram_efficiency_threshold (float): Threshold for VRAM efficiency to consider a job as high waste

        Returns:
            dict: Analysis results with balance patterns and recommendations
        """
        # TODO: Needs refactoring to parametrize the analysis thresholds and make it more flexible
        # TODO: Need to separate the analysis into different methods for clarity
        # TODO: Separate the CPU-GPU balance analysis from the VRAM efficiency analysis

        analysis = {}

        # Ensure jobs_w_efficiency_metrics is available
        if self.jobs_w_efficiency_metrics is None:
            raise ValueError(
                "jobs_w_efficiency_metrics DataFrame is not available. Please run analyze_workload_efficiency first."
            )

        # Efficiency metrics
        self.jobs_w_efficiency_metrics["efficiency_category"] = pd.cut(
            self.jobs_w_efficiency_metrics["alloc_vram_efficiency"],
            bins=[0, 0.1, 0.3, 0.6, 1.0],
            labels=["Very Low (<10%)", "Low (10-30%)", "Medium (30-60%)", "High (60-100%)"],
        )

        # Duration categories
        self.jobs_w_efficiency_metrics["duration_category"] = pd.cut(
            self.jobs_w_efficiency_metrics["job_hours"],
            bins=[0, 1, 6, 24, float("inf")],
            labels=["Short (<1h)", "Medium (1-6h)", "Long (6-24h)", "Very Long (>24h)"],
        )

        # Overall statistics
        analysis["total_jobs"] = len(self.jobs_w_efficiency_metrics)
        analysis["total_gpu_hours"] = self.jobs_w_efficiency_metrics["job_hours"].sum()
        analysis["avg_efficiency"] = self.jobs_w_efficiency_metrics["alloc_vram_efficiency"].mean()
        analysis["median_efficiency"] = self.jobs_w_efficiency_metrics["alloc_vram_efficiency"].median()

        # Efficiency distribution analysis
        efficiency_analysis = (
            self.jobs_w_efficiency_metrics.groupby("efficiency_category", observed=False)
            .agg(
                {
                    "JobID": "count",
                    "job_hours": "sum",
                    "alloc_vram_efficiency": "mean",
                    "allocated_vram": "mean",
                    "used_vram_gib": "mean",
                }
            )
            .round(3)
        )

        efficiency_analysis.columns = [
            "Job_Count",
            "job_hours",
            "Avg_Efficiency",
            "Avg_Allocated_GB",
            "Avg_Used_GB",
        ]
        efficiency_analysis["Share of total GPU Hours"] = (
            efficiency_analysis["job_hours"] / analysis["total_gpu_hours"] * 100
        ).round(1)
        analysis["efficiency_patterns"] = efficiency_analysis

        # CPU-GPU balance analysis (if CPU data available)
        if "cpu_gpu_ratio" in self.jobs_w_efficiency_metrics.columns:
            # Categorize workloads by CPU-GPU balance
            self.jobs_w_efficiency_metrics["workload_type"] = pd.cut(
                self.jobs_w_efficiency_metrics["cpu_gpu_ratio"],
                bins=[0, 1, 5, 20, float("inf")],
                labels=[
                    "GPU-intensive (CPU<GPU)",
                    "Balanced (CPU≈GPU)",
                    "CPU-heavy (CPU>GPU)",
                    "Very CPU-heavy (CPU>>GPU)",
                ],
            )

            balance_analysis = self.jobs_w_efficiency_metrics.groupby("workload_type", observed=False).agg(
                {"JobID": "count", "job_hours": "sum", "alloc_vram_efficiency": "mean", "cpu_gpu_ratio": "mean"}
            )
            analysis["cpu_gpu_balance"] = balance_analysis

        # Over-allocation analysis
        high_waste_jobs = self.jobs_w_efficiency_metrics[
            self.jobs_w_efficiency_metrics["alloc_vram_efficiency"] <= vram_efficiency_threshold
        ]
        analysis["high_waste_jobs"] = len(high_waste_jobs)
        analysis["high_waste_gpu_hours"] = high_waste_jobs["job_hours"].sum()
        analysis["high_waste_hours_share"] = analysis["high_waste_gpu_hours"] / analysis["total_gpu_hours"] * 100

        # Duration vs efficiency correlation
        duration_efficiency = self.jobs_w_efficiency_metrics.groupby("duration_category", observed=False).agg(
            {"JobID": "count", "alloc_vram_efficiency": "mean", "job_hours": "sum"}
        )
        analysis["duration_efficiency_patterns"] = duration_efficiency

        # Generate recommendations
        analysis_report = []

        low_efficiency_hours = efficiency_analysis.loc[
            efficiency_analysis.index.isin(["Very Low (<10%)", "Low (10-30%)"]), "job_hours"
        ].sum()
        low_efficiency_percentage = low_efficiency_hours / analysis["total_gpu_hours"] * 100

        if low_efficiency_percentage > 50:
            analysis_report.append("CRITICAL: >50% of GPU hours have <30% efficiency - immediate optimization needed")
        elif low_efficiency_percentage > 30:
            analysis_report.append("HIGH PRIORITY: Significant inefficiency detected - user education campaign needed")

        if analysis["high_waste_hours_share"] > hours_percentage_threshold:
            analysis_report.append(
                f"MAJOR OVER-ALLOCATION: >{hours_percentage_threshold}% of total GPU hours has been wasted "
                f"with jobs with less than {vram_efficiency_threshold * 100}% efficiency."
            )

        analysis["report"] = analysis_report

        self.analysis_results = analysis

        return self.analysis_results

    def find_inefficient_users_by_alloc_vram_efficiency(self, efficiency_threshold=0.3, min_jobs=5):
        """
        Identify users with low expected allocated VRAM efficiency across their jobs compared to others

        Args:
            efficiency_threshold (float): Maximum threshold for expected allocated VRAM efficiency
            min_jobs (int): Minimum number of jobs a user must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with users and their average VRAM efficiency
        """
        if self.users_w_efficiency_metrics is None:
            raise ValueError(
                "Users with efficiency metrics DataFrame is not available. "
                "Please run calculate_user_efficiency_metrics first."
            )

        mask = pd.Series([True] * len(self.users_w_efficiency_metrics), index=self.users_w_efficiency_metrics.index)

        col = self.users_w_efficiency_metrics["expected_value_alloc_vram_efficiency"]
        mask &= col.le(efficiency_threshold)

        col = self.users_w_efficiency_metrics["job_count"]
        mask &= col.ge(min_jobs)

        inefficient_users = self.users_w_efficiency_metrics[mask]

        # Sort by the metric ascending (lower is worse)
        inefficient_users = inefficient_users.sort_values("expected_value_alloc_vram_efficiency", ascending=True)
        return inefficient_users

    def find_inefficient_users_by_vram_hours(self, vram_hours_threshold=200, min_jobs=5):
        """
        Identify users with high VRAM-hours across their jobs compared to others.

        Args:
            vram_hours_threshold (float): Minimum threshold for VRAM hours to consider a user as inefficient
            min_jobs (int): Minimum number of jobs a user must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with users and their total VRAM hours
        """
        if self.users_w_efficiency_metrics is None:
            raise ValueError(
                "Users with efficiency metrics DataFrame is not available. "
                "Please run calculate_user_efficiency_metrics first."
            )

        mask = pd.Series([True] * len(self.users_w_efficiency_metrics), index=self.users_w_efficiency_metrics.index)

        col = self.users_w_efficiency_metrics["vram_hours"]
        mask &= col.ge(vram_hours_threshold)

        col = self.users_w_efficiency_metrics["job_count"]
        mask &= col.ge(min_jobs)

        inefficient_users = self.users_w_efficiency_metrics[mask]

        # Sort by the metric descending (higher is worse)
        inefficient_users = inefficient_users.sort_values("vram_hours", ascending=False)
        return inefficient_users

    # TODO (Arda): The individual user's metrics is expected_value_alloc_vram_efficiency
    def calculate_pi_account_efficiency_metrics(self):
        """
        Calculate PI account efficiency metrics based on user efficiency data.

        For a group of users, we calculate the expected value of user metrics for the group of users
        The weights for the expected value are the vram_hours of each user in the group

        Returns:
            pd.DataFrame: DataFrame with PI accounts and their efficiency metrics
        """
        if self.users_w_efficiency_metrics is None:
            raise ValueError(
                "Users with efficiency metrics DataFrame is not available. "
                "Please run calculate_user_efficiency_metrics first."
            )

        pi_efficiency_metrics = (
            self.users_w_efficiency_metrics.groupby("pi_account", observed=True)
            .agg(
                job_count=("job_count", "sum"),
                pi_acc_job_hours=("user_job_hours", "sum"),
                user_count=("User", "nunique"),
                pi_acc_vram_hours=("vram_hours", "sum"),
            )
            .reset_index()
        )

        # Compute pi_acc_vram_hours once and reuse for both metrics
        pi_acc_vram_hours = self.users_w_efficiency_metrics.groupby("pi_account", observed=True)[
            "vram_hours"
        ].transform("sum")

        self.users_w_efficiency_metrics.loc[:, "weighted_ev_alloc_vram_efficiency"] = (
            self.users_w_efficiency_metrics["expected_value_alloc_vram_efficiency"]
            * self.users_w_efficiency_metrics["vram_hours"]
            / pi_acc_vram_hours
        )

        pi_efficiency_metrics.loc[:, "expected_value_alloc_vram_efficiency"] = (
            self.users_w_efficiency_metrics.groupby("pi_account", observed=True)["weighted_ev_alloc_vram_efficiency"]
            .sum()
            .to_numpy()
        )

        self.users_w_efficiency_metrics.loc[:, "weighted_ev_gpu_count"] = (
            self.users_w_efficiency_metrics["expected_value_gpu_count"]
            * self.users_w_efficiency_metrics["vram_hours"]
            / pi_acc_vram_hours
        )
        pi_efficiency_metrics.loc[:, "expected_value_gpu_count"] = (
            self.users_w_efficiency_metrics.groupby("pi_account", observed=True)["weighted_ev_gpu_count"]
            .sum()
            .to_numpy()
        )

        self.users_w_efficiency_metrics = self.users_w_efficiency_metrics.drop(
            columns=["weighted_ev_alloc_vram_efficiency"]
        )

        self.pi_accounts_w_efficiency_metrics = pi_efficiency_metrics
        return self.pi_accounts_w_efficiency_metrics

    def find_inefficient_pis_weighted_by_hours(self, efficiency_threshold=0.3, min_jobs=5):
        """
        Identify PIs with low average VRAM efficiency across their jobs, weighted by the hours they were inefficient.

        Args:
            efficiency_threshold (float): Threshold for VRAM efficiency to consider a PI as inefficient
            min_jobs (int): Minimum number of jobs a PI must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with PIs and their average VRAM efficiency
        """
        if self.efficiency_df is None:
            raise ValueError("Efficiency DataFrame is not available. Please run calculate_efficiency_metrics first.")

        inefficient_pis = (
            self.efficiency_df[self.efficiency_df["alloc_vram_efficiency"] < efficiency_threshold]
            .groupby("Account", observed=False)
            .agg(
                Job_Count=("JobID", "count"),
                Avg_Allocated_VRAM=("allocated_vram", "mean"),
                pi_group_gpu_hours=("job_hours", "sum"),
                Avg_GPUs=("GPUs", "mean"),
                Avg_Weighted_VRAM_Efficiency=("pi_weighted_vram_efficiency", "mean"),
            )
            .reset_index()
        )

        # Multiply share of total gpu hours by weighted vram efficiency to get the new metric
        inefficient_pis["Weighted_Efficiency_Contribution"] = (
            inefficient_pis["pi_group_gpu_hours"]
            * inefficient_pis["Avg_Weighted_VRAM_Efficiency"]
            / inefficient_pis["pi_group_gpu_hours"].sum()
        )

        # Only include PIs with at least 5 jobs
        inefficient_pis = inefficient_pis[inefficient_pis["Job_Count"] >= min_jobs]

        # Sort by the new metric ascending (lower is worse)
        inefficient_pis = inefficient_pis.sort_values("Weighted_Efficiency_Contribution", ascending=True)
        return

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
        data = self.jobs_w_efficiency_metrics.copy()

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
            raise ValueError("Invalid time unit. Choose 'Months', 'Weeks', or 'Days'.")

        return data

    def _prepare_time_series_data(self, data, users, time_unit, remove_zero_values=True):
        """
        Helper function to prepare time series data for both interactive and non-interactive plots.

        Args:
            data (pd.DataFrame): Filtered and grouped jobs data
            users (list[str]): List of user names
            time_unit (str): Time unit for grouping
            remove_zero_values (bool): Whether to remove zero values

        Returns:
            tuple: (all_time_groups, all_time_groups_str, user_dfs_dict)
        """
        # Determine all time groups to show on x-axis: union of all non-empty periods for selected users
        user_time_groups = set()
        user_time_groups_map = {}
        for user in users:
            user_data = data[data["User"] == user]
            if remove_zero_values:
                user_data = user_data[user_data["alloc_vram_efficiency"] > 0]
            user_time_groups_map[user] = set(user_data["TimeGroup"].dropna().unique())
            user_time_groups.update(user_time_groups_map[user])

        # Ensure continuous timeline by filling in missing time periods
        if user_time_groups:
            if time_unit == TimeUnitEnum.MONTHS.value:
                # Get min and max periods
                min_period = min(user_time_groups)
                max_period = max(user_time_groups)

                # Create a continuous range of all months between min and max
                all_time_groups = []
                current_period = min_period
                while current_period <= max_period:
                    all_time_groups.append(current_period)
                    # Move to next month
                    if current_period.month == 12:
                        current_period = pd.Period(f"{current_period.year + 1}-01", freq="M")
                    else:
                        current_period = pd.Period(f"{current_period.year}-{current_period.month + 1:02d}", freq="M")
            elif time_unit == TimeUnitEnum.WEEKS.value:
                # Get min and max periods
                min_period = min(user_time_groups)
                max_period = max(user_time_groups)

                # Create a continuous range of all weeks between min and max
                all_time_groups = []
                current_period = min_period
                while current_period <= max_period:
                    all_time_groups.append(current_period)
                    # Move to next week (add 7 days)
                    next_start = pd.to_datetime(str(current_period).split("/")[0]) + pd.Timedelta(days=7)
                    current_period = pd.Period(next_start, freq="W")
            else:
                # For other time units or empty data, just use sorted unique time groups
                all_time_groups = sorted(user_time_groups)
        else:
            # If no time groups found, use empty list
            all_time_groups = []

        # Create a dictionary to track job counts for each time group across all users
        time_group_job_counts = {tg: 0 for tg in all_time_groups}

        # First pass: calculate job counts for all time groups
        for user in users:
            user_data = data[data["User"] == user]
            if user_data.empty:
                continue
            for time_group in all_time_groups:
                group_data = user_data[user_data["TimeGroup"] == time_group]
                job_count = group_data["JobID"].count() if not group_data.empty else 0
                time_group_job_counts[time_group] += job_count

        # Trim leading and trailing zero-job time groups while maintaining continuity in the middle
        all_time_groups = self._trim_zero_job_time_groups(all_time_groups, time_group_job_counts, remove_zero_values)

        # Format time groups as strings appropriately based on time unit
        # Also create datetime objects for proper chronological ordering in interactive plots
        all_time_groups_str = []
        all_time_groups_datetime = []

        if time_unit == TimeUnitEnum.WEEKS.value:
            # For weeks, create a more readable format like "Week of Jun 2, 2025"
            for tg in all_time_groups:
                # Extract the start date from the period (format is like '2025-06-02/2025-06-08')
                week_start = pd.to_datetime(str(tg).split("/")[0])
                all_time_groups_str.append(f"Week of {week_start.strftime('%b %d, %Y')}")
                all_time_groups_datetime.append(week_start)
        elif time_unit == TimeUnitEnum.MONTHS.value:
            for tg in all_time_groups:
                # For months, use the first day of the month as datetime
                month_start = pd.to_datetime(str(tg) + "-01")
                all_time_groups_str.append(str(tg))
                all_time_groups_datetime.append(month_start)
        else:
            for tg in all_time_groups:
                # For other time units, try to convert to datetime
                try:
                    dt = pd.to_datetime(str(tg))
                    all_time_groups_str.append(str(tg))
                    all_time_groups_datetime.append(dt)
                except Exception:
                    # Fallback if conversion fails
                    all_time_groups_str.append(str(tg))
                    all_time_groups_datetime.append(None)

        # Process each user's data
        user_dfs_dict = {}
        for user in users:
            user_data = data[data["User"] == user]
            if user_data.empty:
                user_dfs_dict[user] = pd.DataFrame()
                continue

            grouped_efficiency = []
            grouped_hours = []
            grouped_job_counts = []

            for time_group in all_time_groups:
                group_data = user_data[user_data["TimeGroup"] == time_group]
                user_gpu_hours = group_data["job_hours"].sum() if not group_data.empty else 0
                total_gpu_hours = data[data["TimeGroup"] == time_group]["job_hours"].sum()

                if total_gpu_hours > 0 and not group_data.empty:
                    efficiency = (group_data["alloc_vram_efficiency"] * user_gpu_hours / total_gpu_hours).mean()
                else:
                    efficiency = 0

                grouped_efficiency.append(efficiency)
                grouped_hours.append(user_gpu_hours)
                grouped_job_counts.append(group_data["JobID"].count() if not group_data.empty else 0)

            user_df = pd.DataFrame(
                {
                    "TimeGroup": all_time_groups,
                    "TimeGroup_Str": all_time_groups_str,
                    "TimeGroup_Datetime": all_time_groups_datetime,
                    "Efficiency": grouped_efficiency,
                    "GPU_Hours": grouped_hours,
                    "VRAM_Hours": grouped_hours,  # Same as GPU_Hours for consistency
                    "Job_Count": grouped_job_counts,
                }
            )

            user_dfs_dict[user] = user_df

        return all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict

    def _trim_zero_job_time_groups(self, all_time_groups, time_group_job_counts, remove_zero_values):
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

    def plot_vram_efficiency(
        self,
        users: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        days_back: int | None = None,
        time_unit: str | TimeUnitEnum = TimeUnitEnum.MONTHS.value,
        remove_zero_values: bool = True,
        max_points: int = 100,
        annotation_style: str = "hover",  # "hover", "combined", "table", "none"
        show_secondary_y: bool = False,  # Show job counts on secondary y-axis
        exclude_fields: list[str] | None = None,  # List of fields to exclude from annotation text box
    ) -> pd.DataFrame:
        """
        Plot VRAM efficiency over time for specific users with improved annotation options.

        Args:
            users (list[str]): List of user names to plot.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            days_back (int, optional): Number of days back from today to filter jobs.
            time_unit (str or TimeUnitEnum, optional): Time unit to group by ('Month', 'Week', 'Day').
            remove_zero_values (bool, optional): Whether to remove users with zero efficiency values
                from the plot.
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            annotation_style (str, optional): Style for annotations ("hover", "combined", "table", "none").
            show_secondary_y (bool, optional): Whether to show job counts on secondary y-axis.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.

        Returns:
            pd.DataFrame: DataFrame containing the metrics used for the table and annotations in the plot.
        """

        # Filter data by date range or days back
        data = self.filter_jobs_by_date_range(start_date=start_date, end_date=end_date, days_back=days_back)

        # Group data by the specified time unit
        data = self.group_jobs_by_time(data, time_unit)

        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Prepare secondary axis if needed
        if show_secondary_y:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Job Count", color="tab:gray")
            ax2.tick_params(axis="y", labelcolor="tab:gray")

        # Store annotation data for table display
        annotation_data = []
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(users))]

        if exclude_fields is None:
            exclude_fields = []

        # Determine all time groups to show on x-axis: union of all non-empty periods for selected users
        user_time_groups = set()
        user_time_groups_map = {}
        for user in users:
            user_data = data[data["User"] == user]
            if remove_zero_values:
                user_data = user_data[user_data["alloc_vram_efficiency"] > 0]
            user_time_groups_map[user] = set(user_data["TimeGroup"].dropna().unique())
            user_time_groups.update(user_time_groups_map[user])

        # Ensure continuous timeline by filling in missing time periods
        if user_time_groups:
            if time_unit == TimeUnitEnum.MONTHS.value:
                # Get min and max periods
                min_period = min(user_time_groups)
                max_period = max(user_time_groups)

                # Create a continuous range of all months between min and max
                all_time_groups = []
                current_period = min_period
                while current_period <= max_period:
                    all_time_groups.append(current_period)
                    # Move to next month
                    if current_period.month == 12:
                        current_period = pd.Period(f"{current_period.year + 1}-01", freq="M")
                    else:
                        current_period = pd.Period(f"{current_period.year}-{current_period.month + 1:02d}", freq="M")
            elif time_unit == TimeUnitEnum.WEEKS.value:
                # Get min and max periods
                min_period = min(user_time_groups)
                max_period = max(user_time_groups)

                # Create a continuous range of all weeks between min and max
                all_time_groups = []
                current_period = min_period
                while current_period <= max_period:
                    all_time_groups.append(current_period)
                    # Move to next week (add 7 days)
                    next_start = pd.to_datetime(str(current_period).split("/")[0]) + pd.Timedelta(days=7)
                    current_period = pd.Period(next_start, freq="W")
            else:
                # For other time units or empty data, just use sorted unique time groups
                all_time_groups = sorted(user_time_groups)
        else:
            # If no time groups found, use empty list
            all_time_groups = []

        # Create a dictionary to track job counts for each time group across all users
        time_group_job_counts = {tg: 0 for tg in all_time_groups}

        # First pass: calculate job counts for all time groups
        for user in users:
            user_data = data[data["User"] == user]
            if user_data.empty:
                continue
            for time_group in all_time_groups:
                group_data = user_data[user_data["TimeGroup"] == time_group]
                job_count = group_data["JobID"].count() if not group_data.empty else 0
                time_group_job_counts[time_group] += job_count

        # Trim leading and trailing zero-job time groups while maintaining continuity in the middle
        all_time_groups = self._trim_zero_job_time_groups(all_time_groups, time_group_job_counts, remove_zero_values)

        # Format time groups as strings appropriately based on time unit
        if time_unit == TimeUnitEnum.WEEKS.value:
            # For weeks, create a more readable format like "Week of Jun 2, 2025"
            all_time_groups_str = []
            for tg in all_time_groups:
                # Extract the start date from the period (format is like '2025-06-02/2025-06-08')
                week_start = pd.to_datetime(str(tg).split("/")[0])
                all_time_groups_str.append(f"Week of {week_start.strftime('%b %d, %Y')}")
        else:
            all_time_groups_str = [str(tg) for tg in all_time_groups]

        user_dfs = []
        any_nonzero_efficiency = False
        for user in users:
            user_data = data[data["User"] == user]
            if user_data.empty:
                user_dfs.append(pd.DataFrame())
                continue
            grouped_efficiency = []
            grouped_hours = []
            grouped_job_counts = []
            for time_group in all_time_groups:
                group_data = user_data[user_data["TimeGroup"] == time_group]
                user_gpu_hours = group_data["job_hours"].sum() if not group_data.empty else 0
                total_gpu_hours = data[data["TimeGroup"] == time_group]["job_hours"].sum()
                if total_gpu_hours > 0 and not group_data.empty:
                    efficiency = (group_data["alloc_vram_efficiency"] * user_gpu_hours / total_gpu_hours).mean()
                else:
                    efficiency = 0
                grouped_efficiency.append(efficiency)
                grouped_hours.append(user_gpu_hours)
                grouped_job_counts.append(group_data["JobID"].count() if not group_data.empty else 0)
            user_df = pd.DataFrame(
                {
                    "TimeGroup": all_time_groups,
                    "TimeGroup_Str": all_time_groups_str,
                    "Efficiency": grouped_efficiency,
                    "GPU_Hours": grouped_hours,
                    "Job_Count": grouped_job_counts,
                }
            )

            if not user_df.empty:
                any_nonzero_efficiency = True
            user_dfs.append(user_df)

        # If all users have zero efficiency, plot VRAM Hours instead
        if not any_nonzero_efficiency:
            print("All users have zero efficiency. Plotting VRAM Hours instead.")
            return self.plot_vram_hours(
                users=users,
                start_date=start_date,
                end_date=end_date,
                days_back=days_back,
                time_unit=time_unit,
                remove_zero_values=remove_zero_values,
                max_points=max_points,
                show_secondary_y=show_secondary_y,
                exclude_fields=exclude_fields,
                annotation_style=annotation_style,
            )

        for idx, user_df in enumerate(user_dfs):
            if user_df.empty:
                continue
            user = users[idx]

            # Create mapping of time groups to their position in the x-axis
            time_group_to_index = {tg: i for i, tg in enumerate(all_time_groups)}

            # Map each data point to its correct position on the x-axis
            x_positions = [time_group_to_index[tg] for tg in user_df["TimeGroup"]]

            ax1.plot(
                x_positions,
                user_df["Efficiency"],
                marker="o",
                label=f"{user} (Efficiency)",
                color=colors[idx],
                linewidth=2,
                markersize=6,
            )
            if show_secondary_y:
                ax2.plot(
                    x_positions,
                    user_df["Job_Count"],
                    marker="s",
                    label=f"{user} (Jobs)",
                    color=colors[idx],
                    alpha=0.6,
                    linestyle="--",
                    markersize=4,
                )
            # Annotate data points
            for _, row in user_df.iterrows():
                # Use the mapped position for annotations
                x_pos = time_group_to_index[row["TimeGroup"]]
                annotation_fields = {
                    "User": user,
                    "Time": row["TimeGroup_Str"],
                    "Efficiency": f"{row['Efficiency']:.6f}",
                    "GPU_Hours": f"{row.get('GPU_Hours', 0):.1f}",
                    "Job_Count": row["Job_Count"],
                }
                for field in exclude_fields:
                    annotation_fields.pop(field, None)
                annotation_text = "\n".join([f"{k}: {v}" for k, v in annotation_fields.items()])
                if annotation_style == "hover":
                    ax1.annotate(
                        annotation_text,
                        (x_pos, row["Efficiency"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                    )
                elif annotation_style == "combined":
                    ax1.annotate(
                        annotation_text,
                        (x_pos, row["Efficiency"]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        fontsize=7,
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    )
                annotation_data.append(annotation_fields)

        ax1.set_xlabel(f"Time Period ({time_unit})")
        ax1.set_ylabel("Average VRAM Efficiency")
        ax1.set_title(f"VRAM Efficiency Over Time ({time_unit})")
        ax1.set_xticks(range(len(all_time_groups_str)))
        ax1.set_xticklabels(all_time_groups_str, rotation=45, ha="right")
        lines1, labels1 = ax1.get_legend_handles_labels()
        if show_secondary_y:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.05, 1))
        else:
            ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
        table_df = pd.DataFrame(annotation_data)
        if annotation_style == "table" and annotation_data:
            print("\n" + "=" * 80)
            print("DETAILED METRICS TABLE")
            print("=" * 80)
            if not table_df.empty:
                print(table_df.to_string(index=False))
            print("=" * 80)
        return table_df

    def plot_vram_hours(
        self,
        users: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        days_back: int | None = None,
        time_unit: str | TimeUnitEnum = TimeUnitEnum.MONTHS.value,
        remove_zero_values: bool = True,
        max_points: int = 100,
        show_secondary_y: bool = False,
        exclude_fields: list[str] | None = None,
        annotation_style: str = "hover",
    ) -> pd.DataFrame:
        """
        Plot VRAM Hours over time for specific users (non-interactive version).

        Args:
            users (list[str]): List of user names to plot.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            days_back (int, optional): Number of days back from today to filter jobs.
            time_unit (str or TimeUnitEnum, optional): Time unit to group by ('Month', 'Week', 'Day').
            remove_zero_values (bool, optional): Whether to remove users with zero VRAM hours from the plot.
                Note that all time periods between the first and last available data point will be
                shown regardless.
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            show_secondary_y (bool, optional): Whether to show job counts on secondary y-axis.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            annotation_style (str, optional): Style for annotations ("hover", "combined", "table", "none").

        Returns:
            pd.DataFrame: DataFrame containing the metrics used for the table and annotations in the plot.
        """

        data = self.filter_jobs_by_date_range(start_date=start_date, end_date=end_date, days_back=days_back)
        data = self.group_jobs_by_time(data, time_unit)

        fig, ax1 = plt.subplots(figsize=(12, 8))
        if show_secondary_y:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Job Count", color="tab:gray")
            ax2.tick_params(axis="y", labelcolor="tab:gray")

        annotation_data = []
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(users))]
        if exclude_fields is None:
            exclude_fields = []

        # Determine all time groups to show on x-axis: union of all non-empty periods for selected users
        user_time_groups = set()
        user_time_groups_map = {}
        for user in users:
            user_data = data[data["User"] == user]
            user_time_groups_map[user] = set(user_data["TimeGroup"].dropna().unique())
            user_time_groups.update(user_time_groups_map[user])

        # Ensure continuous timeline by filling in missing time periods
        if user_time_groups:
            if time_unit == TimeUnitEnum.MONTHS.value:
                # Get min and max periods
                min_period = min(user_time_groups)
                max_period = max(user_time_groups)

                # Create a continuous range of all months between min and max
                all_time_groups = []
                current_period = min_period
                while current_period <= max_period:
                    all_time_groups.append(current_period)
                    # Move to next month
                    if current_period.month == 12:
                        current_period = pd.Period(f"{current_period.year + 1}-01", freq="M")
                    else:
                        current_period = pd.Period(f"{current_period.year}-{current_period.month + 1:02d}", freq="M")
            elif time_unit == TimeUnitEnum.WEEKS.value:
                # Get min and max periods
                min_period = min(user_time_groups)
                max_period = max(user_time_groups)

                # Create a continuous range of all weeks between min and max
                all_time_groups = []
                current_period = min_period
                while current_period <= max_period:
                    all_time_groups.append(current_period)
                    # Move to next week (add 7 days)
                    next_start = pd.to_datetime(str(current_period).split("/")[0]) + pd.Timedelta(days=7)
                    current_period = pd.Period(next_start, freq="W")
            else:
                # For other time units or empty data, just use sorted unique time groups
                all_time_groups = sorted(user_time_groups)
        else:
            # If no time groups found, use empty list
            all_time_groups = []

        # Create a dictionary to track job counts for each time group across all users
        time_group_job_counts = {tg: 0 for tg in all_time_groups}

        # First pass: calculate job counts for all time groups
        for user in users:
            user_data = data[data["User"] == user]
            if user_data.empty:
                continue
            for time_group in all_time_groups:
                group_data = user_data[user_data["TimeGroup"] == time_group]
                job_count = group_data["JobID"].count() if not group_data.empty else 0
                time_group_job_counts[time_group] += job_count

        # Trim leading and trailing zero-job time groups while maintaining continuity in the middle
        all_time_groups = self._trim_zero_job_time_groups(all_time_groups, time_group_job_counts, remove_zero_values)

        # Format time groups as strings appropriately based on time unit
        if time_unit == TimeUnitEnum.WEEKS.value:
            # For weeks, create a more readable format like "Week of Jun 2, 2025"
            all_time_groups_str = []
            for tg in all_time_groups:
                # Extract the start date from the period (format is like '2025-06-02/2025-06-08')
                week_start = pd.to_datetime(str(tg).split("/")[0])
                all_time_groups_str.append(f"Week of {week_start.strftime('%b %d, %Y')}")
        else:
            all_time_groups_str = [str(tg) for tg in all_time_groups]

        for idx, user in enumerate(users):
            user_data = data[data["User"] == user]
            if user_data.empty:
                continue
            grouped_hours = []
            grouped_job_counts = []
            for time_group in all_time_groups:
                group_data = user_data[user_data["TimeGroup"] == time_group]
                user_gpu_hours = group_data["job_hours"].sum() if not group_data.empty else 0
                grouped_hours.append(user_gpu_hours)
                grouped_job_counts.append(group_data["JobID"].count() if not group_data.empty else 0)
            user_df = pd.DataFrame(
                {
                    "TimeGroup": all_time_groups,
                    "TimeGroup_Str": all_time_groups_str,
                    "VRAM_Hours": grouped_hours,
                    "Job_Count": grouped_job_counts,
                }
            )

            # Additional filtering for VRAM_Hours if needed
            if remove_zero_values:
                user_df = user_df[user_df["VRAM_Hours"] > 0]
            if len(user_df) > max_points:
                user_df = user_df.iloc[-max_points:]
            if user_df.empty:
                continue

            # Create mapping of time groups to their position in the x-axis
            time_group_to_index = {tg: i for i, tg in enumerate(all_time_groups)}

            # Map each data point to its correct position on the x-axis
            x_positions = [time_group_to_index[tg] for tg in user_df["TimeGroup"]]

            ax1.plot(
                x_positions,
                user_df["VRAM_Hours"],
                marker="o",
                label=f"{user} (VRAM Hours)",
                color=colors[idx],
                linewidth=2,
                markersize=6,
            )
            if show_secondary_y:
                ax2.plot(
                    x_positions,
                    user_df["Job_Count"],
                    marker="s",
                    label=f"{user} (Jobs)",
                    color=colors[idx],
                    alpha=0.6,
                    linestyle="--",
                    markersize=4,
                )
            # Annotate data points
            for _, row in user_df.iterrows():
                # Use the mapped position for annotations
                x_pos = time_group_to_index[row["TimeGroup"]]
                annotation_fields = {
                    "User": user,
                    "Time": row["TimeGroup_Str"],
                    "VRAM_Hours": f"{row['VRAM_Hours']:.1f}",
                    "Job_Count": row["Job_Count"],
                }
                for field in exclude_fields:
                    annotation_fields.pop(field, None)
                annotation_text = "\n".join([f"{k}: {v}" for k, v in annotation_fields.items()])
                if annotation_style == "hover":
                    ax1.annotate(
                        annotation_text,
                        (x_pos, row["VRAM_Hours"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                    )
                elif annotation_style == "combined":
                    ax1.annotate(
                        annotation_text,
                        (x_pos, row["VRAM_Hours"]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        fontsize=7,
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    )
                annotation_data.append(annotation_fields)

        ax1.set_xlabel(f"Time Period ({time_unit})")
        ax1.set_ylabel("VRAM Hours")
        ax1.set_title(f"VRAM Hours Over Time ({time_unit})")
        ax1.set_xticks(range(len(all_time_groups_str)))
        ax1.set_xticklabels(all_time_groups_str, rotation=45, ha="right")
        lines1, labels1 = ax1.get_legend_handles_labels()
        if show_secondary_y:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.05, 1))
        else:
            ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
        table_df = pd.DataFrame(annotation_data)
        if annotation_style == "table" and annotation_data:
            print("\n" + "=" * 80)
            print("DETAILED METRICS TABLE")
            print("=" * 80)
            if not table_df.empty:
                print(table_df.to_string(index=False))
            print("=" * 80)
        return table_df

    def _add_user_time_series_traces(
        self,
        fig,
        users,
        user_dfs,
        hover_texts,
        y_key: str,
        colors,
        job_count_trace: bool = False,
    ):
        """
        Helper to add user time series traces to a plotly figure.

        Args:
            fig: plotly figure
            users: list of user names
            user_dfs: list of user DataFrames (one per user)
            hover_texts: list of hover text lists (one per user)
            y_key: str, column to plot on y-axis ("Efficiency" or "VRAM_Hours")
            colors: list of color hex codes
            job_count_trace: bool, whether to add job count trace
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for interactive plotting. Please install it with 'pip install plotly'."
            ) from None

        for idx, (user_df, hover_text, user) in enumerate(zip(user_dfs, hover_texts, users, strict=True)):
            if user_df is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=user_df["TimeGroup_Datetime"],  # Use datetime for proper chronological ordering
                    y=user_df[y_key],
                    mode="lines+markers",
                    name=f"{user} ({y_key.replace('_', ' ')})",
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=8),
                    hovertext=hover_text,
                    hoverinfo="text",
                ),
                secondary_y=False,
            )
            if job_count_trace:
                fig.add_trace(
                    go.Scatter(
                        x=user_df["TimeGroup_Datetime"],  # Use datetime for proper chronological ordering
                        y=user_df["Job_Count"],
                        mode="lines+markers",
                        name=f"{user} (Job Count)",
                        line=dict(color=colors[idx % len(colors)], width=1, dash="dash"),
                        marker=dict(size=6, symbol="square"),
                        opacity=0.6,
                        hovertext=hover_text,
                        hoverinfo="text",
                    ),
                    secondary_y=True,
                )

    def plot_vram_efficiency_interactive(
        self,
        users: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        days_back: int | None = None,
        time_unit: TimeUnitEnum | str = TimeUnitEnum.MONTHS.value,
        remove_zero_values: bool = True,
        max_points: int = 100,
        exclude_fields: list[str] | None = None,
        job_count_trace: bool = False,
    ):
        """
        Create an interactive plot with tooltips showing detailed metrics.

        Args:
            users (list[str]): List of user names to plot.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            days_back (int, optional): Number of days back from today to filter jobs.
            time_unit (str or TimeUnitEnum, optional): Time unit to group by ('Month', 'Week', 'Day').
            remove_zero_values (bool, optional): Whether to remove zero efficiency values from the plot.
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            job_count_trace (bool, optional): Whether to add a trace for job counts on secondary
                y-axis. False by default.

        Returns:
            None: Generates an interactive plot with detailed tooltips.
        """
        try:
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return self.plot_vram_efficiency(
                users,
                start_date,
                end_date,
                days_back,
                time_unit,
                remove_zero_values,
                max_points,
                annotation_style="table",
            )

        # Filter data by date range or days back
        data = self.filter_jobs_by_date_range(start_date=start_date, end_date=end_date, days_back=days_back)

        # Group data by the specified time unit
        data = self.group_jobs_by_time(data, time_unit)

        if exclude_fields is None:
            exclude_fields = []

        # Use helper function to prepare consistent time series data
        all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict = self._prepare_time_series_data(
            data, users, time_unit, remove_zero_values
        )

        # Create subplots with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]], subplot_titles=[f"VRAM Efficiency Over Time ({time_unit})"]
        )

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        any_nonzero_efficiency = False
        user_dfs = []
        hover_texts: list[list[str]] = []

        for user in users:
            user_df = user_dfs_dict.get(user, pd.DataFrame())

            if user_df.empty:
                user_dfs.append(pd.DataFrame())
                hover_texts.append([])
                continue

            # Check if user has any non-zero efficiency
            if user_df["Efficiency"].sum() > 0:
                any_nonzero_efficiency = True

            # Filter for remove_zero_values and max_points
            if remove_zero_values:
                user_df = user_df[user_df["Efficiency"] > 0]

            if len(user_df) > max_points:
                user_df = user_df.iloc[-max_points:]

            user_dfs.append(user_df)

            # Create hover text for each point
            hover_text = []
            for _, row in user_df.iterrows():
                fields = {
                    "User": user,
                    "Time": row["TimeGroup_Str"],
                    "Efficiency": f"{row['Efficiency']:.6f}",
                    "GPU_Hours": f"{row['GPU_Hours']:.1f}",
                    "Job_Count": row["Job_Count"],
                }
                for field in exclude_fields:
                    fields.pop(field, None)
                hover_text.append("<br>".join([f"{k}: {v}" for k, v in fields.items()]))
            hover_texts.append(hover_text)

        # If all users have zero efficiency, call plot_vram_hours_interactive and return
        if not any_nonzero_efficiency:
            print("All users have zero efficiency. Plotting VRAM Hours instead.")
            return self.plot_vram_hours_interactive(
                users=users,
                start_date=start_date,
                end_date=end_date,
                days_back=days_back,
                time_unit=time_unit,
                max_points=max_points,
                exclude_fields=exclude_fields,
                job_count_trace=job_count_trace,
            )

        # Now plot only for users with nonzero efficiency
        self._add_user_time_series_traces(
            fig=fig,
            users=users,
            user_dfs=user_dfs,
            hover_texts=hover_texts,
            y_key="Efficiency",
            colors=colors,
            job_count_trace=job_count_trace,
        )

        # Update layout
        fig.update_layout(
            title=f"Interactive VRAM Efficiency Analysis ({time_unit})",
            xaxis_title=f"Time Period ({time_unit})",
            hovermode="closest",
            width=1000,
            height=600,
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="VRAM Efficiency / VRAM Hours", secondary_y=False)
        fig.update_yaxes(title_text="Job Count", secondary_y=True)

        fig.show()
        return fig

    def plot_vram_hours_interactive(
        self,
        users: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        days_back: int | None = None,
        time_unit: TimeUnitEnum | str = TimeUnitEnum.MONTHS.value,
        max_points: int = 100,
        exclude_fields: list[str] | None = None,
        job_count_trace: bool = False,
    ):
        """
        Create an interactive plot of VRAM Hours over time for users.

        Args:
            users (list[str]): List of user names to plot.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            days_back (int, optional): Number of days back from today to filter jobs.
            time_unit (str or TimeUnitEnum, optional): Time unit to group by ('Month', 'Week', 'Day').
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            job_count_trace (bool, optional): Whether to add a trace for job counts on secondary y-axis.

        Returns:
            None: Generates an interactive plot of VRAM Hours.
        """
        try:
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return

        data = self.filter_jobs_by_date_range(start_date=start_date, end_date=end_date, days_back=days_back)
        data = self.group_jobs_by_time(data, time_unit)

        if exclude_fields is None:
            exclude_fields = []

        # Use helper function to prepare consistent time series data
        all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict = self._prepare_time_series_data(
            data,
            users,
            time_unit,
            remove_zero_values=False,  # Don't remove zero values for VRAM hours
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles=[f"VRAM Hours Over Time ({time_unit})"])

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        user_dfs = []
        hover_texts: list[list[str]] = []

        for user in users:
            user_df = user_dfs_dict.get(user, pd.DataFrame())

            if user_df.empty:
                user_dfs.append(pd.DataFrame())
                hover_texts.append([])
                continue

            # Limit the number of points to plot
            if len(user_df) > max_points:
                user_df = user_df.iloc[-max_points:]

            user_dfs.append(user_df)

            # Create hover text for each point
            hover_text = []
            for _, row in user_df.iterrows():
                fields = {
                    "User": user,
                    "Time": row["TimeGroup_Str"],
                    "VRAM_Hours": f"{row['VRAM_Hours']:.1f}",
                    "Job_Count": row["Job_Count"],
                }
                for field in exclude_fields:
                    fields.pop(field, None)
                hover_text.append("<br>".join([f"{k}: {v}" for k, v in fields.items()]))
            hover_texts.append(hover_text)

        # Now plot for users with data
        self._add_user_time_series_traces(
            fig=fig,
            users=users,
            user_dfs=user_dfs,
            hover_texts=hover_texts,
            y_key="VRAM_Hours",
            colors=colors,
            job_count_trace=job_count_trace,
        )

        fig.update_layout(
            title=f"Interactive VRAM Hours Analysis ({time_unit})",
            xaxis_title=f"Time Period ({time_unit})",
            hovermode="closest",
            width=1000,
            height=600,
        )
        fig.update_yaxes(title_text="VRAM Hours", secondary_y=False)
        fig.update_yaxes(title_text="Job Count", secondary_y=True)
        fig.show()
        return fig


def filter_zero_vram_requested_with_gpu_allocated(df, requested_vram=0, gpus_min=1):
    """
    Return jobs where requested_vram is greater than or equal to a value (default 0) and GPUs >= gpus_min (default 1).

    Args:
        df (pd.DataFrame): The jobs DataFrame.
        requested_vram (int, float): Value to filter requested_vram
        gpus_min (int): Minimum number of GPUs allocated

    Returns:
        pd.DataFrame: Filtered DataFrame with jobs meeting the criteria
    """
    return df[(df["requested_vram"] >= requested_vram) & (df["GPUs"] >= gpus_min)].copy()
