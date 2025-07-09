"""
Functions to analyze efficiency of Jobs based on their VRAM usage.

The aim is to identify potential inefficiencies in GPU usage and notify users or PIs about these issues.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocess.preprocess import preprocess_data
from src.database import DatabaseConnection
from src.config.constants import DEFAULT_MIN_ELAPSED_SECONDS
from datetime import timedelta, datetime
import warnings


# gpu_meme_usage. {min: 90, max: 1000, exclude: True}


def load_jobs_dataframe_from_duckdb(
    db_path=None,
    table_name="Jobs",
    sample_size=None,
    random_state=None,
    days_back=None,
    custom_query="",
    include_failed_cancelled_jobs=False,
    include_cpu_only_jobs=False,
    min_elasped_seconds=0,
):
    """
    Connect to the DuckDB slurm_data_small.db and return the jobs table as a pandas DataFrame.

    Args:
        db_path (str or Path, optional): Path to the DuckDB database. Defaults to 'data/slurm_data_small.db'.
        table_name (str, optional): Table name to query. Defaults to 'Jobs'.
        days_back (int, optional): Number of days back to filter jobs based on StartTime.
            Deafults to None. If None, will not filter by startTime.
        custom_query(str, optional): Custom SQL query to execute. Defaults to an empty string.
            If empty, will select all jobs.
        include_failed_cancelled_jobs (bool, optional): If True, include jobs with FAILED or CANCELLED status.
            Defaults to False.
        include_cpu_only_jobs (bool, optional): If True, include jobs that do not use GPUs (CPU-only jobs).
            Defaults to False.
        min_eslaped_seconds (int, optional): Minimum elapsed time in seconds to filter jobs by elapsed time.
            Defaults to 0.

    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """

    def _contain_dates_back_condition(query: str) -> bool:
        pattern = r"(?i:WHERE)\s+[^;]*StartTime\s*>=\s*[^;]+"
        return bool(re.search(pattern, query, re.IGNORECASE))

    if db_path is None:
        db_path = Path(__file__).resolve().parents[2] / "data" / "slurm_data_small.db"

    db = None
    try:
        db = DatabaseConnection(str(db_path))
        jobs_df = None
        if not custom_query:
            custom_query = f"SELECT * FROM {table_name}"
        if days_back is not None and not _contain_dates_back_condition(custom_query):
            cutoff = datetime.now() - timedelta(days=days_back)
            if "where" not in custom_query.lower():
                custom_query += f" WHERE StartTime >= '{cutoff}'"
            else:
                custom_query += f" AND StartTime >= '{cutoff}'"
        elif days_back is not None and _contain_dates_back_condition(custom_query):
            warnings.warn(
                f"Parameter days_back = {days_back} is passed but custom_query already contained conditions for "
                "filtering by dates_back. dates_back condition in custom_query will be used.",
                UserWarning,
                stacklevel=2,
            )
        jobs_df = db.fetch_query(custom_query)

    except Exception as e:
        raise Exception(f"Exception at load_jobs_dataframe_from_duck_db: {e}") from e

    processed_data = preprocess_data(
        jobs_df,
        min_elapsed_seconds=min_elasped_seconds,
        include_failed_cancelled_jobs=include_failed_cancelled_jobs,
        include_cpu_only_jobs=include_cpu_only_jobs,
    )
    if sample_size is not None:
        processed_data = processed_data.sample(n=sample_size, random_state=random_state)
    return processed_data


class EfficiencyAnalysis:
    """
    Class to encapsulate the efficiency analysis of jobs based on various metrics.

    It provides methods to load data, analyze workload efficiency, and evaluate CPU-GPU usage patterns.
    """

    def __init__(self, db_path=None, table_name="Jobs", sample_size=None, random_state=None):
        self.jobs_df = load_jobs_dataframe_from_duckdb(db_path, table_name, sample_size, random_state)
        self.efficiency_df = None
        self.analysis_results = None

    def filter_jobs_for_analysis(
        self,
        vram_constraint_filter=None,
        allocated_vram_greater_than=0,
        gpu_mem_usage_min=None,
        gpu_mem_usage_max=None,
        gpu_mem_usage_exact=None,
        gpus_min=1,
        elapsed_seconds_min=DEFAULT_MIN_ELAPSED_SECONDS,
    ):
        """
        Filter jobs based on VRAM constraints, GPU allocation, and usage criteria.

        Args:
            vram_constraint_filter:
                - None: no filtering on vram_constraint
                - int or float: select rows where vram_constraint == value
                - list/set/tuple: select rows where vram_constraint is in the list
                - dict with 'min'/'max': select rows in the range (inclusive)
                - pd.NA or <NA>: select rows where vram_constraint is Nullable Int64 (i.e., pd.NA)
                - callable: custom filter function
            allocated_vram_greater_than (int, float): allocated_vram greater than this value
            gpu_mem_usage_min (int, float, optional): Minimum GPUMemUsage (inclusive)
            gpu_mem_usage_max (int, float, optional): Maximum GPUMemUsage (inclusive)
            gpu_mem_usage_exact (int, float, optional): If set, select only rows where GPUMemUsage == this value
            gpus_min (int): Minimum GPUs allocated
            elapsed_seconds_min (int): Minimum elapsed time in seconds

        Returns:
            DataFrame: Filtered jobs DataFrame based on the specified criteria.
        """

        mask = pd.Series([True] * len(self.jobs_df), index=self.jobs_df.index)

        if vram_constraint_filter is not None:
            col = self.jobs_df["vram_constraint"]
            # Handle pd.NA and nullable Int64
            if callable(vram_constraint_filter):
                mask &= col.apply(vram_constraint_filter)
            elif isinstance(vram_constraint_filter, list | set | tuple):
                mask &= col.isin(vram_constraint_filter)
            elif isinstance(vram_constraint_filter, dict):
                if "min" in vram_constraint_filter:
                    mask &= col.ge(vram_constraint_filter["min"])
                if "max" in vram_constraint_filter:
                    mask &= col.le(vram_constraint_filter["max"])
            elif vram_constraint_filter is pd.NA or (
                isinstance(vram_constraint_filter, float) and np.isnan(vram_constraint_filter)
            ):
                mask &= col.isna()
            else:
                # For nullable Int64, use .eq for safe comparison
                mask &= col.eq(vram_constraint_filter)

        # GPU memory usage filter
        gpu_mem_mask = pd.Series([True] * len(self.jobs_df), index=self.jobs_df.index)
        if gpu_mem_usage_exact is not None:
            gpu_mem_mask &= self.jobs_df["GPUMemUsage"] == gpu_mem_usage_exact
        else:
            if gpu_mem_usage_min is not None:
                gpu_mem_mask &= self.jobs_df["GPUMemUsage"] >= gpu_mem_usage_min
            if gpu_mem_usage_max is not None:
                gpu_mem_mask &= self.jobs_df["GPUMemUsage"] <= gpu_mem_usage_max

        return self.jobs_df[
            mask
            & (self.jobs_df["allocated_vram"] > allocated_vram_greater_than)
            & gpu_mem_mask
            & (self.jobs_df["GPUs"] >= gpus_min)
            & (self.jobs_df["Elapsed"].dt.total_seconds() >= elapsed_seconds_min)
        ].copy()

    def calculate_efficiency_metrics(
        self,
        filtered_jobs: pd.DataFrame,
    ):
        """
        Analyze jobs based on constraints, GPU allocation, and usage criteria.

        Args:
            filtered_jobs (DataFrame): DataFrame containing jobs to analyze.

        Returns:
            DataFrame: Jobs with efficiency metrics added
        """

        # Calculate efficiency metrics
        filtered_jobs["gpu_memory_used_gb"] = filtered_jobs["GPUMemUsage"] / (2**30)
        filtered_jobs["vram_efficiency"] = filtered_jobs["gpu_memory_used_gb"] / filtered_jobs["allocated_vram"]
        filtered_jobs["gpu_hours"] = (filtered_jobs["Elapsed"].dt.total_seconds() * filtered_jobs["GPUs"]) / 3600

        # Calculate weighted_vram_efficiency per job, normalized by total gpu_hours for that specific User
        user_gpu_hours = filtered_jobs.groupby("User")["gpu_hours"].transform("sum")
        filtered_jobs["user_weighted_vram_efficiency"] = (
            filtered_jobs["vram_efficiency"] * 100 * filtered_jobs["gpu_hours"]
        ) / user_gpu_hours

        # Calculate weighted vram efficiency per job, normalized by total gpu_hours for that specific PI
        pi_gpu_hours = filtered_jobs.groupby("Account")["gpu_hours"].transform("sum")
        filtered_jobs["pi_weighted_vram_efficiency"] = (
            filtered_jobs["vram_efficiency"] * 100 * filtered_jobs["gpu_hours"]
        ) / pi_gpu_hours

        # Categorize by efficiency
        filtered_jobs["efficiency_category"] = pd.cut(
            filtered_jobs["vram_efficiency"],
            bins=[0, 0.1, 0.3, 0.6, 1.0],
            labels=["Very Low (<10%)", "Low (10-30%)", "Medium (30-60%)", "High (60-100%)"],
        )

        # Add CPU memory analysis if available
        if "CPUMemUsage" in self.jobs_df.columns:
            filtered_jobs["cpu_memory_gb"] = filtered_jobs["CPUMemUsage"] / (2**30)
            filtered_jobs["cpu_gpu_ratio"] = filtered_jobs["cpu_memory_gb"] / filtered_jobs["gpu_memory_used_gb"].clip(
                lower=0.1
            )

        # Duration analysis
        filtered_jobs["duration_category"] = pd.cut(
            filtered_jobs["gpu_hours"],
            bins=[0, 1, 6, 24, 48, float("inf")],
            labels=["Short (<1h)", "Medium (1-6h)", "Long (6-24h)", "Under two days (24-48h)", "Over two days (>48h)"],
        )

        self.efficiency_df = filtered_jobs
        return self.efficiency_df

    def evaluate_cpu_gpu_usage(self, hours_percentage_threshold=25, vram_efficiency_threshold=0.3):
        """
        This method evaluates the efficiency of GPU jobs based on VRAM usage and CPU-GPU balance.

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

        # Ensure efficiency_df is available
        if self.efficiency_df is None:
            raise ValueError("Efficiency DataFrame is not available. Please run analyze_workload_efficiency first.")

        # Overall statistics
        analysis["total_jobs"] = len(self.efficiency_df)
        analysis["total_gpu_hours"] = self.efficiency_df["gpu_hours"].sum()
        analysis["avg_efficiency"] = self.efficiency_df["vram_efficiency"].mean()
        analysis["median_efficiency"] = self.efficiency_df["vram_efficiency"].median()

        # Efficiency distribution analysis
        efficiency_analysis = (
            self.efficiency_df.groupby("efficiency_category", observed=False)
            .agg(
                {
                    "JobID": "count",
                    "gpu_hours": "sum",
                    "vram_efficiency": "mean",
                    "allocated_vram": "mean",
                    "gpu_memory_used_gb": "mean",
                }
            )
            .round(3)
        )

        efficiency_analysis.columns = [
            "Job_Count",
            "GPU_Hours",
            "Avg_Efficiency",
            "Avg_Allocated_GB",
            "Avg_Used_GB",
        ]
        efficiency_analysis["Share of tota GPU Hours"] = (
            efficiency_analysis["GPU_Hours"] / analysis["total_gpu_hours"] * 100
        ).round(1)
        analysis["efficiency_patterns"] = efficiency_analysis

        # CPU-GPU balance analysis (if CPU data available)
        if "cpu_gpu_ratio" in self.efficiency_df.columns:
            # Categorize workloads by CPU-GPU balance
            self.efficiency_df["workload_type"] = pd.cut(
                self.efficiency_df["cpu_gpu_ratio"],
                bins=[0, 1, 5, 20, float("inf")],
                labels=[
                    "GPU-intensive (CPU<GPU)",
                    "Balanced (CPU≈GPU)",
                    "CPU-heavy (CPU>GPU)",
                    "Very CPU-heavy (CPU>>GPU)",
                ],
            )

            balance_analysis = self.efficiency_df.groupby("workload_type", observed=False).agg(
                {"JobID": "count", "gpu_hours": "sum", "vram_efficiency": "mean", "cpu_gpu_ratio": "mean"}
            )
            analysis["cpu_gpu_balance"] = balance_analysis

        # Over-allocation analysis
        high_waste_jobs = self.efficiency_df[self.efficiency_df["vram_efficiency"] <= vram_efficiency_threshold]
        analysis["high_waste_jobs"] = len(high_waste_jobs)
        analysis["high_waste_gpu_hours"] = high_waste_jobs["gpu_hours"].sum()
        analysis["high_waste_hours_share"] = analysis["high_waste_gpu_hours"] / analysis["total_gpu_hours"] * 100

        # Duration vs efficiency correlation
        duration_efficiency = self.efficiency_df.groupby("duration_category", observed=False).agg(
            {"JobID": "count", "vram_efficiency": "mean", "gpu_hours": "sum"}
        )
        analysis["duration_efficiency_patterns"] = duration_efficiency

        # Generate recommendations
        analysis_report = []

        low_efficiency_hours = efficiency_analysis.loc[
            efficiency_analysis.index.isin(["Very Low (<10%)", "Low (10-30%)"]), "GPU_Hours"
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

    def find_inefficient_users_weighted_by_hours(self, efficiency_threshold=0.3, min_jobs=5):
        """
        Identify users with low average VRAM efficiency across their jobs, weighted by the hours they were inefficient.

        Args:
            efficiency_threshold (float): Threshold for VRAM efficiency to consider a user as inefficient
            min_jobs (int): Minimum number of jobs a user must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with users and their average VRAM efficiency
        """
        if self.efficiency_df is None:
            raise ValueError("Efficiency DataFrame is not available. Please run calculate_efficiency_metrics first.")

        inefficient_users = (
            self.efficiency_df[self.efficiency_df["vram_efficiency"] < efficiency_threshold]
            .groupby("User", observed=False)
            .agg(
                Job_Count=("JobID", "count"),
                Avg_Allocated_VRAM=("allocated_vram", "mean"),
                Total_GPU_Hours=("gpu_hours", "sum"),
                Avg_GPUs=("GPUs", "mean"),
                Avg_Weighted_VRAM_Efficiency=("user_weighted_vram_efficiency", "mean"),
            )
            .reset_index()
        )

        # Multiply share of total gpu hours by weighted vram efficiency to get the new metric
        inefficient_users["Weighted_Efficiency_Contribution"] = (
            inefficient_users["Total_GPU_Hours"]
            * inefficient_users["Avg_Weighted_VRAM_Efficiency"]
            / inefficient_users["Total_GPU_Hours"].sum()
        )

        # Only include users with at least 5 jobs
        inefficient_users = inefficient_users[inefficient_users["Job_Count"] >= 5]

        # Sort by the new metric ascending (lower is worse)
        inefficient_users = inefficient_users.sort_values("Weighted_Efficiency_Contribution", ascending=True)
        return inefficient_users

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
            self.efficiency_df[self.efficiency_df["vram_efficiency"] < efficiency_threshold]
            .groupby("Account", observed=False)
            .agg(
                Job_Count=("JobID", "count"),
                Avg_Allocated_VRAM=("allocated_vram", "mean"),
                Total_GPU_Hours=("gpu_hours", "sum"),
                Avg_GPUs=("GPUs", "mean"),
                Avg_Weighted_VRAM_Efficiency=("pi_weighted_vram_efficiency", "mean"),
            )
            .reset_index()
        )

        # Multiply share of total gpu hours by weighted vram efficiency to get the new metric
        inefficient_pis["Weighted_Efficiency_Contribution"] = (
            inefficient_pis["Total_GPU_Hours"]
            * inefficient_pis["Avg_Weighted_VRAM_Efficiency"]
            / inefficient_pis["Total_GPU_Hours"].sum()
        )

        # Only include PIs with at least 5 jobs
        inefficient_pis = inefficient_pis[inefficient_pis["Job_Count"] >= min_jobs]

        # Sort by the new metric ascending (lower is worse)
        inefficient_pis = inefficient_pis.sort_values("Weighted_Efficiency_Contribution", ascending=True)
        return inefficient_pis


def filter_zero_vram_requested_with_gpu_allocated(df, requested_vram=0, gpus_min=1):
    """
    Return jobs where requested_vram is greater than or equal to a value (default 0) and GPUs >= gpus_min (default 1).

    Args:
        df (pd.DataFrame): The jobs DataFrame.
        requested_vram (int, float): Value to filter requested_vram
        gpus_min (int): Minimum GPUs allocated

    Returns:
        pd.DataFrame: Filtered DataFrame with jobs matching the criteria.
    """
    return df[(df["requested_vram"] >= requested_vram) & (df["GPUs"] >= gpus_min)]
