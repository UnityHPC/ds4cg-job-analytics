"""
Functions to analyze efficiency of Jobs based on their VRAM usage.

The aim is to identify potential inefficiencies in GPU usage and notify users or PIs about these issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections.abc import Callable
from src.preprocess.preprocess import preprocess_data
from src.database import DatabaseConnection
from src.config.constants import DEFAULT_MIN_ELAPSED_SECONDS


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
        jobs_df,
        min_elapsed_seconds=0,
        include_failed_cancelled_jobs=False,
        include_cpu_only_jobs=False
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
        try:
            self.jobs_df = load_jobs_dataframe_from_duckdb(db_path, table_name, sample_size, random_state)
            self.efficiency_df = None
            self.analysis_results = None
        except Exception as e:
            raise ValueError(f"Failed to load jobs DataFrame: {e}") from e

    def _apply_numeric_filter(
        self,
        col: pd.Series,
        filter: int | list | set | tuple | dict | Callable,
        inclusive: bool | None
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
                        col.ge(vram_constraint_filter["min"])
                        if inclusive
                        else col.gt(vram_constraint_filter["min"])
                    )
                if "max" in vram_constraint_filter:
                    mask &= (
                        col.le(vram_constraint_filter["max"])
                        if inclusive
                        else col.lt(vram_constraint_filter["max"])
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
            mask &= self._apply_numeric_filter(
                self.jobs_df["GPUs"], gpu_count_filter, gpu_count_inclusive
            )

        return self.jobs_df[
            mask
            & (self.jobs_df["Elapsed"].dt.total_seconds() >= elapsed_seconds_min)
        ].copy()
        

    def calculate_efficiency_metrics(
        self,
        filtered_jobs :pd.DataFrame,
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
            filtered_jobs["cpu_gpu_ratio"] = (
                filtered_jobs["cpu_memory_gb"] /
                filtered_jobs["gpu_memory_used_gb"].clip(lower=0.1)
            )

        # Duration analysis
        filtered_jobs["duration_category"] = pd.cut(
            filtered_jobs["gpu_hours"],
            bins=[0, 1, 6, 24, 48, float("inf")],
            labels=["Short (<1h)", "Medium (1-6h)", "Long (6-24h)", "Under two days (24-48h)", "Over two days (>48h)"],
        )

        self.efficiency_df = filtered_jobs
        return self.efficiency_df


    def evaluate_cpu_gpu_usage(
            self,
            hours_percentage_threshold=25,
            vram_efficiency_threshold=0.3
        ):
        """
        This method evaluates the efficiency of GPU jobs based on VRAM usage and CPU-GPU balance.

        Args:
            hours_percentage_threshold (float): Threshold for high waste GPU hours as a percentage of total GPU hours
            vram_efficiency_threshold (float): Threshold for VRAM efficiency to consider a job as high waste

        Returns:
            dict: Analysis results with balance patterns and recommendations
        """
        #TODO: Needs refactoring to parametrize the analysis thresholds and make it more flexible
        #TODO: Need to separate the analysis into different methods for clarity
        #TODO: Separate the CPU-GPU balance analysis from the VRAM efficiency analysis

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
        inefficient_users = inefficient_users.sort_values(
            "Weighted_Efficiency_Contribution",
            ascending=True
        )
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
        inefficient_pis = inefficient_pis.sort_values(
            "Weighted_Efficiency_Contribution",
            ascending=True
        )
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
