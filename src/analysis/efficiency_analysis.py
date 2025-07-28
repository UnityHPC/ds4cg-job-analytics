"""
Tools to analyze efficiency of Jobs based on their VRAM usage.

The aim is to identify potential inefficiencies in GPU usage and notify users or PIs about these issues.
"""

from typing import cast
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocess.preprocess import preprocess_data
from src.database import DatabaseConnection
from src.config.constants import DEFAULT_MIN_ELAPSED_SECONDS
from src.config.enum_constants import (
    FilterTypeEnum,
    MetricsDataFrameNameEnum,
    JobEfficiencyMetricsEnum,
    UserEfficiencyMetricsEnum,
)


def load_preprocessed_jobs_dataframe_from_duckdb(
    db_path: str | Path,
    table_name: str = "Jobs",
    sample_size: int | None = None,
    random_state: pd._typing.RandomState | None = None,
) -> pd.DataFrame:
    """
    Load jobs DataFrame from a DuckDB database and preprocess it.

    Args:
        db_path (str or Path): Path to the DuckDB database.
        table_name (str, optional): Table name to query. Defaults to 'Jobs'.
        sample_size (int, optional): Number of rows to sample from the DataFrame. Defaults to None (no sampling).
        random_state (pd._typing.RandomState, optional): Random state for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the table data.

    Raises:
        RuntimeError: If the jobs DataFrame cannot be loaded from the database.
    """
    if isinstance(db_path, Path):
        db_path = db_path.resolve()
    try:
        db = DatabaseConnection(str(db_path))

        jobs_df = db.fetch_all_jobs(table_name=table_name)
        processed_data = preprocess_data(
            jobs_df, min_elapsed_seconds=0, include_failed_cancelled_jobs=False, include_cpu_only_jobs=False
        )
        if sample_size is not None:
            processed_data = processed_data.sample(n=sample_size, random_state=random_state)
        return processed_data
    except Exception as e:
        raise RuntimeError(f"Failed to load jobs DataFrame: {e}") from e


class EfficiencyAnalysis:
    """
    Class to encapsulate the efficiency analysis of jobs based on various metrics.

    It provides methods to load data, analyze workload efficiency, and evaluate CPU-GPU usage patterns.

    The metrics are generated in separate DataFrames for each category in MetricsDataFrameNameEnum.
    """

    def __init__(
        self,
        jobs_df: pd.DataFrame,
    ) -> None:
        """
        Initialize the EfficiencyAnalysis class.

        Args:
            jobs_df (pd.DataFrame): DataFrame containing job data.

        Raises:
            ValueError: If the jobs DataFrame is empty.
        """
        if jobs_df.empty:
            raise ValueError("The jobs DataFrame is empty. Please provide a valid DataFrame with job data.")
        self.jobs_df = jobs_df
        # Initialize efficiency metric class attributes to None
        for var in MetricsDataFrameNameEnum:
            setattr(self, var.value, None)
        self.analysis_results: dict | None = None

    @staticmethod
    def is_numeric_type(val: object) -> bool:
        """
        Check if the value is a numeric type (int, float, np.integer, np.floating, pd.Int64Dtype, pd.Float64Dtype).

        Args:

            val (object): The value to check.

        Returns:
            bool: True if the value is numeric, False otherwise.
        """
        return pd.api.types.is_integer_dtype(type(val)) or pd.api.types.is_float_dtype(type(val))

    @staticmethod
    def apply_numeric_filter(
        col: pd.Series,
        filter: int | float | list | set | tuple | dict | pd.api.typing.NAType,
        permissible_filter_types: set[FilterTypeEnum],
        filter_name: str,
    ) -> pd.Series:
        """
        Helper to apply a numeric filter to a pandas Series.

        Args:
            col (pd.Series): The column to filter.
            filter (int | float | list | set | tuple | dict | pd.api.typing.NAType): The filter value(s).
            permissible_filter_types (set[FilterTypeEnum]): Set of permissible filter types.
            filter_name (str): Name of the filter.

        Returns:
            pd.Series: Boolean mask.

        Raises:
            ValueError: If the filter is invalid.
        """
        mask = pd.Series([True] * len(col), index=col.index)
        if filter is not None:
            if filter is pd.NA or (isinstance(filter, float) and np.isnan(filter)):
                if FilterTypeEnum.PD_NA not in permissible_filter_types:
                    raise ValueError(f"{filter_name} cannot be pd.NA or <NA>.")
                mask &= col.isna()
            elif isinstance(filter, list | set | tuple):
                # Check if the filter is a list, set, or tuple and if all values are numeric
                # If one of list, set, or tuple is allowed, then we assume all are allowed
                valid_filter_types_set = {
                    FilterTypeEnum.LIST,
                    FilterTypeEnum.SET,
                    FilterTypeEnum.TUPLE,
                }
                if not valid_filter_types_set.issubset(permissible_filter_types):
                    raise ValueError(f"{filter_name} cannot be a list, set, or tuple.")
                if not all(EfficiencyAnalysis.is_numeric_type(val) for val in filter):
                    raise ValueError("All filter values must be integers or floats.")
                mask &= col.isin(filter)
            elif isinstance(filter, dict):
                if FilterTypeEnum.DICTIONARY not in permissible_filter_types:
                    raise ValueError(f"{filter_name} cannot be a dictionary.")
                if "inclusive" not in filter or not isinstance(filter["inclusive"], bool):
                    raise ValueError("If a filter is a dict, it must include an 'inclusive' boolean key.")
                inclusive = filter["inclusive"]
                # Check min/max values are int or float if present using pandas type checks
                for key in ("min", "max"):
                    if key in filter and not EfficiencyAnalysis.is_numeric_type(filter[key]):
                        raise ValueError(f"['{key}'] must be an integer or float.")
                if "min" in filter:
                    mask &= col.ge(filter["min"]) if inclusive else col.gt(filter["min"])
                if "max" in filter:
                    mask &= col.le(filter["max"]) if inclusive else col.lt(filter["max"])
            else:
                if FilterTypeEnum.NUMERIC_SCALAR not in permissible_filter_types:
                    raise ValueError(f"{filter_name} cannot be a numeric scalar.")
                else:
                    # Only allow numeric types
                    if EfficiencyAnalysis.is_numeric_type(filter):
                        numeric_filter = cast(float | int, filter)
                        mask &= col.eq(numeric_filter)
                    else:
                        raise ValueError(f"{filter_name} must be a numeric type.")
        return mask

    def filter_jobs_for_analysis(
        self,
        vram_constraint_filter: int | float | list | set | tuple | dict | pd.api.typing.NAType | None = None,
        gpu_mem_usage_filter: int | float | dict | None = None,
        allocated_vram_filter: int | float | list | set | tuple | dict | None = None,
        gpu_count_filter: int | float | list | set | tuple | dict | None = None,
        elapsed_seconds_min: int | float = DEFAULT_MIN_ELAPSED_SECONDS,
    ) -> pd.DataFrame:
        """
        Filter jobs based on VRAM constraints, GPU allocation, and usage criteria.

        Args:
            vram_constraint_filter:
                - None: no filtering on vram_constraint
                - int | float : select rows where vram_constraint == value
                - list/set/tuple: select rows where vram_constraint is in the values provided
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
                - pd.NA or <NA>: select rows where vram_constraint is Nullable Int64 (i.e., pd.NA)
            gpu_mem_usage_filter: the unit is bytes to match the GPUMemUsage column
                - None: no filtering on GPUMemUsage
                - int | float : select rows where GPUMemUsage == value
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
            allocated_vram_filter:
                - None: no filtering on allocated_vram
                - int | float : select rows where allocated_vram == value
                - list/set/tuple: select rows where allocated_vram is in the values provided
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
            gpu_count_filter:
                - None: no filtering on gpu_count
                - int | float : select rows where gpu_count == value
                - list/set/tuple: select rows where gpu_count is in the values provided
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
            elapsed_seconds_min (int): Minimum elapsed time in seconds

        Returns:
            pd.DataFrame: Filtered jobs DataFrame based on the specified criteria.

        Raises:
            ValueError: If the filter is invalid.
        """

        mask = pd.Series([True] * len(self.jobs_df), index=self.jobs_df.index)

        # vram_constraint
        if vram_constraint_filter is not None:
            try:
                mask &= EfficiencyAnalysis.apply_numeric_filter(
                    self.jobs_df["vram_constraint"],
                    vram_constraint_filter,
                    set(FilterTypeEnum.__members__.values()),
                    "vram_constraint_filter",
                )
            except ValueError as e:
                raise ValueError("Invalid vram_constraint_filter.") from e

        # GPU memory usage filter
        if gpu_mem_usage_filter is not None:
            try:
                mask &= EfficiencyAnalysis.apply_numeric_filter(
                    self.jobs_df["GPUMemUsage"],
                    gpu_mem_usage_filter,
                    {FilterTypeEnum.NUMERIC_SCALAR, FilterTypeEnum.DICTIONARY},
                    "gpu_mem_usage_filter",
                )
            except ValueError as e:
                raise ValueError("Invalid GPU memory usage filter.") from e

        # Allocated VRAM filter
        if allocated_vram_filter is not None:
            try:
                mask &= EfficiencyAnalysis.apply_numeric_filter(
                    self.jobs_df["allocated_vram"],
                    allocated_vram_filter,
                    set(FilterTypeEnum.__members__.values()).difference({FilterTypeEnum.PD_NA}),
                    "allocated_vram_filter",
                )
            except ValueError as e:
                raise ValueError("Invalid allocated VRAM filter.") from e

        # GPU count filter
        if gpu_count_filter is not None:
            try:
                mask &= EfficiencyAnalysis.apply_numeric_filter(
                    self.jobs_df["GPUs"],
                    gpu_count_filter,
                    set(FilterTypeEnum.__members__.values()).difference({FilterTypeEnum.PD_NA}),
                    "gpu_count_filter",
                )
            except ValueError as e:
                raise ValueError("Invalid GPU count filter.") from e

        # Filter by elapsed time
        if not EfficiencyAnalysis.is_numeric_type(elapsed_seconds_min) or elapsed_seconds_min < 0:
            raise ValueError("elapsed_seconds_min must be a positive integer or float representing seconds.")

        return self.jobs_df[mask & (self.jobs_df["Elapsed"].dt.total_seconds() >= elapsed_seconds_min)].copy()

    def calculate_job_efficiency_metrics(
        self,
        filtered_jobs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate jobs efficiency metrics for the filtered jobs DataFrame.

        Refer to the documentation for the definition of the metrics calculated.

        Args:
            filtered_jobs (pd.DataFrame): DataFrame containing jobs to analyze.

        Returns:
            pd.DataFrame: Jobs with efficiency metrics added
        """

        vram_hour_col_name = JobEfficiencyMetricsEnum.VRAM_HOURS.value
        job_hour_col_name = JobEfficiencyMetricsEnum.JOB_HOURS.value
        gpu_count_col_name = JobEfficiencyMetricsEnum.GPU_COUNT.value
        used_vram_col_name = JobEfficiencyMetricsEnum.USED_VRAM_GIB.value

        # rename GPUs to gpu_count for clarity
        filtered_jobs = filtered_jobs.rename(columns={"GPUs": gpu_count_col_name})

        # Calculate job efficiency metrics
        filtered_jobs.loc[:, job_hour_col_name] = (
            filtered_jobs["Elapsed"].dt.total_seconds() * filtered_jobs[gpu_count_col_name] / 3600
        )
        filtered_jobs.loc[:, vram_hour_col_name] = filtered_jobs["allocated_vram"] * filtered_jobs[job_hour_col_name]
        filtered_jobs.loc[:, used_vram_col_name] = filtered_jobs["GPUMemUsage"] / (2**30)
        # Compute alloc_vram_efficiency, a float in the range [0, 1].
        filtered_jobs.loc[:, JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY.value] = (
            filtered_jobs[used_vram_col_name] / filtered_jobs["allocated_vram"]
        )

        # Compute vram_constraint_efficiency, a nullable float in the range [0, 1]. Set to NA if vram_constraint is NA
        filtered_jobs.loc[:, JobEfficiencyMetricsEnum.VRAM_CONSTRAINT_EFFICIENCY.value] = (
            filtered_jobs[used_vram_col_name] / filtered_jobs["vram_constraint"]
        )

        # Calculate job allocated VRAM efficiency score
        # This is a log-transformed score that penalizes low efficiency and longer vram_hours
        alloc_vram_eff = filtered_jobs[JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY.value]
        filtered_jobs.loc[:, JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY_SCORE.value] = np.where(
            alloc_vram_eff > 0, np.log(alloc_vram_eff) * filtered_jobs[vram_hour_col_name], -np.inf
        )

        # Calculate vram_constraint_efficiency score
        vram_constraint_eff = filtered_jobs[JobEfficiencyMetricsEnum.VRAM_CONSTRAINT_EFFICIENCY.value]
        # Avoid log(0) and propagate pd.NA: if NA, score is NA; if 0, score is -np.inf
        score = pd.Series(pd.NA, index=filtered_jobs.index, dtype=pd.Float64Dtype())
        mask_valid = vram_constraint_eff.notna() & (vram_constraint_eff > 0)
        mask_zero = vram_constraint_eff.notna() & (vram_constraint_eff == 0)
        score[mask_valid] = np.log(vram_constraint_eff[mask_valid]) * filtered_jobs.loc[mask_valid, vram_hour_col_name]
        score[mask_zero] = -np.inf
        filtered_jobs.loc[:, JobEfficiencyMetricsEnum.VRAM_CONSTRAINT_EFFICIENCY_SCORE.value] = score

        # Add CPU memory metrics if available
        used_cpu_mem_col = JobEfficiencyMetricsEnum.USED_CPU_MEMORY_GIB.value
        allocated_cpu_mem_col = JobEfficiencyMetricsEnum.ALLOCATED_CPU_MEM_GIB.value
        if "CPUMemUsage" in self.jobs_df.columns and "Memory" in self.jobs_df.columns:
            filtered_jobs.loc[:, used_cpu_mem_col] = filtered_jobs["CPUMemUsage"] / (2**30)
            filtered_jobs.loc[:, allocated_cpu_mem_col] = filtered_jobs["Memory"] / (2**10)  # Memory is in MiB
            filtered_jobs.loc[:, JobEfficiencyMetricsEnum.CPU_MEM_EFFICIENCY.value] = (
                filtered_jobs[used_cpu_mem_col] / filtered_jobs[allocated_cpu_mem_col]
            )
            filtered_jobs = filtered_jobs.drop(columns=["CPUMemUsage", "Memory"])

        self.jobs_with_efficiency_metrics = filtered_jobs
        return self.jobs_with_efficiency_metrics

    def calculate_user_efficiency_metrics(self) -> pd.DataFrame:
        """
        Calculate user efficiency metrics based on job efficiency data.

        Returns:
            pd.DataFrame: DataFrame with users and their average VRAM efficiency
        """
        if self.jobs_with_efficiency_metrics is None:
            self.calculate_job_efficiency_metrics(self.jobs_df)
            print(
                "Jobs DataFrame with efficiency metrics was not available. "
                "Calculated it using the input jobs DataFrame."
            )

        job_vram_hour_col = JobEfficiencyMetricsEnum.VRAM_HOURS.value
        job_gpu_count_col = JobEfficiencyMetricsEnum.GPU_COUNT.value
        job_job_hour_col = JobEfficiencyMetricsEnum.JOB_HOURS.value
        # Compute user_job_hours_per_job once and reuse for both metrics
        user_job_hours_per_job = self.jobs_with_efficiency_metrics.groupby("User", observed=True)[
            job_job_hour_col
        ].transform("sum")

        def avg_non_inf(x: pd.Series) -> float | pd.api.typing.NAType:
            """
            Helper function to calculate the average of a Series, ignoring -np.inf values.

            Args:
                x (pd.Series): Series to calculate the average from.

            Returns:
                float: Average of the Series, ignoring -np.inf values. Returns pd.NA if no valid values.
            """
            valid = x[x != -np.inf]
            return valid.mean() if not valid.empty else pd.NA

        users_w_efficiency_metrics = (
            self.jobs_with_efficiency_metrics.groupby("User", observed=True)
            .agg(
                job_count=("JobID", "count"),
                user_job_hours=("job_hours", "sum"),
                pi_account=("Account", "first"),
                avg_alloc_vram_eff_score=("alloc_vram_efficiency_score", avg_non_inf),
                avg_vram_constraint_eff_score=("vram_constraint_efficiency_score", avg_non_inf),
            )
            .reset_index()
        )

        # change name of 2 calculated avergage column
        users_w_efficiency_metrics.rename({
            "avg_alloc_vram_eff_score": UserEfficiencyMetricsEnum.AVG_ALLOC_VRAM_EFFICIENCY_SCORE.value,
            "avg_vram_constraint_eff_score": UserEfficiencyMetricsEnum.AVG_VRAM_CONSTRAINT_EFFICIENCY_SCORE.value,
        })

        self.jobs_with_efficiency_metrics.loc[:, "weighted_alloc_vram_efficiency"] = (
            self.jobs_with_efficiency_metrics[JobEfficiencyMetricsEnum.ALLOC_VRAM_EFFICIENCY.value]
            * self.jobs_with_efficiency_metrics[job_vram_hour_col]
            / user_job_hours_per_job  # TODO: should be divided by sum of vram_hours, not job_hours
        )

        users_w_efficiency_metrics.loc[:, UserEfficiencyMetricsEnum.WEIGHTED_AVG_ALLOC_VRAM_EFFICIENCY.value] = (
            self.jobs_with_efficiency_metrics.groupby("User", observed=True)["weighted_alloc_vram_efficiency"]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        self.jobs_with_efficiency_metrics.loc[:, "weighted_vram_constraint_efficiency"] = (
            self.jobs_with_efficiency_metrics[JobEfficiencyMetricsEnum.VRAM_CONSTRAINT_EFFICIENCY.value]
            * self.jobs_with_efficiency_metrics[job_vram_hour_col]
            / user_job_hours_per_job  # TODO: should be divided by sum of vram_hours, not job_hours
        ).astype(pd.Float64Dtype())

        users_w_efficiency_metrics.loc[:, UserEfficiencyMetricsEnum.WEIGHTED_AVG_VRAM_CONSTRAINTS_EFFICIENCY.value] = (
            self.jobs_with_efficiency_metrics.groupby("User", observed=True)["weighted_vram_constraint_efficiency"]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        self.jobs_with_efficiency_metrics.loc[:, "weighted_gpu_count"] = (
            self.jobs_with_efficiency_metrics[job_gpu_count_col]
            * self.jobs_with_efficiency_metrics[job_vram_hour_col]
            / user_job_hours_per_job
        )
        users_w_efficiency_metrics.loc[:, UserEfficiencyMetricsEnum.WEIGHTED_AVG_GPU_COUNT.value] = (
            self.jobs_with_efficiency_metrics.groupby("User", observed=True)["weighted_gpu_count"]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        # Calculate metric representing the total amount of GPU memory resources a user has been allocated over time.
        # It answers the question: “How much VRAM, and for how long, did this user occupy?”
        users_w_efficiency_metrics.loc[:, UserEfficiencyMetricsEnum.VRAM_HOURS.value] = (
            (self.jobs_with_efficiency_metrics["allocated_vram"] * self.jobs_with_efficiency_metrics[job_job_hour_col])
            .groupby(self.jobs_with_efficiency_metrics["User"], observed=True)
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        self.jobs_with_efficiency_metrics = self.jobs_with_efficiency_metrics.drop(
            columns=["weighted_alloc_vram_efficiency", "weighted_vram_constraint_efficiency", "weighted_gpu_count"]
        )

        self.users_with_efficiency_metrics = users_w_efficiency_metrics
        return self.users_with_efficiency_metrics

    def find_inefficient_users_by_alloc_vram_efficiency(
        self, alloc_vram_efficiency_filter: int | float | dict | None, min_jobs: int = 5
    ) -> pd.DataFrame:
        """
        Identify users with low expected allocated VRAM efficiency across their jobs compared to others

        Args:
            alloc_vram_efficiency_filter:
                - int | float : select rows where expected_value_alloc_vram_efficiency == value
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
            min_jobs (int): Minimum number of jobs a user must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with users and their average VRAM efficiency

        Raises:
            ValueError: If the filter for expected_value_alloc_vram_efficiency is invalid.
        """
        if self.users_with_efficiency_metrics is None:
            self.calculate_user_efficiency_metrics()
            print(
                "Users DataFrame with efficiency metrics was not available. "
                "Calculated it using the DataFrame of jobs with efficiency metrics."
            )

        mask = pd.Series(
            [True] * len(self.users_with_efficiency_metrics), index=self.users_with_efficiency_metrics.index
        )

        if alloc_vram_efficiency_filter is not None:
            try:
                mask &= EfficiencyAnalysis.apply_numeric_filter(
                    self.users_with_efficiency_metrics["expected_value_alloc_vram_efficiency"],
                    alloc_vram_efficiency_filter,
                    {FilterTypeEnum.NUMERIC_SCALAR, FilterTypeEnum.DICTIONARY},
                    filter_name="expected_value_alloc_vram_efficiency",
                )
            except ValueError as e:
                raise ValueError("Invalid filter for expected_value_alloc_vram_efficiency.") from e

        col = self.users_with_efficiency_metrics["job_count"]
        mask &= col.ge(min_jobs)

        inefficient_users = self.users_with_efficiency_metrics[mask]

        # Sort by the metric ascending (lower is worse)
        inefficient_users = inefficient_users.sort_values("expected_value_alloc_vram_efficiency", ascending=True)
        return inefficient_users

    def find_inefficient_users_by_vram_hours(
        self, vram_hours_filter: int | float | dict = 200, min_jobs: int = 5
    ) -> pd.DataFrame:
        """
        Identify users with high VRAM-hours across their jobs compared to others.

        Args:
            vram_hours_filter:
                - None: no filtering on vram_hours
                - int | float: select rows where vram_hours == value
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
            min_jobs (int): Minimum number of jobs a user must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with users and their total VRAM hours

        Raises:
            ValueError: If the filter is invalid
        """
        if self.users_with_efficiency_metrics is None:
            self.calculate_user_efficiency_metrics()
            print(
                "Users DataFrame with efficiency metrics was not available. "
                "Calculated it using the DataFrame of jobs with efficiency metrics."
            )

        mask = pd.Series(
            [True] * len(self.users_with_efficiency_metrics), index=self.users_with_efficiency_metrics.index
        )

        if vram_hours_filter is not None:
            try:
                mask &= EfficiencyAnalysis.apply_numeric_filter(
                    self.users_with_efficiency_metrics["vram_hours"],
                    vram_hours_filter,
                    {FilterTypeEnum.NUMERIC_SCALAR, FilterTypeEnum.DICTIONARY},
                    filter_name="vram_hours_filter",
                )
            except ValueError as e:
                raise ValueError("Invalid filter for vram_hours.") from e

        col = self.users_with_efficiency_metrics["job_count"]
        mask &= col.ge(min_jobs)

        inefficient_users = self.users_with_efficiency_metrics[mask]

        # Sort by the metric descending (higher is worse)
        inefficient_users = inefficient_users.sort_values("vram_hours", ascending=False)
        return inefficient_users

    def calculate_all_efficiency_metrics(self, filtered_jobs: pd.DataFrame) -> dict:
        """
        Calculate all efficiency metrics for jobs, users, and PI accounts.

        This method is a convenience wrapper that calculates job efficiency metrics,
        user efficiency metrics, and PI account efficiency metrics in sequence.

        Args:
            filtered_jobs (pd.DataFrame): DataFrame containing jobs to analyze.

        Returns:
            dict: A dictionary containing DataFrames with efficiency metrics for jobs, users, and PI accounts.

        Raises:
            RuntimeError: If any of the calculations fail.
        """
        try:
            self.calculate_job_efficiency_metrics(filtered_jobs)
            self.calculate_user_efficiency_metrics()
            self.calculate_pi_account_efficiency_metrics()
            return {var.value: getattr(self, var.value) for var in MetricsDataFrameNameEnum}

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            raise RuntimeError(f"Failed to calculate all efficiency metrics: {e}") from e

    def calculate_pi_account_efficiency_metrics(self) -> pd.DataFrame:
        """
        Calculate PI account efficiency metrics based on user efficiency data.

        For a group of users, we calculate the expected value of user metrics for the group of users
        The weights for the expected value are the vram_hours of each user in the group

        Returns:
            pd.DataFrame: DataFrame with PI accounts and their efficiency metrics
        """
        if self.users_with_efficiency_metrics is None:
            self.calculate_user_efficiency_metrics()
            print(
                "Users DataFrame with efficiency metrics was not available. "
                "Calculated it using the  DataFrame of jobs with efficiency metrics."
            )

        pi_efficiency_metrics = (
            self.users_with_efficiency_metrics.groupby("pi_account", observed=True)
            .agg(
                job_count=("job_count", "sum"),
                pi_acc_job_hours=("user_job_hours", "sum"),
                user_count=("User", "nunique"),
                pi_acc_vram_hours=("vram_hours", "sum"),
                avg_alloc_vram_efficiency_score=("avg_alloc_vram_efficiency_score", "mean"),
                avg_vram_constraint_efficiency_score=("avg_vram_constraint_efficiency_score", "mean"),
            )
            .reset_index()
        )

        # Compute pi_acc_vram_hours once and reuse for both metrics
        pi_acc_vram_hours = self.users_with_efficiency_metrics.groupby("pi_account", observed=True)[
            "vram_hours"
        ].transform("sum")

        self.users_with_efficiency_metrics.loc[:, "weighted_ev_alloc_vram_efficiency"] = (
            self.users_with_efficiency_metrics["expected_value_alloc_vram_efficiency"]
            * self.users_with_efficiency_metrics["vram_hours"]
            / pi_acc_vram_hours
        )

        pi_efficiency_metrics.loc[:, "expected_value_alloc_vram_efficiency"] = (
            self.users_with_efficiency_metrics.groupby("pi_account", observed=True)[
                "weighted_ev_alloc_vram_efficiency"
            ]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        self.users_with_efficiency_metrics.loc[:, "weighted_ev_vram_constraint_efficiency"] = (
            self.users_with_efficiency_metrics["expected_value_vram_constraint_efficiency"]
            * self.users_with_efficiency_metrics["vram_hours"]
            / pi_acc_vram_hours
        )

        pi_efficiency_metrics.loc[:, "expected_value_vram_constraint_efficiency"] = (
            self.users_with_efficiency_metrics.groupby("pi_account", observed=True)[
                "weighted_ev_vram_constraint_efficiency"
            ]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        self.users_with_efficiency_metrics.loc[:, "weighted_ev_gpu_count"] = (
            self.users_with_efficiency_metrics["expected_value_gpu_count"]
            * self.users_with_efficiency_metrics["vram_hours"]
            / pi_acc_vram_hours
        )
        pi_efficiency_metrics.loc[:, "expected_value_gpu_count"] = (
            self.users_with_efficiency_metrics.groupby("pi_account", observed=True)["weighted_ev_gpu_count"]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        self.users_with_efficiency_metrics = self.users_with_efficiency_metrics.drop(
            columns=[
                "weighted_ev_alloc_vram_efficiency",
                "weighted_ev_vram_constraint_efficiency",
                "weighted_ev_gpu_count",
            ]
        )

        self.pi_accounts_with_efficiency_metrics = pi_efficiency_metrics
        return self.pi_accounts_with_efficiency_metrics

    def find_inefficient_pis_by_vram_hours(
        self, vram_hours_filter: int | float | dict = 200, min_jobs: int = 5
    ) -> pd.DataFrame:
        """
        Identify inefficient PI accounts based on VRAM hours.

        Args:
            vram_hours_filter:
                - None: no filtering on vram_hours
                - int | float: select rows where pi_acc_vram_hours == value
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
            min_jobs (int): Minimum number of jobs a PI account must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with PI accounts and their VRAM hours

        Raises:
            ValueError: If the filter is invalid
        """
        if self.pi_accounts_with_efficiency_metrics is None:
            self.calculate_pi_account_efficiency_metrics()
            print(
                "PI accounts with efficiency metrics DataFrame was not available. "
                "Calculated it using the DataFrame of users with efficiency metrics."
            )

        mask = pd.Series(
            [True] * len(self.pi_accounts_with_efficiency_metrics),
            index=self.pi_accounts_with_efficiency_metrics.index,
        )

        if vram_hours_filter is not None:
            try:
                mask &= EfficiencyAnalysis.apply_numeric_filter(
                    self.pi_accounts_with_efficiency_metrics["pi_acc_vram_hours"],
                    vram_hours_filter,
                    {FilterTypeEnum.NUMERIC_SCALAR, FilterTypeEnum.DICTIONARY},
                    filter_name="pi_acc_vram_hours_filter",
                )
            except ValueError as e:
                raise ValueError("Invalid filter for pi_acc_vram_hours.") from e

        col = self.pi_accounts_with_efficiency_metrics["job_count"]
        mask &= col.ge(min_jobs)

        inefficient_pi_accounts = self.pi_accounts_with_efficiency_metrics[mask]

        # Sort by the metric descending (higher is worse)
        inefficient_pi_accounts = inefficient_pi_accounts.sort_values("pi_acc_vram_hours", ascending=False)
        return inefficient_pi_accounts

    def sort_and_filter_records_with_metrics(
        self,
        metrics_df_name_enum: MetricsDataFrameNameEnum,
        sorting_key: str,
        ascending: bool,
        filter_criteria: dict[str, int | float | dict | pd.api.typing.NAType],
    ) -> pd.DataFrame:
        """
        Sort and filter records based on specified criteria.

        Args:
            metrics_df_name_enum (MetricsDataFrameNameEnum): The type of metrics DataFrame to use.
            sorting_key (str): Column name to sort the results by
            ascending (bool): Whether to sort in ascending order
            filter_criteria (dict[str, int | float | dict | pd.NA]): Dictionary of filter criteria to apply.
                Each key is a column name, and the value is the filter to apply to that column. The filter can be:
                    - int | float: select rows where the column value equals the filter value
                    - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
                    - pd.NA: select rows where the column value is pd.NA

        Returns:
            pd.DataFrame: DataFrame with the filtered records sorted by the specified key and their order

        Raises:
            ValueError: If the sorting key is not valid or if ascending is not a boolean value
            ValueError: If the filter criteria are invalid
        """
        if not isinstance(metrics_df_name_enum, MetricsDataFrameNameEnum):
            raise ValueError(
                f"Invalid efficiency metric type: {metrics_df_name_enum}. "
                f"Must be a member of MetricsDataFrameNameEnum."
            )
        metrics_df = getattr(self, metrics_df_name_enum.value)

        if metrics_df is None:
            print(
                f"The {metrics_df_name_enum.value} DataFrame is not available. "
                "Calculating it by running all metrics calculations:"
            )
            self.calculate_all_efficiency_metrics(self.jobs_df)

        if sorting_key not in getattr(self, metrics_df_name_enum.value).columns:
            raise ValueError(f"Sorting key '{sorting_key}' is not a valid column in the DataFrame.")
        if not isinstance(ascending, bool):
            raise ValueError("ascending must be a boolean value.")

        mask = pd.Series(
            [True] * len(getattr(self, metrics_df_name_enum.value)),
            index=getattr(self, metrics_df_name_enum.value).index,
        )

        for column, filter in filter_criteria.items():
            try:
                mask &= EfficiencyAnalysis.apply_numeric_filter(
                    getattr(self, metrics_df_name_enum.value)[column],
                    filter,
                    {FilterTypeEnum.NUMERIC_SCALAR, FilterTypeEnum.DICTIONARY, FilterTypeEnum.PD_NA},
                    filter_name=f"{column}_filter",
                )
            except ValueError as e:
                raise ValueError(f"Invalid filter for {column}.") from e

        filtered_records = getattr(self, metrics_df_name_enum.value)[mask]

        # Sort by the specified key and order

        filtered_records = filtered_records.sort_values(sorting_key, ascending=ascending)

        return filtered_records
