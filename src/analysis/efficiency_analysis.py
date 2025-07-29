"""
Tools to analyze efficiency of Jobs based on their VRAM usage.

The aim is to identify potential inefficiencies in GPU usage and notify users or PIs about these issues.
"""

from typing import cast
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from collections.abc import Callable

from src.preprocess.preprocess import preprocess_data
from src.database import DatabaseConnection
from src.config.constants import DEFAULT_MIN_ELAPSED_SECONDS
from src.config.enum_constants import FilterTypeEnum, MetricsDataFrameNameEnum, TimeUnitEnum


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
                    raise ValueError(
                        f"{filter_name} cannot be pd.NA or <NA>. "
                        f"Permissible filter types are {permissible_filter_types}."
                    )
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

        # rename GPUs to gpu_count for clarity
        filtered_jobs = filtered_jobs.rename(columns={"GPUs": "gpu_count"})

        # Calculate job efficiency metrics
        filtered_jobs.loc[:, "job_hours"] = (
            filtered_jobs["Elapsed"].dt.total_seconds() * filtered_jobs["gpu_count"] / 3600
        )
        filtered_jobs.loc[:, "vram_hours"] = filtered_jobs["allocated_vram"] * filtered_jobs["job_hours"]
        filtered_jobs.loc[:, "used_vram_gib"] = filtered_jobs["GPUMemUsage"] / (2**30)
        # Compute alloc_vram_efficiency, a float in the range [0, 1].
        filtered_jobs.loc[:, "alloc_vram_efficiency"] = (
            filtered_jobs["used_vram_gib"] / filtered_jobs["allocated_vram"]
        )

        # Compute vram_constraint_efficiency, a nullable float in the range [0, 1]. Set to NA if vram_constraint is NA
        filtered_jobs.loc[:, "vram_constraint_efficiency"] = (
            filtered_jobs["used_vram_gib"] / filtered_jobs["vram_constraint"]
        )

        # Calculate job allocated VRAM efficiency score
        # This is a log-transformed score that penalizes low efficiency and longer vram_hours
        alloc_vram_eff = filtered_jobs["alloc_vram_efficiency"]
        filtered_jobs.loc[:, "alloc_vram_efficiency_score"] = (
            np.log(alloc_vram_eff.where(alloc_vram_eff > 0)) * filtered_jobs["vram_hours"]
        ).where(alloc_vram_eff > 0, -np.inf)

        # Calculate vram_constraint_efficiency score
        vram_constraint_eff = filtered_jobs["vram_constraint_efficiency"]
        # Avoid log(0) and propagate pd.NA: if NA, score is NA; if 0, score is -np.inf
        score = pd.Series(pd.NA, index=filtered_jobs.index, dtype=pd.Float64Dtype())
        mask_valid = vram_constraint_eff.notna() & (vram_constraint_eff > 0)
        mask_zero = vram_constraint_eff.notna() & (vram_constraint_eff == 0)
        score[mask_valid] = np.log(vram_constraint_eff[mask_valid]) * filtered_jobs.loc[mask_valid, "vram_hours"]
        score[mask_zero] = -np.inf
        filtered_jobs.loc[:, "vram_constraint_efficiency_score"] = score

        # Add CPU memory metrics if available
        if "CPUMemUsage" in self.jobs_df.columns and "Memory" in self.jobs_df.columns:
            filtered_jobs.loc[:, "used_cpu_mem_gib"] = filtered_jobs["CPUMemUsage"] / (2**30)
            filtered_jobs.loc[:, "allocated_cpu_mem_gib"] = filtered_jobs["Memory"] / (2**10)  # Memory is in MiB
            filtered_jobs.loc[:, "cpu_mem_efficiency"] = (
                filtered_jobs["used_cpu_mem_gib"] / filtered_jobs["allocated_cpu_mem_gib"]
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

        # Compute user_job_hours_per_job once and reuse for both metrics
        user_job_hours_per_job = self.jobs_with_efficiency_metrics.groupby("User", observed=True)[
            "job_hours"
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
                avg_alloc_vram_efficiency_score=("alloc_vram_efficiency_score", avg_non_inf),
                avg_vram_constraint_efficiency_score=("vram_constraint_efficiency_score", avg_non_inf),
            )
            .reset_index()
        )

        self.jobs_with_efficiency_metrics.loc[:, "weighted_alloc_vram_efficiency"] = (
            self.jobs_with_efficiency_metrics["alloc_vram_efficiency"]
            * self.jobs_with_efficiency_metrics["vram_hours"]
            / user_job_hours_per_job
        )

        users_w_efficiency_metrics.loc[:, "expected_value_alloc_vram_efficiency"] = (
            self.jobs_with_efficiency_metrics.groupby("User", observed=True)["weighted_alloc_vram_efficiency"]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        self.jobs_with_efficiency_metrics.loc[:, "weighted_vram_constraint_efficiency"] = (
            self.jobs_with_efficiency_metrics["vram_constraint_efficiency"]
            * self.jobs_with_efficiency_metrics["vram_hours"]
            / user_job_hours_per_job
        ).astype(pd.Float64Dtype())

        users_w_efficiency_metrics.loc[:, "expected_value_vram_constraint_efficiency"] = (
            self.jobs_with_efficiency_metrics.groupby("User", observed=True)["weighted_vram_constraint_efficiency"]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        self.jobs_w_efficiency_metrics.loc[:, "weighted_gpu_count"] = (
            self.jobs_w_efficiency_metrics["gpu_count"]
            * self.jobs_w_efficiency_metrics["vram_hours"]
            / user_job_hours_per_job
        )
        users_w_efficiency_metrics.loc[:, "expected_value_gpu_count"] = (
            self.jobs_with_efficiency_metrics.groupby("User", observed=True)["weighted_gpu_count"]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        # Calculate metric representing the total amount of GPU memory resources a user has been allocated over time.
        # It answers the question: “How much VRAM, and for how long, did this user occupy?”
        users_w_efficiency_metrics.loc[:, "vram_hours"] = (
            (self.jobs_with_efficiency_metrics["allocated_vram"] * self.jobs_with_efficiency_metrics["job_hours"])
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
            raise ValueError(f"Invalid time unit {time_unit}. Choose 'Months', 'Weeks', or 'Days'.")

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
            return all_time_groups[first_non_zero_idx: last_non_zero_idx + 1]
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
            if user_df is None or user_df.empty:
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
                if user_df is None or user_df.empty:
                    continue
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

