"""
Tools to analyze efficiency of Jobs based on their VRAM usage.

The aim is to identify potential inefficiencies in GPU usage and notify users or PIs about these issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocess.preprocess import preprocess_data
from src.database import DatabaseConnection
from src.config.constants import DEFAULT_MIN_ELAPSED_SECONDS


def load_jobs_dataframe_from_duckdb(
    db_path: str | Path,
    table_name: str = "Jobs",
    sample_size: int | None = None,
    random_state: pd._typing.RandomState | None = None,
) -> pd.DataFrame:
    """
    Connect to the DuckDB database and return the relevant table as a pandas DataFrame.

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
                
    # Store the variable names as a class-level constant for maintainability
    _efficiency_metric_vars = [
        "jobs_with_efficiency_metrics",
        "users_with_efficiency_metrics",
        "pi_accounts_with_efficiency_metrics",
    ]

    def __init__(
        self,
        db_path: str | Path,
        table_name: str = "Jobs",
        sample_size: int | None = None,
        random_state: pd._typing.RandomState | None = None,
    ) -> None:
        try:
            self.jobs_df = load_jobs_dataframe_from_duckdb(db_path, table_name, sample_size, random_state)
            for var in self._efficiency_metric_vars:
                setattr(self, var, None)
            self.analysis_results: dict | None = None
        except Exception as e:
            raise ValueError(f"Failed to load jobs DataFrame: {e}") from e

    def filter_jobs_for_analysis(
        self,
        vram_constraint_filter: int | list | set | tuple | dict | pd._libs.missing.NAType | None = None,
        gpu_mem_usage_filter: int | list | set | tuple | dict | None = None,
        allocated_vram_filter: int | list | set | tuple | dict | None = None,
        gpu_count_filter: int | list | set | tuple | dict | None = None,
        elapsed_seconds_min: int | float = DEFAULT_MIN_ELAPSED_SECONDS,
    ) -> pd.DataFrame:
        """
        Filter jobs based on VRAM constraints, GPU allocation, and usage criteria.

        Args:
            vram_constraint_filter:
                - None: no filtering on vram_constraint
                - int or float: select rows where vram_constraint == value
                - list/set/tuple: select rows where vram_constraint is in the values provided
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
                - pd.NA or <NA>: select rows where vram_constraint is Nullable Int64 (i.e., pd.NA)
            gpu_mem_usage_filter: the unit is bytes to match the GPUMemUsage column
                - None: no filtering on GPUMemUsage
                - int: select rows where GPUMemUsage == value
                - dict with 'min'/'max' and required 'inclusive' (bool): select rows in the range
            allocated_vram_filter:
                - Same as above; if list/set/tuple: select rows where allocated_vram is in the values provided
            gpu_count_filter:
                - Same as above; if list/set/tuple: select rows where gpu_count is in the values provided
            elapsed_seconds_min (int): Minimum elapsed time in seconds

        Returns:
            DataFrame: Filtered jobs DataFrame based on the specified criteria.

        Raises:
            ValueError: If the filter is invalid.
        """

        mask = pd.Series([True] * len(self.jobs_df), index=self.jobs_df.index)

        # Helper to extract 'inclusive' from dict filter, must be present if dict
        def get_inclusive(filter_val: int | list | set | tuple | dict | pd._libs.missing.NAType | None) -> bool | None:
            if isinstance(filter_val, dict):
                if "inclusive" not in filter_val or not isinstance(filter_val["inclusive"], bool):
                    raise ValueError("If a filter is a dict, it must include an 'inclusive' boolean key.")
                return filter_val["inclusive"]
            return None

        def is_numeric_type(val: object) -> bool:
            """
            Check if the value is a numeric type (int, float, np.integer, np.floating, pd.Int64Dtype, pd.Float64Dtype).
            """
            return pd.api.types.is_integer_dtype(type(val)) or pd.api.types.is_float_dtype(type(val))

        def apply_numeric_filter(
            col: pd.Series,
            filter: int | list | set | tuple | dict | pd._libs.missing.NAType,
            inclusive: bool | None,
            filter_name: str | None = None,
        ) -> pd.Series:
            """
            Helper to apply a numeric filter to a pandas Series.

            Args:
                col (pd.Series): The column to filter.
                value: The filter value (scalar, list/tuple/set, dict with 'min'/'max', or callable).
                inclusive (bool): Whether min/max are inclusive.
            Returns:
                pd.Series: Boolean mask.

            Raises:
                ValueError: If the filter is invalid.
            """
            mask = pd.Series([True] * len(col), index=col.index)
            if filter is not None:
                if filter is pd.NA or (isinstance(filter, float) and np.isnan(filter)):
                    if filter_name not in ["vram_constraint_filter"]:
                        raise ValueError(f"{filter_name} cannot be pd.NA or <NA>.")
                    mask &= col.isna()
                elif isinstance(filter, list | set | tuple):
                    if filter_name in ["gpu_mem_usage_filter"]:
                        raise ValueError(f"{filter_name} cannot be a list, set, or tuple.")
                    if not all(is_numeric_type(val) for val in filter):
                        raise ValueError("All filter values must be integers or floats.")
                    mask &= col.isin(filter)
                elif isinstance(filter, dict):
                    # Check min/max values are int or float if present using pandas type checks
                    for key in ("min", "max"):
                        if key in filter and is_numeric_type(filter[key]) is False:
                            raise ValueError(f"['{key}'] must be an integer or float.")
                    if "min" in filter:
                        mask &= col.ge(filter["min"]) if inclusive else col.gt(filter["min"])
                    if "max" in filter:
                        mask &= col.le(filter["max"]) if inclusive else col.lt(filter["max"])
                else:
                    # Only allow numeric types
                    if is_numeric_type(filter):
                        if isinstance(filter, np.number):
                            # Convert numpy number to native Python type
                            filter = filter.item()
                            mask &= col.eq(filter)
                    else:
                        raise ValueError("Filter must be a numeric value if not one of the other types.")
            return mask

        # vram_constraint
        if vram_constraint_filter is not None:
            try:
                vram_constraint_inclusive = get_inclusive(vram_constraint_filter)
                mask &= apply_numeric_filter(
                    self.jobs_df["vram_constraint"],
                    vram_constraint_filter,
                    vram_constraint_inclusive,
                    "vram_constraint_filter",
                )
            except ValueError as e:
                raise ValueError("Invalid vram_constraint_filter.") from e

        # GPU memory usage filter
        if gpu_mem_usage_filter is not None:
            try:
                gpu_mem_usage_inclusive = get_inclusive(gpu_mem_usage_filter)
                mask &= apply_numeric_filter(
                    self.jobs_df["GPUMemUsage"], gpu_mem_usage_filter, gpu_mem_usage_inclusive, "gpu_mem_usage_filter"
                )
            except ValueError as e:
                raise ValueError("Invalid GPU memory usage filter.") from e

        # Allocated VRAM filter
        if allocated_vram_filter is not None:
            try:
                allocated_vram_inclusive = get_inclusive(allocated_vram_filter)
                mask &= apply_numeric_filter(
                    self.jobs_df["allocated_vram"], allocated_vram_filter, allocated_vram_inclusive
                )
            except ValueError as e:
                raise ValueError("Invalid allocated VRAM filter.") from e

        # GPU count filter
        if gpu_count_filter is not None:
            try:
                gpu_count_inclusive = get_inclusive(gpu_count_filter)
                mask &= apply_numeric_filter(self.jobs_df["GPUs"], gpu_count_filter, gpu_count_inclusive)
            except ValueError as e:
                raise ValueError("Invalid GPU count filter.") from e

        # Filter by elapsed time
        if not is_numeric_type(elapsed_seconds_min) or elapsed_seconds_min < 0:
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
        # TODO (Arda): Update the implementation of alloc_vram_efficiency_score
        # Set the score to -inf where alloc_vram_efficiency is zero to avoid divide by zero/log of zero
        alloc_vram_eff = filtered_jobs["alloc_vram_efficiency"]
        filtered_jobs["alloc_vram_efficiency_score"] = (
            np.log(alloc_vram_eff.where(alloc_vram_eff > 0)) * filtered_jobs["job_hours"]
        ).where(alloc_vram_eff > 0, -np.inf)

        # Calculate weighted vram efficiency per job, normalized by total job_hours for that specific PI
        pi_gpu_hours = filtered_jobs.groupby("Account", observed=True)["job_hours"].transform("sum")
        filtered_jobs.loc[:, "pi_weighted_vram_efficiency"] = (
            filtered_jobs["alloc_vram_efficiency"] * filtered_jobs["job_hours"]
        ) / pi_gpu_hours

        # Add CPU memory metrics if available
        if "CPUMemUsage" in self.jobs_df.columns:
            filtered_jobs.loc[:, "used_cpu_gib"] = filtered_jobs["CPUMemUsage"] / (2**30)

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

        users_w_efficiency_metrics = (
            self.jobs_with_efficiency_metrics.groupby("User", observed=False)
            .agg(
                job_count=("JobID", "count"),
                user_job_hours=("job_hours", "sum"),
                pi_account=("Account", "first"),
            )
            .reset_index()
        )

        self.jobs_with_efficiency_metrics.loc[:, "weighted_alloc_vram_efficiency"] = (
            self.jobs_with_efficiency_metrics["alloc_vram_efficiency"]
            * self.jobs_with_efficiency_metrics["job_hours"]
            / user_job_hours_per_job
        )
        users_w_efficiency_metrics.loc[:, "expected_value_alloc_vram_efficiency"] = (
            self.jobs_with_efficiency_metrics.groupby("User", observed=True)["weighted_alloc_vram_efficiency"]
            .sum()
            .to_numpy()
        )

        self.jobs_with_efficiency_metrics.loc[:, "weighted_gpu_count"] = (
            self.jobs_with_efficiency_metrics["gpu_count"]
            * self.jobs_with_efficiency_metrics["job_hours"]
            / user_job_hours_per_job
        )
        users_w_efficiency_metrics.loc[:, "expected_value_gpu_count"] = (
            self.jobs_with_efficiency_metrics.groupby("User", observed=True)["weighted_gpu_count"].sum().to_numpy()
        )

        # Calculate metric representing the total amount of GPU memory resources a user has been allocated over time.
        # It answers the question: “How much VRAM, and for how long, did this user occupy?”
        users_w_efficiency_metrics.loc[:, "vram_hours"] = (
            (self.jobs_with_efficiency_metrics["allocated_vram"] * self.jobs_with_efficiency_metrics["job_hours"])
            .groupby(self.jobs_with_efficiency_metrics["User"], observed=True)
            .sum()
            .to_numpy()
        )

        self.jobs_with_efficiency_metrics = self.jobs_with_efficiency_metrics.drop(
            columns=["weighted_alloc_vram_efficiency", "weighted_gpu_count"]
        )

        self.users_with_efficiency_metrics = users_w_efficiency_metrics
        return self.users_with_efficiency_metrics

    def find_inefficient_users_by_alloc_vram_efficiency(
        self,
        efficiency_threshold: float = 0.3,
        min_jobs: int = 5
    ) -> pd.DataFrame:
        """
        Identify users with low expected allocated VRAM efficiency across their jobs compared to others

        Args:
            efficiency_threshold (float): Maximum threshold for expected allocated VRAM efficiency
            min_jobs (int): Minimum number of jobs a user must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with users and their average VRAM efficiency
        """
        if self.users_with_efficiency_metrics is None:
            raise ValueError(
                "Users with efficiency metrics DataFrame is not available. "
                "Please run calculate_user_efficiency_metrics first."
            )

        mask = pd.Series(
            [True] * len(self.users_with_efficiency_metrics), index=self.users_with_efficiency_metrics.index
        )

        col = self.users_with_efficiency_metrics["expected_value_alloc_vram_efficiency"]
        mask &= col.le(efficiency_threshold)

        col = self.users_with_efficiency_metrics["job_count"]
        mask &= col.ge(min_jobs)

        inefficient_users = self.users_with_efficiency_metrics[mask]

        # Sort by the metric ascending (lower is worse)
        inefficient_users = inefficient_users.sort_values("expected_value_alloc_vram_efficiency", ascending=True)
        return inefficient_users

    def find_inefficient_users_by_vram_hours(
        self,
        vram_hours_threshold: float = 200,
        min_jobs: int = 5
    ) -> pd.DataFrame:
        """
        Identify users with high VRAM-hours across their jobs compared to others.

        Args:
            vram_hours_threshold (float): Minimum threshold for VRAM hours to consider a user as inefficient
            min_jobs (int): Minimum number of jobs a user must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with users and their total VRAM hours
        """
        if self.users_with_efficiency_metrics is None:
            raise ValueError(
                "Users with efficiency metrics DataFrame is not available. "
                "Please run calculate_user_efficiency_metrics first."
            )

        mask = pd.Series(
            [True] * len(self.users_with_efficiency_metrics), index=self.users_with_efficiency_metrics.index
        )

        col = self.users_with_efficiency_metrics["vram_hours"]
        mask &= col.ge(vram_hours_threshold)

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
            ValueError: If any of the calculations fail.
        """
        try:
            self.calculate_job_efficiency_metrics(filtered_jobs)
            self.calculate_user_efficiency_metrics()
            self.calculate_pi_account_efficiency_metrics()
            return {var: getattr(self, var) for var in self._efficiency_metric_vars}

        except Exception as e:
            raise ValueError(f"Failed to calculate all efficiency metrics: {e}") from e

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
            .sum()
            .to_numpy()
        )

        self.users_with_efficiency_metrics.loc[:, "weighted_ev_gpu_count"] = (
            self.users_with_efficiency_metrics["expected_value_gpu_count"]
            * self.users_with_efficiency_metrics["vram_hours"]
            / pi_acc_vram_hours
        )
        pi_efficiency_metrics.loc[:, "expected_value_gpu_count"] = (
            self.users_with_efficiency_metrics.groupby("pi_account", observed=True)["weighted_ev_gpu_count"]
            .sum()
            .to_numpy()
        )

        self.users_with_efficiency_metrics = self.users_with_efficiency_metrics.drop(
            columns=["weighted_ev_alloc_vram_efficiency"]
        )

        self.pi_accounts_with_efficiency_metrics = pi_efficiency_metrics
        return self.pi_accounts_with_efficiency_metrics

    def find_inefficient_pis_by_vram_hours(self, vram_hours_threshold: float = 200, min_jobs: int = 5) -> pd.DataFrame:
        if self.pi_accounts_with_efficiency_metrics is None:
            raise ValueError(
                "PI accounts with efficiency metrics DataFrame is not available. "
                "Please run calculate_pi_account_efficiency_metrics first."
            )

        mask = pd.Series(
            [True] * len(self.pi_accounts_with_efficiency_metrics),
            index=self.pi_accounts_with_efficiency_metrics.index,
        )

        col = self.pi_accounts_with_efficiency_metrics["pi_acc_vram_hours"]
        mask &= col.ge(vram_hours_threshold)

        col = self.pi_accounts_with_efficiency_metrics["job_count"]
        mask &= col.ge(min_jobs)

        inefficient_pi_accounts = self.pi_accounts_with_efficiency_metrics[mask]

        # Sort by the metric descending (higher is worse)
        inefficient_pi_accounts = inefficient_pi_accounts.sort_values("pi_acc_vram_hours", ascending=False)
        return inefficient_pi_accounts
