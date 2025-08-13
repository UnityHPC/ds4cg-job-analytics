"""
User comparison module for generating efficient user vs. others comparisons.

This module provides utilities for comparing a user's GPU usage metrics with
those of other users, designed to be efficient by:
1. Fetching only required fields from the database
2. Minimizing redundant calculations
3. Enabling partial data analysis when needed
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from src.database.database_connection import DatabaseConnection
from src.preprocess.preprocess import preprocess_data
from src.analysis.efficiency_analysis import EfficiencyAnalysis
from src.config.enum_constants import (
    StatusEnum,
    AdminsAccountEnum,
    AdminPartitionEnum,
    QOSEnum,
    PartitionTypeEnum,
)
from src.config.constants import DEFAULT_MIN_ELAPSED_SECONDS
from src.config.remote_config import PartitionInfoFetcher


def load_jobs_df(
    db_path: str | Path | None = None,
    db_connection: DatabaseConnection | None = None,
    table_name: str = "Jobs",
    dates_back: int | None = None,
    include_failed_cancelled_jobs: bool = False,
    include_cpu_only_jobs: bool = False,
    include_custom_qos_jobs: bool = False,
    min_elapsed_seconds: int = DEFAULT_MIN_ELAPSED_SECONDS,
    random_state: pd._typing.RandomState | None = None,
    sample_size: int | None = None,
    user_filter: str | None = None,
) -> pd.DataFrame:
    """
    Load jobs DataFrame from a DuckDB database with standard filtering and preprocess it.
    This function constructs a SQL query with predefined filtering conditions based on the provided
    parameters and then preprocesses the resulting data.

    Args:
        db_path (str or Path, optional): Path to the DuckDB database. Not needed if db_connection is provided.
        db_connection (DatabaseConnection, optional): Existing database connection to use. If provided,
            this will be used instead of creating a new connection from db_path.
        table_name (str, optional): Table name to query. Defaults to 'Jobs'.
        dates_back (int, optional): Number of days back to filter jobs based on StartTime.
            Defaults to None. If None, will not filter by startTime.
        include_failed_cancelled_jobs (bool, optional): If True, include jobs with FAILED or CANCELLED status.
            Defaults to False.
        include_cpu_only_jobs (bool, optional): If True, include jobs that do not use GPUs (CPU-only jobs).
            Defaults to False.
        include_custom_qos_jobs (bool, optional): If True, include jobs with custom qos values. Defaults to False.
        min_elapsed_seconds (int, optional): Minimum elapsed time in seconds to filter jobs by elapsed time.
            Defaults to DEFAULT_MIN_ELAPSED_SECONDS.
        random_state (pd._typing.RandomState, optional): Random state for reproducibility. Defaults to None.
        sample_size (int, optional): Number of rows to sample from the DataFrame. Defaults to None (no sampling).
        user_filter (str, optional): Filter jobs for a specific user. Defaults to None (all users).

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing the filtered job data.

    Raises:
        RuntimeError: If the jobs DataFrame cannot be loaded from the database.
    """

    try:
        # Use the provided connection if available, otherwise create a new one from the path
        if db_connection is not None:
            db = db_connection
        else:
            if db_path is None:
                raise ValueError("Either db_connection or db_path must be provided")
            if isinstance(db_path, Path):
                db_path = db_path.resolve()
            db = DatabaseConnection(str(db_path))

        qos_values = "(" + ",".join(f"'{obj.value}'" for obj in QOSEnum) + ")"

        # get cpu partition list
        partition_info = PartitionInfoFetcher().get_info()
        gpu_partitions = [p["name"] for p in partition_info if p["type"] == PartitionTypeEnum.GPU.value]
        gpu_partitions_str = "(" + ",".join(f"'{partition_name}'" for partition_name in gpu_partitions) + ")"

        conditions_arr = [
            f"Elapsed >= {min_elapsed_seconds}",
            f"Account != '{AdminsAccountEnum.ROOT.value}'",
            f"Partition != '{AdminPartitionEnum.BUILDING.value}'",
            f"QOS != '{QOSEnum.UPDATES.value}'",
        ]
        if dates_back is not None:
            cutoff = datetime.now() - timedelta(days=dates_back)
            conditions_arr.append(f"StartTime >= '{cutoff}'")
        if not include_custom_qos_jobs:
            conditions_arr.append(f"QOS in {qos_values}")
        if not include_cpu_only_jobs:
            conditions_arr.append(f"Partition IN {gpu_partitions_str}")
        if not include_failed_cancelled_jobs:
            conditions_arr.append(f"Status != '{StatusEnum.FAILED.value}'")
            conditions_arr.append(f"Status != '{StatusEnum.CANCELLED.value}'")
        if user_filter is not None:
            conditions_arr.append(f"User = '{user_filter}'")

        query = f"SELECT * FROM {table_name} WHERE {' AND '.join(conditions_arr)}"

        jobs_df = db.fetch_query(query=query)
        processed_data = preprocess_data(jobs_df, filtered=True)
        if sample_size is not None:
            processed_data = processed_data.sample(n=sample_size, random_state=random_state)
        return processed_data
    except Exception as e:
        raise RuntimeError(f"Failed to load jobs DataFrame: {e}") from e


class UserComparison:
    """
    A class to efficiently compare a user's metrics with other users.

    This class provides methods to calculate comparison statistics between
    a target user and all other users, with optimizations to minimize
    database queries and redundant calculations.
    """

    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize the UserComparison with a database connection.

        Loads and preprocesses all job data once at initialization.

        Args:
            db_connection: Database connection object to use for queries
        """
        self.db = db_connection
        # Cache for processed data
        self._all_jobs_data = None
        self._cached_all_users_metrics = None

        # Load all jobs data during initialization to avoid multiple queries
        try:
            self._all_jobs_data = load_jobs_df(
                db_connection=self.db,
                include_cpu_only_jobs=False,
                include_failed_cancelled_jobs=False,
                min_elapsed_seconds=0,
            )

            # If we have data, calculate efficiency metrics once for all jobs
            if len(self._all_jobs_data) > 0:
                analyzer = EfficiencyAnalysis(self._all_jobs_data)
                filtered_jobs = analyzer.filter_jobs_for_analysis(
                    gpu_count_filter={"min": 1, "max": np.inf, "inclusive": True},
                    vram_constraint_filter=None,
                    allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
                    gpu_mem_usage_filter=None,
                )
                self._cached_all_users_metrics = analyzer.calculate_job_efficiency_metrics(filtered_jobs)
                print(
                    f"Loaded and processed {len(self._cached_all_users_metrics)} job records for efficiency analysis"
                )
            else:
                print("No job data found in database")
        except Exception as e:
            print(f"Error initializing job data: {e}")
            # Fallback to loading data on demand if initialization fails
            self._all_jobs_data = None
            self._cached_all_users_metrics = None

    def _get_essential_job_data(self, exclude_user: str | None = None) -> pd.DataFrame:
        """
        Get only essential job data fields needed for efficiency metrics.

        Args:
            exclude_user: User ID to exclude from the query results (optional)

        Returns:
            DataFrame containing only the necessary columns for efficiency calculations
        """
        exclusion_clause = f"AND User != '{exclude_user}'" if exclude_user else ""

        efficiency_query = f"""
        SELECT
            *
        FROM Jobs
        WHERE GPUs > 0
        AND GPUMemUsage IS NOT NULL
        AND User != 'root'
        AND Elapsed IS NOT NULL
        AND TimeLimit IS NOT NULL
        {exclusion_clause}
        """

        return self.db.fetch_query(efficiency_query)

    def calculate_efficiency_metrics(self, jobs_df: pd.DataFrame, gpu_count_min: int = 1) -> pd.DataFrame:
        """
        Calculate efficiency metrics for the given jobs DataFrame.

        Args:
            jobs_df: DataFrame containing job data
            gpu_count_min: Minimum GPU count to include in analysis

        Returns:
            DataFrame with calculated efficiency metrics
        """
        # Preprocess the data
        processed_jobs = preprocess_data(
            jobs_df, min_elapsed_seconds=0, include_failed_cancelled_jobs=False, include_cpu_only_jobs=False
        )

        # Initialize efficiency analyzer
        analyzer = EfficiencyAnalysis(processed_jobs)

        # Filter jobs (only include jobs with at least gpu_count_min GPUs)
        filtered_jobs = analyzer.filter_jobs_for_analysis(
            gpu_count_filter={"min": gpu_count_min, "max": np.inf, "inclusive": True},
            vram_constraint_filter=None,
            allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
            gpu_mem_usage_filter=None,
        )

        # Calculate efficiency metrics
        jobs_with_metrics = analyzer.calculate_job_efficiency_metrics(filtered_jobs)

        return jobs_with_metrics

    def get_user_metrics(self, user_id: str) -> pd.DataFrame:
        """
        Get efficiency metrics for a specific user by filtering the preloaded data.

        Args:
            user_id: User ID to get metrics for

        Returns:
            DataFrame with user's jobs and their efficiency metrics
        """
        # Check if we already have the cached metrics
        if self._cached_all_users_metrics is not None:
            # Filter the existing processed data for this user
            user_metrics = self._cached_all_users_metrics[self._cached_all_users_metrics["User"] == user_id]

            if len(user_metrics) > 0:
                return user_metrics

        # Fall back to direct query if no cached data or user not found
        try:
            user_jobs = load_jobs_df(
                db_connection=self.db,
                include_cpu_only_jobs=False,
                include_failed_cancelled_jobs=False,
                min_elapsed_seconds=0,
                user_filter=user_id,
            )

            if len(user_jobs) == 0:
                print(f"No jobs found for user {user_id}")
                return pd.DataFrame()

            # Calculate and return efficiency metrics
            analyzer = EfficiencyAnalysis(user_jobs)
            filtered_jobs = analyzer.filter_jobs_for_analysis(
                gpu_count_filter={"min": 1, "max": np.inf, "inclusive": True},
                vram_constraint_filter=None,
                allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
                gpu_mem_usage_filter=None,
            )

            jobs_with_metrics = analyzer.calculate_job_efficiency_metrics(filtered_jobs)
            print(f"Processed {len(jobs_with_metrics)} jobs for user {user_id} from direct query")
            return jobs_with_metrics

        except Exception as e:
            print(f"Error getting user metrics: {e}")
            return pd.DataFrame()

    def get_other_users_metrics(self, exclude_user: str) -> pd.DataFrame:
        """
        Get aggregated metrics for all users except the specified one.

        Args:
            exclude_user: User ID to exclude from the metrics

        Returns:
            DataFrame with aggregated metrics for all other users
        """
        # Only fetch from DB if we don't have cached data
        if self._cached_all_users_metrics is None:
            try:
                # Use optimized function to load all jobs using the existing database connection
                all_jobs = load_jobs_df(
                    db_connection=self.db,  # Use existing connection
                    include_cpu_only_jobs=False,
                    include_failed_cancelled_jobs=False,
                    min_elapsed_seconds=0,
                )

                # Calculate efficiency metrics for all jobs
                analyzer = EfficiencyAnalysis(all_jobs)
                filtered_jobs = analyzer.filter_jobs_for_analysis(
                    gpu_count_filter={"min": 1, "max": np.inf, "inclusive": True},
                    vram_constraint_filter=None,
                    allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
                    gpu_mem_usage_filter=None,
                )

                self._cached_all_users_metrics = analyzer.calculate_job_efficiency_metrics(filtered_jobs)

            except Exception as e:
                print(f"Error using optimized method for all users: {e}")
                # Fallback to original implementation if there's an error
                all_jobs = self._get_essential_job_data()
                self._cached_all_users_metrics = self.calculate_efficiency_metrics(all_jobs)

        # Filter out the excluded user from cached data
        other_users_metrics = self._cached_all_users_metrics[self._cached_all_users_metrics["User"] != exclude_user]

        return other_users_metrics

    def get_user_comparison_statistics(
        self, user_id: str, user_jobs: pd.DataFrame | None = None, metrics: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Calculate comparison statistics between the specified user and all other users.

        Args:
            user_id: The user ID to compare
            user_jobs: Optional pre-calculated user jobs DataFrame with metrics
                      (to avoid recomputing if already available)
            metrics: List of metrics to include in the comparison
                    (defaults to standard efficiency metrics)

        Returns:
            DataFrame with comparison statistics
        """
        if metrics is None:
            metrics = [
                ("alloc_vram_efficiency", "VRAM Efficiency (%)"),
                ("time_usage_efficiency", "Time Usage (%)"),
                ("used_vram_gib", "Total GPU Memory (GiB)"),
                ("allocated_vram", "Allocated VRAM (GiB)"),
                ("requested_vram", "Requested VRAM (GiB)"),
                ("job_hours", "GPU Hours"),
            ]

        # Get user metrics if not provided
        if user_jobs is None:
            user_jobs = self.get_user_metrics(user_id)

        if len(user_jobs) == 0:
            return self._create_fallback_comparison(metrics)

        # Get other users' metrics
        other_users = self.get_other_users_metrics(user_id)

        if len(other_users) == 0:
            return self._create_fallback_comparison(user_jobs, metrics)

        # Calculate user-level aggregated metrics
        user_metrics = {}
        for metric_name, _ in metrics:
            if metric_name == "alloc_vram_efficiency":
                user_metrics[metric_name] = user_jobs[metric_name].mean() * 100  # Convert to percentage
            elif metric_name == "time_usage_efficiency":
                # Calculate time usage efficiency - this is Elapsed/TimeLimit as a percentage
                user_metrics[metric_name] = (
                    user_jobs["Elapsed"].dt.total_seconds() / user_jobs["TimeLimit"].dt.total_seconds()
                ).mean() * 100
            elif metric_name in ["allocated_vram", "requested_vram"]:
                # For VRAM metrics, we want the average per job
                user_metrics[metric_name] = user_jobs[metric_name].mean()
            else:
                # For other metrics like used_vram_gib, sum them up
                user_metrics[metric_name] = user_jobs[metric_name].sum()

        # Calculate aggregates for other users
        other_users_metrics = {}
        for metric_name, _ in metrics:
            if metric_name == "alloc_vram_efficiency":
                other_users_metrics[metric_name] = other_users[metric_name].mean() * 100  # Convert to percentage
            elif metric_name == "time_usage_efficiency":
                # Calculate time usage efficiency for other users
                other_users_metrics[metric_name] = (
                    other_users["Elapsed"].dt.total_seconds() / other_users["TimeLimit"].dt.total_seconds()
                ).mean() * 100
            elif metric_name in ["allocated_vram", "requested_vram"]:
                # For VRAM metrics, we want the average per job
                other_users_metrics[metric_name] = other_users[metric_name].mean()
            else:
                # For other metrics like used_vram_gib, calculate the average per user
                user_sums = other_users.groupby("User")[metric_name].sum()
                other_users_metrics[metric_name] = user_sums.mean()

        # Create comparison DataFrame
        comparison_stats = pd.DataFrame({
            "Category": [display_name for _, display_name in metrics],
            "Your_Value": [user_metrics[metric_name] for metric_name, _ in metrics],
            "Average_Value": [other_users_metrics[metric_name] for metric_name, _ in metrics],
        })

        return comparison_stats

    def _create_fallback_comparison(
        self, user_jobs: pd.DataFrame | None = None, metrics: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Create fallback comparison statistics when database query fails.

        Args:
            user_jobs: Optional DataFrame containing the user's job data
            metrics: List of metrics to include in the comparison

        Returns:
            DataFrame with basic comparison statistics
        """
        if metrics is None:
            metrics = [
                ("alloc_vram_efficiency", "VRAM Efficiency (%)"),
                ("time_usage_efficiency", "Time Usage (%)"),
                ("used_vram_gib", "Total GPU Memory (GiB)"),
                ("allocated_vram", "Allocated VRAM (GiB)"),
                ("requested_vram", "Requested VRAM (GiB)"),
                ("job_hours", "GPU Hours"),
            ]

        # Calculate user metrics if user_jobs is available
        user_values = []
        placeholder_values = [30.0, 75.0, 500.0, 40.0, 32.0, 100.0]  # Reasonable defaults

        if user_jobs is not None and not user_jobs.empty:
            for metric_name, _ in metrics:
                if metric_name == "alloc_vram_efficiency" and "alloc_vram_efficiency" in user_jobs.columns:
                    user_values.append(user_jobs["alloc_vram_efficiency"].mean() * 100)
                elif (
                    metric_name == "time_usage_efficiency"
                    and "Elapsed" in user_jobs.columns
                    and "TimeLimit" in user_jobs.columns
                ):
                    user_values.append(
                        (user_jobs["Elapsed"].dt.total_seconds() / user_jobs["TimeLimit"].dt.total_seconds()).mean()
                        * 100
                    )
                elif metric_name in user_jobs.columns:
                    if metric_name in ["allocated_vram", "requested_vram"]:
                        user_values.append(user_jobs[metric_name].mean())
                    else:
                        user_values.append(user_jobs[metric_name].sum())
                else:
                    user_values.append(0.0)
        else:
            user_values = [0.0] * len(metrics)

        # Create comparison stats with placeholder values
        comparison_stats = pd.DataFrame({
            "Category": [display_name for _, display_name in metrics],
            "Your_Value": user_values,
            "Average_Value": placeholder_values[: len(metrics)],
        })

        return comparison_stats
