import pandas as pd
from pathlib import Path
from src.preprocess.preprocess import preprocess_data
from src.database import DatabaseConnection
from src.config.constants import DEFAULT_MIN_ELAPSED_SECONDS
import re
from datetime import timedelta, datetime
import warnings


def load_preprocessed_jobs_dataframe_from_duckdb(
    db_path: str | Path,
    table_name: str = "Jobs",
    dates_back: int | None = None,
    custom_query: str = "",
    random_state: pd._typing.RandomState | None = None,
    sample_size: int | None = None,
    include_failed_cancelled_jobs: bool = False,
    include_cpu_only_jobs: bool = False,
    include_custom_qos_jobs: bool = False,
    min_elapsed_seconds: int = DEFAULT_MIN_ELAPSED_SECONDS,
) -> pd.DataFrame:
    """
    Load jobs DataFrame from a DuckDB database and preprocess it.

    Args:
        db_path (str or Path): Path to the DuckDB database.
        table_name (str, optional): Table name to query. Defaults to 'Jobs'.
        dates_back (int, optional): Number of days back to filter jobs based on StartTime.
            Defaults to None. If None, will not filter by startTime.
        custom_query(str, optional): Custom SQL query to execute. Defaults to an empty string.
            If empty, will select all jobs.
        random_state (pd._typing.RandomState, optional): Random state for reproducibility. Defaults to None.
        sample_size (int, optional): Number of rows to sample from the DataFrame. Defaults to None (no sampling).
        include_failed_cancelled_jobs (bool, optional): If True, include jobs with FAILED or CANCELLED status.
            Defaults to False.
        include_cpu_only_jobs (bool, optional): If True, include jobs that do not use GPUs (CPU-only jobs).
            Defaults to False.
        include_custom_qos_jobs (bool, optional): If True, include jobs with custom qos values. Defaults to False.
        min_elapsed_seconds (int, optional): Minimum elapsed time in seconds to filter jobs by elapsed time.
            Defaults to DEFAULT_MIN_ELAPSED_SECONDS.

    Returns:
        pd.DataFrame: DataFrame containing the table data.

    Raises:
        RuntimeError: If the jobs DataFrame cannot be loaded from the database.
    """

    # check if the query contains condition of date_back in the form "StartTime > date"
    def _contain_dates_back_condition(query: str) -> bool:
        pattern = r"(?:WHERE)\s+[^;]*StartTime\s*>=?\s*[^;]+"
        return bool(re.search(pattern, query, re.IGNORECASE))

    if isinstance(db_path, Path):
        db_path = db_path.resolve()
    try:
        db = DatabaseConnection(str(db_path))

        if not custom_query:
            custom_query = f"SELECT * FROM {table_name}"
        if dates_back is not None and not _contain_dates_back_condition(custom_query):
            cutoff = datetime.now() - timedelta(days=dates_back)
            if "where" not in custom_query.lower():
                custom_query += f" WHERE StartTime >= '{cutoff}'"
            else:
                custom_query += f" AND StartTime >= '{cutoff}'"
        elif dates_back is not None and _contain_dates_back_condition(custom_query):
            warnings.warn(
                f"Parameter dates_back = {dates_back} is passed but custom_query already contained conditions for "
                "filtering by dates_back. dates_back condition in custom_query will be used.",
                UserWarning,
                stacklevel=2,
            )

        jobs_df = db.fetch_query(custom_query)
        processed_data = preprocess_data(
            jobs_df,
            min_elapsed_seconds=min_elapsed_seconds,
            include_failed_cancelled_jobs=include_failed_cancelled_jobs,
            include_cpu_only_jobs=include_cpu_only_jobs,
            include_custom_qos_jobs=include_custom_qos_jobs,
        )
        if sample_size is not None:
            processed_data = processed_data.sample(n=sample_size, random_state=random_state)
        return processed_data
    except Exception as e:
        raise RuntimeError(f"Failed to load jobs DataFrame: {e}") from e


def load_preprocessed_jobs_dataframe_from_duckdb_custom_query(
    db_path: str | Path,
    table_name: str = "Jobs",
    custom_query: str | None = None,
    random_state: pd._typing.RandomState | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    # check if the query contains condition of date_back in the form "StartTime > date"
    def _contain_dates_back_condition(query: str) -> bool:
        pattern = r"(?:WHERE)\s+[^;]*StartTime\s*>=?\s*[^;]+"
        return bool(re.search(pattern, query, re.IGNORECASE))

    if isinstance(db_path, Path):
        db_path = db_path.resolve()
    try:
        db = DatabaseConnection(str(db_path))

        if custom_query is None:
            custom_query = f"SELECT * FROM {table_name}"

        jobs_df = db.fetch_query(custom_query)
        # set appropriate argument to avoid filtering out any unexpected records user want
        processed_data = preprocess_data(
            jobs_df,
            min_elapsed_seconds=0,
            include_failed_cancelled_jobs=True,
            include_cpu_only_jobs=True,
            include_custom_qos_jobs=True,
        )
        if sample_size is not None:
            processed_data = processed_data.sample(n=sample_size, random_state=random_state)
        return processed_data
    except Exception as e:
        raise RuntimeError(f"Failed to load jobs DataFrame: {e}") from e
