import pandas as pd
from pathlib import Path
from src.preprocess.preprocess import preprocess_data
from src.database import DatabaseConnection
from src.config.constants import DEFAULT_MIN_ELAPSED_SECONDS
from src.config.enum_constants import QOSEnum, AdminPartitionEnum, AdminsAccountEnum, StatusEnum, PartitionTypeEnum
from src.config.remote_config import PartitionInfoFetcher
from datetime import datetime, timedelta


def load_and_preprocessed_jobs(
    db_path: str | Path,
    table_name: str = "Jobs",
    dates_back: int | None = None,
    include_failed_cancelled_jobs: bool = False,
    include_cpu_only_jobs: bool = False,
    include_custom_qos_jobs: bool = False,
    min_elapsed_seconds: int = DEFAULT_MIN_ELAPSED_SECONDS,
    random_state: pd._typing.RandomState | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Load jobs DataFrame from a DuckDB database with standard filtering and preprocess it.

    This function constructs a SQL query with predefined filtering conditions based on the provided
    parameters and then preprocesses the resulting data.

    Args:
        db_path (str or Path): Path to the DuckDB database.
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

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing the filtered job data.

    Raises:
        RuntimeError: If the jobs DataFrame cannot be loaded from the database.
    """

    # check if the query contains condition of date_back in the form "StartTime > date"

    if isinstance(db_path, Path):
        db_path = db_path.resolve()
    try:
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

        query = f"SELECT * FROM {table_name} WHERE {' AND '.join(conditions_arr)}"

        jobs_df = db.fetch_query(query=query)
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


def load_and_preprocessed_jobs_custom_query(
    db_path: str | Path,
    table_name: str = "Jobs",
    custom_query: str | None = None,
    random_state: pd._typing.RandomState | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Load jobs DataFrame from a DuckDB database using a custom SQL query and preprocess it.

    This function allows for complete control over the SQL query used to fetch data from the database.
    The preprocessing is done with permissive settings to avoid filtering out any records that the
    user specifically requested through their custom query.

    Args:
        db_path (str or Path): Path to the DuckDB database.
        table_name (str, optional): Table name to use in default query if custom_query is None. Defaults to 'Jobs'.
        custom_query (str, optional): Custom SQL query to execute. If None, defaults to "SELECT * FROM {table_name}".
        random_state (pd._typing.RandomState, optional): Random state for reproducibility. Defaults to None.
        sample_size (int, optional): Number of rows to sample from the DataFrame. Defaults to None (no sampling).

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing the data returned by the custom query.

    Notes:
        The preprocessing is performed with the following permissive settings:
        - min_elapsed_seconds=0 (no minimum elapsed time filtering)
        - include_failed_cancelled_jobs=True (include all job statuses)
        - include_cpu_only_jobs=True (include CPU-only jobs)
        - include_custom_qos_jobs=True (include custom QOS jobs)

        This ensures that the function doesn't inadvertently filter out records that the user
        explicitly requested through their custom query.

    Raises:
        RuntimeError: If the jobs DataFrame cannot be loaded from the database.
    """
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
