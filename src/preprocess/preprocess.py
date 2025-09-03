import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from pandas.api.typing import NAType

from ..config.constants import (
    DEFAULT_MIN_ELAPSED_SECONDS,
    ATTRIBUTE_CATEGORIES,
)
from ..config.enum_constants import (
    StatusEnum,
    AdminsAccountEnum,
    AdminPartitionEnum,
    QOSEnum,
    PartitionTypeEnum,
    PreprocessingErrorTypeEnum,
    OptionalColumnsEnum,
    RequiredColumnsEnum,
    ExcludedColumnsEnum,
)
from .allocated_vram import _get_approx_allocated_vram
from .constraints import _get_vram_constraint, _get_partition_constraint, _get_requested_vram
from ..config.remote_config import PartitionInfoFetcher
from ..config.paths import PREPROCESSING_ERRORS_LOG_FILE
from .errors import JobPreprocessingError

processing_error_logs: list = []
error_indices: set = set()


def _validate_gpu_type(
    gpu_type_value: NAType | np.ndarray | list | dict, include_cpu_only_jobs: bool
) -> list | dict | NAType:
    """
    Validate and process GPU type value.

    Args:
        gpu_type_value (list | dict | np.ndarray): The GPU type value from the DataFrame.
        include_cpu_only_jobs (bool): Whether CPU-only jobs are allowed.

    Returns:
        list | dict | NATpe: Processed GPU type value for GPU jobs or pd.NA for CPU-only jobs.

    Raises:
        JobPreprocessingError: If GPU type is null and CPU-only jobs are not allowed.
    """

    # Handle dict and list types first (these are never NA)
    if isinstance(gpu_type_value, (dict, list)):
        return gpu_type_value

    # Handle numpy arrays
    elif isinstance(gpu_type_value, np.ndarray):
        return gpu_type_value.tolist()

    # Handle missing/empty values (now only NAType remains)
    elif pd.isna(gpu_type_value):
        if not include_cpu_only_jobs:
            raise JobPreprocessingError(
                error_type=PreprocessingErrorTypeEnum.GPU_TYPE_NULL,
                info="GPU Type is null but include_cpu_only_jobs is False",
            )
        return pd.NA


def _safe_apply_function(
    func: Callable, *args: object, job_id: int | None = None, idx: int | None = None
) -> int | NAType:
    """
    Safely apply calculation functions, catching JobPreprocessingError and logging it.

    This function wraps calculation functions to catch JobPreprocessingError exceptions
    that may occur during column or metric processing, logs the error details for
    later review, and returns pd.NA instead of allowing the error to propagate.

    Args:
        func (Callable): The calculation function to call with the provided arguments.
        *args: Variable length argument list to pass to the metric function.
        job_id (int | None, optional): Job ID associated with the row being processed,
            used for error logging and tracking. Defaults to None.
        idx (int | None, optional): DataFrame index of the row being processed,
            used for error tracking and row removal. Defaults to None.

    Returns:
        int | NAType: The result of calling func(*args) if successful, or pd.NA if a
            JobPreprocessingError occurs.

    Note:
        When a JobPreprocessingError is caught, the function:
        - Adds the index to error_indices for later row removal
        - Logs error details to processing_error_logs for summary reporting
        - Returns pd.NA to maintain DataFrame structure
    """
    try:
        return func(*args)
    except JobPreprocessingError as e:
        if idx is not None:
            error_indices.add(idx)
        processing_error_logs.append({
            "job_id": job_id,
            "index": idx,
            "error_type": e.error_type,
            "info": e.info,
        })
        return pd.NA


def _fill_missing(res: pd.DataFrame, include_cpu_only_jobs: bool) -> None:
    """
    Intended for internal use inside preprocess_data() only. Fill missing values in the DataFrame with default values.

    Args:
        res (pd.DataFrame): The DataFrame to fill missing values in.
        include_cpu_only_jobs (bool): Whether to include CPU-only jobs in the DataFrame.

    Returns:
        None: The function modifies the DataFrame in place.
    """

    # all NaN values are np.nan
    # fill default values for specific columns
    fill_map = {
        "ArrayID": lambda col: col.fillna(-1),
        "Interactive": lambda col: col.fillna("non-interactive"),
        "Constraints": lambda col: col.fillna("").apply(lambda x: [] if isinstance(x, str) and x == "" else list(x)),
        "GPUs": lambda col: col.fillna(0),
        "NodeList": lambda col:  col.fillna("").apply(lambda x: [] if isinstance(x, str) and x == "" else list(x)),
    }

    res.loc[:, "GPUType"] = res.apply(
        lambda row: _safe_apply_function(
            _validate_gpu_type, row["GPUType"], include_cpu_only_jobs, job_id=row["JobID"], idx=row.name
        ),
        axis=1,
    )

    for col, fill_func in fill_map.items():
        if col in res.columns:
            res.loc[:, col] = fill_func(res[col])


def _validate_columns_and_filter_records(
    data: pd.DataFrame,
    min_elapsed_seconds: int,
    include_failed_cancelled_jobs: bool,
    include_cpu_only_jobs: bool,
    include_custom_qos_jobs: bool,
) -> pd.DataFrame:
    """
    Validate required columns and filter records based on specified criteria.

    This function performs two main operations:
    1. Validates that all required columns are present and warns about missing optional columns
    2. Applies filtering conditions to remove unwanted records based on various criteria

    Args:
        data (pd.DataFrame): The input dataframe to validate and filter.
        min_elapsed_seconds (int): Minimum elapsed time in seconds to keep a job record.
        include_failed_cancelled_jobs (bool): Whether to include jobs with status FAILED or CANCELLED.
        include_cpu_only_jobs (bool): Whether to include jobs that do not use GPUs (CPU-only jobs).
        include_custom_qos_jobs (bool): Whether to include entries with custom qos values.

    Returns:
        pd.DataFrame: The validated and filtered dataframe.

    Raises:
        KeyError: If any columns in RequiredColumnsEnum do not exist in the dataframe.

    Notes:
        # Handling missing columns logic:
        - columns in REQUIRED_COLUMNS are columns that are must-have for basic metrics calculation.
        - columns in OPTIONAL_COLUMNS are columns that are involved in preprocessing logics.
        - For any columns in REQUIRED_COLUMNS that do not exist, a KeyError will be raised.
        - For any columns in OPTIONAL_COLUMNS but not in REQUIRED_COLUMNS, a warning will be raised.
        - _fill_missing, records filtering, and type conversion logic will happen only if columns involved exist

    """
    qos_values = set([member.value for member in QOSEnum])
    exist_column_set = set(data.columns.to_list())

    # Ensure required columns are present
    for required_col in RequiredColumnsEnum:
        if required_col.value not in exist_column_set:
            raise KeyError(f"Column {required_col.value} does not exist in dataframe.")

    # raise warnings if optional columns are not present
    for optional_col in OptionalColumnsEnum:
        if optional_col.value not in exist_column_set:
            warnings.warn(
                (
                    f"Column '{optional_col.value}' is missing from the dataframe. "
                    "This may impact filtering operations and downstream processing."
                ),
                UserWarning,
                stacklevel=2,
            )

    # filtering records
    mask = pd.Series([True] * len(data), index=data.index)

    # Get partition info for GPU filtering
    partition_info = PartitionInfoFetcher().get_info()
    gpu_partitions = [p["name"] for p in partition_info if p["type"] == PartitionTypeEnum.GPU.value]

    filter_conditions = {
        "Elapsed": lambda df: df["Elapsed"] >= min_elapsed_seconds,
        "Account": lambda df: df["Account"] != AdminsAccountEnum.ROOT.value,
        "Partition": lambda df: (df["Partition"] != AdminPartitionEnum.BUILDING.value)
        & (include_cpu_only_jobs | df["Partition"].isin(gpu_partitions)),
        "QOS": lambda df: (df["QOS"] != QOSEnum.UPDATES.value)
        & (include_custom_qos_jobs | df["QOS"].isin(qos_values)),
        "Status": lambda df: include_failed_cancelled_jobs
        | ((df["Status"] != StatusEnum.FAILED.value) & (df["Status"] != StatusEnum.CANCELLED.value)),
    }

    for col, func in filter_conditions.items():
        if col not in exist_column_set:
            continue
        mask &= func(data)

    return data[mask].copy()


def _cast_type_and_add_columns(data: pd.DataFrame) -> None:
    """
    Cast existing columns to appropriate data types and add derived metrics as new columns.

    Handles both empty and non-empty dataframes by applying type casting to existing columns
        and either adding empty columns with correct dtypes or calculating actual derived values.

    Raises a warning if the dataframe is empty after preprocessing operations.

    Args:
        data (pd.DataFrame): The dataframe to modify. Must contain the required columns for processing.

    Returns:
        None: The function modifies the DataFrame in place.

    Warnings:
        UserWarning: If the dataframe is empty after filtering and preprocessing operations.
    """
    exist_column_set = set(data.columns.to_list())

    if data.empty:
        # Raise warning for empty dataframe
        warnings.warn("Dataframe results from database and filtering is empty.", UserWarning, stacklevel=3)

    # Type casting for columns involving time
    time_columns = ["StartTime", "SubmitTime"]
    for col in time_columns:
        if col not in exist_column_set:
            continue
        data[col] = pd.to_datetime(data[col], errors="coerce")

    duration_columns = ["TimeLimit", "Elapsed"]
    for col in duration_columns:
        if col not in exist_column_set:
            continue
        target_col = data[col] * 60 if col == "TimeLimit" else data[col]
        data[col] = pd.to_timedelta(target_col, unit="s", errors="coerce")

    # Convert columns to categorical
    for col, enum_obj in ATTRIBUTE_CATEGORIES.items():
        if col not in exist_column_set:
            continue
        enum_values = [e.value for e in enum_obj]
        unique_values = data[col].unique().tolist()
        all_categories = list(set(enum_values) | set(unique_values))
        data[col] = pd.Categorical(data[col], categories=all_categories, ordered=False)

    if data.empty:
        # Add new columns with correct types for empty dataframe
        data["Queued"] = pd.Series([], dtype="timedelta64[ns]")
        data["vram_constraint"] = pd.Series([], dtype=pd.Int64Dtype())
        data["partition_constraint"] = pd.Series([], dtype=pd.Int64Dtype())
        data["requested_vram"] = pd.Series([], dtype=pd.Int64Dtype())
        data["allocated_vram"] = pd.Series([], dtype=pd.Int64Dtype())
        # Only add user_jobs/account_jobs if columns exist
        if "User" in data.columns:
            data["user_jobs"] = pd.Series([], dtype=pd.Int64Dtype())
        if "Account" in data.columns:
            data["account_jobs"] = pd.Series([], dtype=pd.Int64Dtype())
    else:
        # Calculate queue time
        data.loc[:, "Queued"] = data["StartTime"] - data["SubmitTime"]

        # Apply all metrics using the single safe function
        data.loc[:, "vram_constraint"] = data.apply(
            lambda row: _safe_apply_function(
                _get_vram_constraint, row["Constraints"], row["GPUs"], job_id=row["JobID"], idx=row.name
            ),
            axis=1,
        ).astype(pd.Int64Dtype())

        data.loc[:, "partition_constraint"] = data.apply(
            lambda row: _safe_apply_function(
                _get_partition_constraint, row["Partition"], row["GPUs"], job_id=row["JobID"], idx=row.name
            ),
            axis=1,
        ).astype(pd.Int64Dtype())

        data.loc[:, "requested_vram"] = data.apply(
            lambda row: _safe_apply_function(
                _get_requested_vram,
                row["vram_constraint"],
                row["partition_constraint"],
                job_id=row["JobID"],
                idx=row.name,
            ),
            axis=1,
        ).astype(pd.Int64Dtype())

        data.loc[:, "allocated_vram"] = data.apply(
            lambda row: _safe_apply_function(
                _get_approx_allocated_vram,
                row["GPUType"],
                row["NodeList"],
                row["GPUs"],
                row["GPUMemUsage"],
                job_id=row["JobID"],
                idx=row.name,
            ),
            axis=1,
        )

        if error_indices:
            data = data.drop(index=list(error_indices)).reset_index(drop=True)

    # Add derived columns for user_jobs and account_jobs only if the source columns exist
    if "User" in exist_column_set:
        data.loc[:, "user_jobs"] = data.groupby("User", observed=True)["User"].transform("size")
    if "Account" in exist_column_set:
        data.loc[:, "account_jobs"] = data.groupby("Account", observed=True)["Account"].transform("size")


def _check_for_infinity_values(data: pd.DataFrame) -> None:
    """
    Check for infinity values in memory usage columns and raise warnings if found.

    Args:
        data (pd.DataFrame): The dataframe to check for infinity values.

    Returns:
        None: The function only raises warnings if infinity values are found.
    """
    mem_usage_columns = ["CPUMemUsage", "GPUMemUsage"]
    exist_column_set = set(data.columns.to_list())
    for col_name in mem_usage_columns:
        if col_name not in exist_column_set:
            continue
        filtered = data[data[col_name] == np.inf].copy()
        if len(filtered) > 0:
            message = f"Some entries in {col_name} having infinity values. This may be caused by an overflow."
            warnings.warn(message=message, stacklevel=2, category=UserWarning)


def _write_preprocessing_error_logs(preprocessing_error_logs: list[dict]) -> None:
    """
    Write preprocessing error logs to a log file.

    Args:
        preprocessing_error_logs (list[dict]): List of error records to write to file.

    Returns:
        None: Writes the error summary to PREPROCESSING_ERRORS_LOG_FILE.
    """
    if not preprocessing_error_logs:
        return

    print(
        f"Found {len(preprocessing_error_logs)} records with errors. "
        f"Reporting them to a summary file {PREPROCESSING_ERRORS_LOG_FILE}."
    )

    if PREPROCESSING_ERRORS_LOG_FILE.exists():
        print(f"Processing error log file already exists. Overwriting {PREPROCESSING_ERRORS_LOG_FILE.name}")

    summary_lines = [
        "Records causing processing errors that were ignored:\n",
        "=" * 30 + "\n",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Total errors: {len(preprocessing_error_logs)}\n\n",
    ]

    for record in preprocessing_error_logs:
        summary_lines.append(
            f"Job ID: {record['job_id']}\n"
            f"Error Type: {PreprocessingErrorTypeEnum(record['error_type']).value}\n"
            f"Info: {record['info']}\n"
            f"{'-' * 40}\n\n"
        )

    # Add all job IDs in one line for easy copying
    job_ids = [str(record["job_id"]) for record in preprocessing_error_logs if record["job_id"] is not None]
    if job_ids:
        summary_lines.append(f"All Job IDs: {', '.join(job_ids)}\n")

    with open(PREPROCESSING_ERRORS_LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)


def preprocess_data(
    input_df: pd.DataFrame,
    min_elapsed_seconds: int = DEFAULT_MIN_ELAPSED_SECONDS,
    include_failed_cancelled_jobs: bool = False,
    include_cpu_only_jobs: bool = False,
    include_custom_qos_jobs: bool = False,
    apply_filter: bool = True,
) -> pd.DataFrame:
    """
    Preprocess dataframe, filtering out unwanted rows and columns, filling missing values and converting types.

    This function will take in a dataframe to create a new dataframe satisfying given criteria.


    Args:
        input_df (pd.DataFrame): The input dataframe containing job data.
        min_elapsed_seconds (int, optional): Minimum elapsed time in seconds to keep a job record. Defaults to 600.
        include_failed_cancelled_jobs (bool, optional): Whether to include jobs with status FAILED or CANCELLED.
        include_cpu_only_jobs (bool, optional): Whether to include jobs that do not use GPUs (CPU-only jobs).
        include_custom_qos_jobs (bool, optional): Whether to include entries with custom qos values or not.
            Default to False
        apply_filter (bool, optional): Whether to apply filtering operations and columns removal to the data.
            Defaults to True.

    Notes:
        - The function supports two formats for the 'GPUType' column in the dataframe:
            1. Old format: list of GPU type strings, e.g. ['a100', 'v100']
            2. New format: dict mapping GPU type to count, e.g. {'a100': 2, 'v100': 1}
        - Both formats are automatically detected and handled for all VRAM calculations and downstream processing.
        - The output DataFrame will have missing values filled, time columns converted,
          and new columns added for VRAM and job statistics.

    Returns:
        pd.DataFrame: The preprocessed dataframe

    """
    data = input_df.copy()
    if apply_filter:
        # Drop unnecessary columns, ignoring errors in case any of them is not in the dataframe
        data = input_df.drop(
            columns=[member.value for member in ExcludedColumnsEnum if member.value in input_df.columns],
            axis=1,
            inplace=False,
        )
        # Perform column validation and filtering
        data = _validate_columns_and_filter_records(
            data,
            min_elapsed_seconds,
            include_failed_cancelled_jobs,
            include_cpu_only_jobs,
            include_custom_qos_jobs,
        )
    # Log the format of GPUType being used
    if not data.empty:
        first_non_null = data["GPUType"].dropna().iloc[0]
        if isinstance(first_non_null, dict):
            print("[Preprocessing] Running with new database format: GPU types as dictionary.")
        elif isinstance(first_non_null, list):
            print("[Preprocessing] Running with old database format: GPU types as list.")
    _fill_missing(data, include_cpu_only_jobs)
    _cast_type_and_add_columns(data)

    # Check for infinity values in memory usage columns
    _check_for_infinity_values(data)

    # Identify and handle duplicate JobIDs
    duplicate_rows = data[data["JobID"].duplicated(keep=False)]
    if not duplicate_rows.empty:
        duplicate_message = (
            f"{len(duplicate_rows['JobID'].unique().tolist())} duplicate JobIDs detected. "
            "Keeping only the latest entry for each JobID."
        )
        warnings.warn(message=duplicate_message, stacklevel=2, category=UserWarning)
        data_sorted = data.sort_values(by="SubmitTime", ascending=False)  # Sort by SubmitTime to keep the latest entry
        data = data_sorted.drop_duplicates(subset=["JobID"], keep="first")  # Keep the latest entry for each JobID

    # Save preprocessing error logs to a file.
    _write_preprocessing_error_logs(processing_error_logs)

    # Reset the error logs after writing to file
    processing_error_logs.clear()
    error_indices.clear()

    return data