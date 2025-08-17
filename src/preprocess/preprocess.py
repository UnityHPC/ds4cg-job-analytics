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
    res.loc[:, "ArrayID"] = res["ArrayID"].fillna(-1)
    res.loc[:, "Interactive"] = res["Interactive"].fillna("non-interactive")
    res.loc[:, "Constraints"] = (
        res["Constraints"].fillna("").apply(lambda x: [] if isinstance(x, str) and x == "" else list(x))
    )
    res.loc[:, "GPUType"] = res.apply(
        lambda row: _safe_apply_function(
            _validate_gpu_type, row["GPUType"], include_cpu_only_jobs, job_id=row["JobID"], idx=row.name
        ),
        axis=1,
    )
    res.loc[:, "GPUs"] = res["GPUs"].fillna(0)
    res.loc[:, "NodeList"] = (
        res["NodeList"].fillna("").apply(lambda x: [] if isinstance(x, str) and x == "" else list(x))
    )


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
) -> pd.DataFrame:
    """
    Preprocess dataframe, filtering out unwanted rows and columns, filling missing values and converting types.

    This function will take in a dataframe to create a new dataframe satisfying given criteria.

    Args:
        input_df (pd.DataFrame): The input dataframe containing job data.
        min_elapsed_seconds (int, optional): Minimum elapsed time in seconds to keep a job record. Defaults to 600.
        include_failed_cancelled_jobs (bool, optional): Whether to include jobs with status FAILED or CANCELLED.
        include_cpu_only_jobs (bool, optional): Whether to include jobs that do not use GPUs (CPU-only jobs).

    Returns:
        pd.DataFrame: The preprocessed dataframe

    Notes:
        - The function supports two formats for the 'GPUType' column in the dataframe:
            1. Old format: list of GPU type strings, e.g. ['a100', 'v100']
            2. New format: dict mapping GPU type to count, e.g. {'a100': 2, 'v100': 1}
        - Both formats are automatically detected and handled for all VRAM calculations and downstream processing.
        - The output DataFrame will have missing values filled, time columns converted,
          and new columns added for VRAM and job statistics.
    """

    cols_to_remove = [col for col in ["UUID", "EndTime", "Nodes", "Preempted"] if col in input_df.columns]
    data = input_df.drop(columns=cols_to_remove, axis=1, inplace=False)

    first_non_null = data["GPUType"].dropna().iloc[0]
    # Log the format of GPUType being used
    if isinstance(first_non_null, dict):
        print("[Preprocessing] Running with new database format: GPU types as dictionary.")
    elif isinstance(first_non_null, list):
        print("[Preprocessing] Running with old database format: GPU types as list.")

    mask = pd.Series([True] * len(data), index=data.index)

    mask &= data["Elapsed"] >= min_elapsed_seconds
    mask &= data["Account"] != AdminsAccountEnum.ROOT.value
    mask &= data["Partition"] != AdminPartitionEnum.BUILDING.value
    mask &= data["QOS"] != QOSEnum.UPDATES.value
    # Filter out failed or cancelled jobs, except when include_failed_cancel_jobs is True
    mask &= (
        (data["Status"] != StatusEnum.FAILED.value) & (data["Status"] != StatusEnum.CANCELLED.value)
    ) | include_failed_cancelled_jobs
    # Filter out jobs whose partition type is not 'gpu', unless include_cpu_only_jobs is True.
    partition_info = PartitionInfoFetcher().get_info()
    gpu_partitions = [p["name"] for p in partition_info if p["type"] == PartitionTypeEnum.GPU.value]
    mask &= data["Partition"].isin(gpu_partitions) | include_cpu_only_jobs

    data = data[mask].copy()

    _fill_missing(data, include_cpu_only_jobs)

    # Type casting for columns involving time
    time_columns = ["StartTime", "SubmitTime"]
    for col in time_columns:
        data[col] = pd.to_datetime(data[col], errors="coerce")

    time_limit_in_seconds = data["TimeLimit"] * 60
    data["TimeLimit"] = pd.to_timedelta(time_limit_in_seconds, unit="s", errors="coerce")
    data["Elapsed"] = pd.to_timedelta(data["Elapsed"], unit="s", errors="coerce")

    # Added parameters for calculating VRAM metrics
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
            _get_requested_vram, row["vram_constraint"], row["partition_constraint"], job_id=row["JobID"], idx=row.name
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

    data.loc[:, "user_jobs"] = data.groupby("User")["User"].transform("size")
    data.loc[:, "account_jobs"] = data.groupby("Account")["Account"].transform("size")

    # Convert columns to categorical
    for col, enum_obj in ATTRIBUTE_CATEGORIES.items():
        enum_values = [e.value for e in enum_obj]
        unique_values = data[col].unique().tolist()
        all_categories = list(set(enum_values) | set(unique_values))
        data[col] = pd.Categorical(data[col], categories=all_categories, ordered=False)

    # Raise warning if GPUMemUsage or CPUMemUsage having infinity values
    mem_usage_columns = ["CPUMemUsage", "GPUMemUsage"]
    for col_name in mem_usage_columns:
        filtered = data[data[col_name] == np.inf].copy()
        if len(filtered) > 0:
            message = f"Some entries in {col_name} having infinity values. This may be caused by an overflow."
            warnings.warn(message=message, stacklevel=2, category=UserWarning)

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
