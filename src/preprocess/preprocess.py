import pandas as pd
from pandas.api.typing import NAType
import numpy as np
import re
import warnings

from ..config.constants import (
    VRAM_VALUES,
    DEFAULT_MIN_ELAPSED_SECONDS,
    ATTRIBUTE_CATEGORIES,
    MULTIVALENT_GPUS,
    # REQUIRED_COLUMNS,
    # OPTIONAL_COLUMNS,
)
from ..config.enum_constants import (
    StatusEnum,
    AdminsAccountEnum,
    AdminPartitionEnum,
    QOSEnum,
    PartitionTypeEnum,
    OptionalColumnsEnum,
    RequiredColumnsEnum,
    ExcludedColumnsEnum,
)
from ..config.remote_config import PartitionInfoFetcher


def _get_vram_from_node(gpu_type: str, node: str) -> int:
    """
    Calculate specific VRAM based on a node name for GPUs with multiple VRAM sizes.

    The function checks if a pairing of a given GPU and a node name exists. If it exists, it determines the amount of
    VRAM available for that GPU on that node.

    Args:
        gpu_type (str): Type of GPU (e.g., "a100", "v100").
        node (str): Name of the node.

    Returns:
        int: VRAM size in GiB for the given GPU type and node.
             Returns 0 if the node does not match any of the patterns for the given GPU type.

    Notes:
        This logic is based on the cluster specifications documented at:
        https://docs.unity.rc.umass.edu/documentation/cluster_specs/nodes/

        TODO (Ayush): Consider reading this information from a config file or database.
    """
    vram = 0
    if gpu_type not in MULTIVALENT_GPUS:
        # if the GPU is not multivalent we do not need to check the node
        vram = 0

    else:
        if gpu_type == "a100":
            if node.startswith("ece-gpu"):
                vram = 40  # A100 with 40GB
            elif re.match("^(gpu0(1[3-9]|2[0-4]))|(gpu042)|(umd-cscdr-gpu00[1-2])|(uri-gpu00[1-8])$", node):
                vram = 80  # A100 with 80GB
            else:
                # if the node does not match any of the patterns, it is not a valid node for this GPU type
                # so we return 0
                vram = 0
        elif gpu_type == "v100":
            if re.match("^(gpu00[1-7])|(power9-gpu009)|(power9-gpu01[0-6])$", node):
                vram = 16  # V100 with 16GB
            elif re.match("^(gpu01[1-2])|(power9-gpu00[1-8])$", node):
                vram = 32  # V100 with 32GB
            else:
                # if the node does not match any of the patterns, it is not a valid node for this GPU type
                # so we return 0
                vram = 0
    return vram


def _get_vram_constraint(constraints: list[str], gpu_count: int, gpu_mem_usage: int) -> int | NAType:
    """
    Get the VRAM assigned for a job based on its constraints and GPU usage.

    This function extracts VRAM requests from the job constraints and returns the maximum requested VRAM from the
    constraints. For GPU names that correspond to multiple VRAM values, take the minimum value that is not smaller
    than the amount of VRAM used by that job.

    Args:
        constraints (list[str]): List of constraints from the job, which may include VRAM requests.
        gpu_count (int): Number of GPUs requested by the job.
        gpu_mem_usage (int): GPU memory usage in bytes.

    Returns:
        int | NAType: Maximum VRAM amount in GiB obtained based on the provided constraints, multiplied by the
                    number of GPUs. Returns NAType if no VRAM constraints are present.

    """
    vram_constraints = []
    for constr in constraints:
        constr = constr.strip("'")
        if constr.startswith("vram"):
            vram_constraints.append(int(constr.replace("vram", "")))
        elif constr.startswith("gpu"):
            gpu_type = constr.split(":")[1]
            if gpu_type in MULTIVALENT_GPUS and (gpu_mem_usage / (2**30)) > VRAM_VALUES[gpu_type] * gpu_count:
                # For GPU names that correspond to multiple VRAM values, take the minimum value
                # that is not smaller than the amount of VRAM used by that job.
                vram_constraints.append(max(MULTIVALENT_GPUS[gpu_type]))
            else:
                vram_constraints.append(VRAM_VALUES[gpu_type])

    if not (len(vram_constraints)):
        return pd.NA  # if nothing is requested, return NAType

    return max(vram_constraints) * gpu_count


def _calculate_approx_vram_single_gpu_type(
    gpu_types: list[str], node_list: list[str], gpu_count: int, gpu_mem_usage: int
) -> int:
    """
    Calculate the approximate VRAM for a job with a single GPU type.

    This helper function computes the total VRAM allocated for a job based on the GPU type,
    the nodes where the job ran, the number of GPUs requested, and the GPU memory usage.

    Args:
        gpu_types (list[str]): List containing a single GPU type used in the job.
        node_list (list[str]): List of nodes where the job ran.
        gpu_count (int): Number of GPUs requested by the job.
        gpu_mem_usage (int): GPU memory usage in bytes.

    Returns:
        int: Total allocated VRAM for the job in GiB (gibibyte).

    Raises:
        ValueError: If no valid nodes are found for a multivalent GPU type in the node list.
    """
    gpu = gpu_types[0]
    if gpu not in MULTIVALENT_GPUS:
        # if the GPU is not multivalent, return the VRAM value for that GPU
        return VRAM_VALUES[gpu] * gpu_count

    # calculate VRAM for multivalent GPUs
    total_vram = 0

    # if all GPUs are on the same node, multiply the VRAM of that node by the number of GPUs
    if len(node_list) == 1:
        node = node_list[0]
        total_vram = _get_vram_from_node(gpu, node) * gpu_count

    # if all GPUs are on different nodes, sum the VRAM of each node
    # and return the total VRAM
    elif len(node_list) == gpu_count:
        for node in node_list:
            total_vram += _get_vram_from_node(gpu, node)

    # if there are multiple nodes, but not all GPUs are on different nodes
    # we need to calculate the total VRAM based on the minimum VRAM of the nodes
    else:
        # calculate all VRAM for all nodes in the node_list
        node_values = set()  # to avoid duplicates
        for node in node_list:
            node_vram = _get_vram_from_node(gpu, node)
            if node_vram != 0:  # only consider nodes with non-zero VRAM
                node_values.add(_get_vram_from_node(gpu, node))

        if not node_values:
            raise ValueError(f"No valid nodes found for multivalent GPU type '{gpu}' in node list: {node_list}")

        sorted_node_values = sorted(list(node_values))
        total_vram = sorted_node_values.pop(0) * gpu_count  # use the node with the minimum VRAM value
        # if the total VRAM is less than the GPU memory usage, use the VRAM from the GPU in the larger node
        while total_vram < (gpu_mem_usage / 2**30) and sorted_node_values:
            total_vram = sorted_node_values.pop(0) * gpu_count

    return total_vram


def _get_approx_allocated_vram(gpu_types: list[str], node_list: list[str], gpu_count: int, gpu_mem_usage: int) -> int:
    """
    Get the total allocated VRAM for a job based on its GPU type and node list.

    This function estimates the total VRAM allocated for a job based on the GPU types used
    and the nodes where the job ran.

    Args:
        gpu_types (list[str]): List of GPU types used in the job.
        node_list (list[str]): List of nodes where the job ran.
        gpu_count (int): Number of GPUs requested by the job.
        gpu_mem_usage (int): GPU memory usage in bytes.

    Returns:
        int: Total allocated (estimate) VRAM for the job in GiB (gibibyte).
    """

    # one type of GPU
    if len(gpu_types) == 1:
        total_vram = _calculate_approx_vram_single_gpu_type(gpu_types, node_list, gpu_count, gpu_mem_usage)
        return total_vram

    # Calculate allocated VRAM when there are multiple GPU types in a job
    if len(gpu_types) == gpu_count:
        total_vram = 0
        for gpu in gpu_types:
            if gpu in MULTIVALENT_GPUS:
                for node in node_list:
                    total_vram += _get_vram_from_node(gpu, node)
            else:
                total_vram += VRAM_VALUES[gpu]
        return total_vram

    # estimate VRAM for multiple GPUs where exact number isn't known
    # TODO (Ayush): update this based on the updated GPU types which specify exact number of GPUs
    allocated_vrams = set()
    for gpu in gpu_types:
        if gpu in MULTIVALENT_GPUS:
            for node in node_list:
                node_vram = _get_vram_from_node(gpu, node)
                if node_vram != 0:
                    allocated_vrams.add(_get_vram_from_node(gpu, node))
        else:
            allocated_vrams.add(VRAM_VALUES[gpu])

    vram_values = sorted(list(allocated_vrams))
    total_vram = vram_values.pop(0) * gpu_count  # use the GPU with the minimum VRAM value
    # if the total VRAM is less than the GPU memory usage, use the VRAM from the next smallest GPU
    while total_vram < (gpu_mem_usage / 2**30) and vram_values:
        total_vram = vram_values.pop(0) * gpu_count
    return total_vram


def _fill_missing(res: pd.DataFrame) -> None:
    """
    Intended for internal use inside preprocess_data() only. Fill missing values in the DataFrame with default values.

    Args:
        res (pd.DataFrame): The DataFrame to fill missing values in.

    Returns:
        None: The function modifies the DataFrame in place.
    """

    # all Nan value are np.nan
    # fill default values for specific columns
    fill_map = {
        "ArrayID": lambda col: col.fillna(-1),
        "Interactive": lambda col: col.fillna("non-interactive"),
        "Constraints": lambda col: col.fillna("").apply(lambda x: [] if isinstance(x, str) and x == "" else list(x)),
        "GPUType": lambda col: col.fillna("").apply(
            lambda x: (["cpu"] if (isinstance(x, str) and x == "") else x.tolist() if isinstance(x, np.ndarray) else x)
        ),
        "GPUs": lambda col: col.fillna(0),
    }

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
        data["allocated_vram"] = pd.Series([], dtype=pd.Int64Dtype())
    else:
        # Calculate queue time
        data.loc[:, "Queued"] = data["StartTime"] - data["SubmitTime"]

        # Calculate VRAM constraint from job constraints
        data.loc[:, "vram_constraint"] = data.apply(
            lambda row: _get_vram_constraint(row["Constraints"], row["GPUs"], row["GPUMemUsage"]), axis=1
        ).astype(pd.Int64Dtype())  # Use Int64Dtype to allow for nullable integers

        # Calculate allocated VRAM based on GPU type and nodes
        data.loc[:, "allocated_vram"] = data.apply(
            lambda row: _get_approx_allocated_vram(row["GPUType"], row["NodeList"], row["GPUs"], row["GPUMemUsage"]),
            axis=1,
        )


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


    Returns:
        pd.DataFrame: The preprocessed dataframe

    """
    data = input_df
    if apply_filter:
        # Drop unnecessary columns, ignoring errors in case any of them is not in the dataframe
        data = input_df.drop(
            columns=[member.value for member in ExcludedColumnsEnum], axis=1, inplace=False, errors="ignore"
        )
        # Perform column validation and filtering
        data = _validate_columns_and_filter_records(
            data,
            min_elapsed_seconds,
            include_failed_cancelled_jobs,
            include_cpu_only_jobs,
            include_custom_qos_jobs,
        )

    _fill_missing(data)

    _cast_type_and_add_columns(data)

    # Check for infinity values in memory usage columns
    _check_for_infinity_values(data)

    return data
