import re
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from pandas.api.typing import NAType

from ..config.constants import (
    VRAM_VALUES,
    DEFAULT_MIN_ELAPSED_SECONDS,
    ATTRIBUTE_CATEGORIES,
    MULTIVALENT_GPUS,
    PARTITION_TO_GPU_MAP,
)
from ..config.enum_constants import (
    StatusEnum,
    AdminsAccountEnum,
    AdminPartitionEnum,
    QOSEnum,
    PartitionTypeEnum,
    PreprocessingErrorTypeEnum,
)
from ..config.remote_config import PartitionInfoFetcher
from ..config.paths import PREPROCESSING_ERRORS_LOG_FILE
from .errors import JobProcessingError

processing_error_logs: list = []
error_indices: set = set()


def _get_multivalent_vram_based_on_node(gpu_type: str, node: str) -> int:
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
    """
    gpu_type = gpu_type.lower()
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


def _get_vram_constraint(constraints: list[str], gpu_count: int) -> int | NAType:
    """
    Get the VRAM assigned to a job based on its constraints and GPU usage.

    This function extracts VRAM requests from the job constraints and returns the maximum requested VRAM from the
    constraints.

    Args:
        constraints (list[str]): List of constraints from the job, which may include VRAM requests.
        gpu_count (int): Number of GPUs requested by the job.

    Returns:
        int | NAType: Maximum VRAM amount in GiB obtained based on the provided constraints, multiplied by the
                    number of GPUs. Returns pd.NA if no VRAM constraints are provided or if no GPUs are requested.

    Raises:
        JobProcessingError: If a malformed constraint is encountered or if an unknown GPU type is specified.
    """
    vram_constraints = []
    for constr in constraints:
        constr = constr.strip("'").lower()  # Normalize constraints to lowercase and strip quotes
        if constr.startswith("vram"):
            vram_constraints.append(int(constr.replace("vram", "")))
        elif constr.startswith("gpu"):
            # if the constraint starts with "gpu", it is expected to be in the format "gpu:type"
            split_constr = constr.split(":")
            if len(split_constr) <= 1:
                # Add error records for malformed constraints and missing GPU types
                raise JobProcessingError(PreprocessingErrorTypeEnum.MALFORMED_CONSTRAINT, constr)

            gpu_type = split_constr[1].lower()

            if gpu_type in VRAM_VALUES:
                vram_constraints.append(VRAM_VALUES[gpu_type])
            else:
                raise JobProcessingError(PreprocessingErrorTypeEnum.UNKNOWN_GPU_TYPE, gpu_type)
        else:
            # if they enter a GPU name without the prefix
            if constr in VRAM_VALUES:
                vram_constraints.append(VRAM_VALUES[constr])

    if not (len(vram_constraints)):
        return pd.NA  # if no VRAM constraints are provided or no GPUs are requested return pd.NA

    return max(vram_constraints) * gpu_count


def _get_partition_gpu(partition: str) -> str | None:
    """
    Get the GPU type based on the partition if it only has one type of GPU.

    This function maps specific partition names to their corresponding GPU types.

    Args:
        partition (str): The name of the partition (e.g., "superpod-a100", "umd-cscdr-gpu").

    Returns:
        str | None: The GPU type associated with the partition or None if no specific mapping exists.
    """
    return PARTITION_TO_GPU_MAP.get(partition.lower(), None)


def _get_partition_constraint(partition: str, gpu_count: int) -> int | NAType:
    """
    Get the VRAM size based on the partition name requested.

    This function returns the VRAM size in GiB for a given partition name if it has only one type of GPU.
    If the partition is not recognized, or if it has multiple types of GPUs, it returns NAType.

    Args:
        partition (str): The name of the partition (e.g., "superpod-a100", "umd-cscdr-gpu").
        gpu_count (int): The number of GPUs requested by the job.

    Returns:
        int | NAType: The requested VRAM in GiB or NAType if the partition is not recognized.
    """
    gpu_type = _get_partition_gpu(partition)
    if gpu_type is None:
        # if the GPU Type is not inferrable from the partition, return NAType
        return pd.NA
    return VRAM_VALUES[gpu_type] * gpu_count


def _get_requested_vram(vram_constraint: int | NAType, partition_constraint: int | NAType) -> int | NAType:
    """
    Get the requested VRAM for a job based on its constraints and partition.

    This function determines the requested VRAM for a job by checking the VRAM constraint and the partition constraint.
    If both are provided, it returns the partition constraint as that is more accurate.
    If only one is provided, it returns that value.
    If neither is provided, it returns NAType.

    Args:
        vram_constraint (int | NAType): The VRAM constraint from the job's constraints.
        partition_constraint (int | NAType): The VRAM size based on the partition name.

    Returns:
        int | NAType: The requested VRAM in GiB or NAType if no constraints are provided.
    """
    if pd.isna(vram_constraint) and pd.isna(partition_constraint):
        return pd.NA
    if pd.isna(partition_constraint):
        return vram_constraint

    # if a partition constraint is provided, we use it
    return partition_constraint


def _calculate_approx_vram_single_gpu_type(
    gpu_types: list[str] | dict[str, int], node_list: list[str], gpu_count: int, gpu_mem_usage: int
) -> int:
    """
    Calculate the approximate VRAM for a job with a single GPU type.

    This helper function computes the total VRAM allocated for a job based on the GPU type,
    the nodes where the job ran, the number of GPUs requested, and the GPU memory usage.

    Args:
        gpu_types:
            - list[str]: list containing a single GPU type used in the job.
            - dict[str, int]: dictionary of GPU types and the count of GPUs of each type used in the job.
        node_list (list[str]): List of nodes that the job ran on.
        gpu_count (int): Number of GPUs requested by the job.
        gpu_mem_usage (int): GPU memory usage in bytes.

    Returns:
        int: Total allocated VRAM for the job in GiB (gibibyte).

    Raises:
        JobProcessingError: If an unknown GPU type is encountered or if no valid nodes are found for a multivalent GPU.
    """

    if isinstance(gpu_types, dict):
        gpu, gpu_count = list(gpu_types.items())[0]
    else:
        gpu = gpu_types[0]
    gpu = gpu.lower()

    if gpu not in MULTIVALENT_GPUS:
        # if the GPU is not multivalent, return the VRAM value for that GPU
        if gpu in VRAM_VALUES:
            return VRAM_VALUES[gpu] * gpu_count
        else:
            raise JobProcessingError(PreprocessingErrorTypeEnum.UNKNOWN_GPU_TYPE, gpu)

    # calculate VRAM for multivalent GPUs
    total_vram = 0

    # if all GPUs are on the same node, multiply the VRAM of that node by the number of GPUs
    if len(node_list) == 1:
        node = node_list[0]
        total_vram = _get_multivalent_vram_based_on_node(gpu, node) * gpu_count

    # if all GPUs are on different nodes, sum the VRAM of each node
    # and return the total VRAM
    elif len(node_list) == gpu_count:
        for node in node_list:
            total_vram += _get_multivalent_vram_based_on_node(gpu, node)

    # if there are multiple nodes, but not all GPUs are on different nodes
    # we need to calculate the total VRAM based on the minimum VRAM of the nodes
    else:
        # calculate available VRAM for all nodes in the node_list
        vram_values = set()  # to avoid duplicates
        for node in node_list:
            node_vram = _get_multivalent_vram_based_on_node(gpu, node)
            if node_vram != 0:  # only consider nodes with non-zero VRAM
                vram_values.add(_get_multivalent_vram_based_on_node(gpu, node))

        if not vram_values:
            # if no valid nodes are found for the multivalent GPU type in the node list, log an error
            raise JobProcessingError(
                PreprocessingErrorTypeEnum.UNKNOWN_GPU_TYPE,
                f"No valid nodes found for multivalent GPU type '{gpu}' in node list: {node_list}",
            )

        sorted_vram_values = sorted(list(vram_values))
        total_vram = sorted_vram_values.pop(0) * gpu_count  # use the node with the minimum VRAM value
        # if the total VRAM is less than the GPU memory usage, use the VRAM from the GPU in the next larger node
        while total_vram < (gpu_mem_usage / 2**30) and sorted_vram_values:
            total_vram = sorted_vram_values.pop(0) * gpu_count

    return total_vram


def _adjust_vram_for_multivalent_gpus(
    multivalent: dict,
    allocated_vram: int,
    gpu_mem_usage: int | float,
    gpus_with_exact_values: dict[str, int],
) -> int:
    """
    Adjust the allocated VRAM for multivalent GPUs to meet or exceed the GPU memory usage.

    This function increases the allocated VRAM by adding the minimum VRAM for each multivalent GPU
    until the total allocated VRAM is at least as large as the required GPU memory usage.

    Args:
        multivalent (dict): Dictionary of GPU types (str) to counts (int) for multivalent GPUs.
        allocated_vram (int): Current total allocated VRAM in GiB.
        gpu_mem_usage (int | float): GPU memory usage in bytes.
        gpus_with_exact_values (dict[str, int]): Dictionary of GPU types (str) to exact VRAM values (int).

    Returns:
        int: Adjusted total allocated VRAM in GiB.
    """
    # Adjust VRAM for GPUs with exact values first
    for gpu, exact_vram in gpus_with_exact_values.items():
        allocated_vram += exact_vram
        multivalent[gpu] -= 1  # Reduce count for GPUs with exact values

    # Assume they wanted the bigger VRAM variant for each GPU until the condition is satisfied
    for gpu, gpu_count in multivalent.items():
        while gpu_count > 0 and allocated_vram < (gpu_mem_usage / 2**30):
            allocated_vram += min(MULTIVALENT_GPUS[gpu])
            gpu_count -= 1

    return allocated_vram


def _get_possible_vram_values(multivalent_gpu_type: str, node_list: list[str]) -> list[int]:
    """
    Return all possible VRAM values for a given multivalent GPU type across a list of nodes.

    Args:
        multivalent_gpu_type (str): The GPU type (e.g., "a100", "v100").
        node_list (list[str]): List of node names that the job ran on.

    Returns:
        list[int]: List of non-zero VRAM values for the given GPU across nodes.
    """

    multivalent_gpu = multivalent_gpu_type.lower()
    possible_vrams = []
    for node in node_list:
        vram = _get_multivalent_vram_based_on_node(multivalent_gpu, node)
        if vram in MULTIVALENT_GPUS[multivalent_gpu]:  # if it matches a node for the given GPU
            possible_vrams.append(vram)
    return possible_vrams


def _can_calculate_accurately(multivalent_gpu_type: str, count: int, node_list: list[str]) -> bool:
    """
    Determine whether VRAM can be calculated accurately for a multivalent GPU type based on the job's node list.

    Args:
        multivalent_gpu_type (str): The GPU type (e.g., "a100", "v100").
        count (int): Number of GPUs of this type.
        node_list (list[str]): List of node names that the job ran on.

    Returns:
        bool:  True if all there's one possible VRAM value for this GPU type across given nodes or
               if each node corresponds to a different possible VRAM value for this GPU type.
    """
    multivalent_gpu = multivalent_gpu_type.lower()
    possible_vrams = _get_possible_vram_values(multivalent_gpu, node_list)
    return len(possible_vrams) == count or len(set(possible_vrams)) == 1


def _calculate_vram_accurately(multivalent_gpu_type: str, count: int, node_list: list[str]) -> int:
    """
    Calculate VRAM for a multivalent GPU type based on the job's node list.

    This can be done when all matched nodes have consistent VRAM configuration or enough distinct nodes exist.

    Args:
        multivalent_gpu_type (str): The GPU type (e.g., "a100", "v100").
        count (int): The number of GPUs of this type used in the job.
        node_list (list[str]): List of node names that the job ran on.

    Returns:
        int: Total VRAM in GiB for the given GPU type.
    """
    multivalent_gpu = multivalent_gpu_type.lower()
    possible_vrams = _get_possible_vram_values(multivalent_gpu, node_list)
    # If all possible VRAM values are the same, return that value multiplied by the count
    if len(set(possible_vrams)) == 1:
        return possible_vrams[0] * count

    # Otherwise, return the sum of all matching VRAM values
    return sum(possible_vrams)


def _calculate_non_multivalent_vram(non_multivalent: dict) -> int:
    """
    Calculate the VRAM allocated for non-multivalent GPUs.

    Args:
        non_multivalent (dict): Dictionary with non-multivalent GPU types as keys and their counts as values.

    Returns:
        int: Total allocated VRAM for non-multivalent GPUs in GiB.

    Raises:
        JobProcessingError: If an unknown GPU type is encountered.
    """
    allocated_vram = 0
    for gpu, count in non_multivalent.items():
        if gpu in VRAM_VALUES:
            allocated_vram += VRAM_VALUES[gpu] * count
        else:
            raise JobProcessingError(PreprocessingErrorTypeEnum.UNKNOWN_GPU_TYPE, gpu)
    return allocated_vram


def _calculate_multivalent_vram(multivalent: dict, node_list: list[str], gpu_mem_usage: int) -> int:
    """
    Calculate the VRAM allocated for multivalent GPUs.

    Args:
        multivalent (dict): Dictionary with multivalent GPU types as keys and their counts as values.
        node_list (list[str]): List of nodes that the job ran on.
        gpu_mem_usage (int): GPU memory usage in bytes.

    Returns:
        int: Total allocated VRAM for multivalent GPUs in GiB.
    """
    allocated_vram = 0
    gpus_with_exact_values: dict[str, int] = dict()

    for gpu, count in multivalent.items():
        if _can_calculate_accurately(gpu, count, node_list):
            vram_value = _calculate_vram_accurately(gpu, count, node_list)
            allocated_vram += vram_value
            gpus_with_exact_values[gpu] = vram_value
        else:
            allocated_vram += min(MULTIVALENT_GPUS[gpu]) * count

    if allocated_vram < gpu_mem_usage / 2**30 and len(gpus_with_exact_values) < len(multivalent):
        allocated_vram = _adjust_vram_for_multivalent_gpus(
            multivalent, allocated_vram, gpu_mem_usage, gpus_with_exact_values
        )
    return allocated_vram


def _calculate_alloc_vram_multiple_gpu_types_with_count(
    gpu_types: dict[str, int], node_list: list[str], gpu_mem_usage: int
) -> int:
    """
    Calculate allocated VRAM for a job with multiple GPU types given a dictionary.

    The dictionary has GPU Types as keys and their respective counts as values.

    Args:
        gpu_types (dict[str, int]): Dictionary with GPU types as keys and their counts as values.
        node_list (list[str]): List of nodes that the job ran on.
        gpu_mem_usage (int): GPU memory usage in bytes.

    Returns:
        int: Total allocated VRAM for the job in GiB.
    """
    multivalent_gpus = {gpu.lower(): count for gpu, count in gpu_types.items() if gpu.lower() in MULTIVALENT_GPUS}
    non_multivalent_gpus = {
        gpu.lower(): count for gpu, count in gpu_types.items() if gpu.lower() not in MULTIVALENT_GPUS
    }

    alloc_vram = 0
    alloc_vram += _calculate_non_multivalent_vram(non_multivalent_gpus)
    alloc_vram += _calculate_multivalent_vram(multivalent_gpus, node_list, gpu_mem_usage)

    return alloc_vram


def _get_approx_allocated_vram(
    gpu_types: list[str] | dict[str, int], node_list: list[str], gpu_count: int, gpu_mem_usage: int
) -> int:
    """
    Get the total allocated VRAM for a job based on its GPU type and node list.

    This function estimates the total VRAM allocated for a job based on the GPU types used
    and the nodes that the job ran on.

    Args:
        gpu_types:
            This could be a list of strings (if using the old database format) or a dictionary with GPU types as keys
            and their counts as values (if using the new database format).
            - list[str]: List containing the types of GPUs used in the job.
            - dict[str, int]: Dictionary with the type of GPUs and the exact count used in the job.
        node_list (list[str]): List of nodes that the job ran on.
        gpu_count (int): Number of GPUs requested by the job.
        gpu_mem_usage (int): GPU memory usage in bytes.

    Returns:
        int: Total allocated (estimate) VRAM for the job in GiB (gibibyte).

    Raises:
        JobProcessingError: If an unknown GPU type is encountered or if the GPU types are malformed.

    Notes:
        - When `gpu_types` is a dictionary, the function calculates VRAM based on the counts of each GPU type.
        - For multivalent GPUs, the VRAM is determined based on the nodes where the GPUs are located.
        - If the exact number of GPUs is not known, the function uses the minimum VRAM value among the available GPUs.
    """

    if isinstance(gpu_types, (list, dict)):
        if not gpu_types:
            return 0
    elif pd.isna(gpu_types):
        return 0

    # Case 1: Handle jobs with one type of GPU
    if len(gpu_types) == 1:
        return _calculate_approx_vram_single_gpu_type(gpu_types, node_list, gpu_count, gpu_mem_usage)

    # Case 2: Handle jobs with multiple types of GPUs
    # Case 2.1: Handle jobs using the new GPUType format
    if isinstance(gpu_types, dict):
        gpu_types = {gpu.lower(): count for gpu, count in gpu_types.items()}
        total_vram = _calculate_alloc_vram_multiple_gpu_types_with_count(gpu_types, node_list, gpu_mem_usage)
        return total_vram

    # Case 2.2: Handle jobs with the old GPUType format (a list)

    # Calculate allocated VRAM when there are multiple GPU types in a job
    if len(gpu_types) == gpu_count:
        total_vram = 0
        for gpu in gpu_types:
            gpu = gpu.lower()
            if gpu in MULTIVALENT_GPUS:
                for node in node_list:
                    total_vram += _get_multivalent_vram_based_on_node(gpu, node)
            else:
                if gpu in VRAM_VALUES:
                    total_vram += VRAM_VALUES[gpu]
                else:
                    raise JobProcessingError(PreprocessingErrorTypeEnum.UNKNOWN_GPU_TYPE, gpu)
        return total_vram

    # Handle cases where the number of GPUs is different from number of GPUTypes.
    allocated_vrams = set()
    for gpu in gpu_types:
        gpu = gpu.lower()
        if gpu in MULTIVALENT_GPUS:
            for node in node_list:
                multivalent_vram = _get_multivalent_vram_based_on_node(gpu, node)
                if multivalent_vram != 0:
                    allocated_vrams.add(multivalent_vram)
        else:
            if gpu in VRAM_VALUES:
                allocated_vrams.add(VRAM_VALUES[gpu])
            else:
                raise JobProcessingError(PreprocessingErrorTypeEnum.UNKNOWN_GPU_TYPE, gpu)

    vram_values = sorted(list(allocated_vrams))
    total_vram = vram_values.pop(0) * gpu_count  # use the GPU with the minimum VRAM value
    # if the total VRAM is less than the GPU memory usage, use the VRAM from the next smallest GPU
    while total_vram < (gpu_mem_usage / 2**30) and vram_values:
        total_vram = vram_values.pop(0) * gpu_count
    return total_vram


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
        JobProcessingError: If GPU type is null and CPU-only jobs are not allowed.
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
            raise JobProcessingError(
                error_type=PreprocessingErrorTypeEnum.GPU_TYPE_NULL,
                info="GPU Type is null but include_cpu_only_jobs is False",
            )
        return pd.NA


def _safe_apply_function(
    func: Callable, *args: object, job_id: int | None = None, idx: int | None = None
) -> int | NAType:
    """
    Safely apply calculation functions, catching JobProcessingError and logging it.

    This function wraps calculation functions to catch JobProcessingError exceptions
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
            JobProcessingError occurs.

    Note:
        When a JobProcessingError is caught, the function:
        - Adds the index to error_indices for later row removal
        - Logs error details to processing_error_logs for summary reporting
        - Returns pd.NA to maintain DataFrame structure
    """
    try:
        return func(*args)
    except JobProcessingError as e:
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
