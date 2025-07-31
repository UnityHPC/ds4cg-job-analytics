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
)
from ..config.enum_constants import StatusEnum, AdminsAccountEnum, PartitionEnum, QOSEnum


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

        TODO (Ayush): Consider reading this information from a config file or database.
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


def _get_vram_constraint(constraints: list[str], gpu_count: int, gpu_mem_usage: int) -> int | NAType:
    """
    Get the VRAM assigned to a job based on its constraints and GPU usage.

    This function extracts VRAM requests from the job constraints and returns the maximum requested VRAM from the
    constraints.

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
        constr = constr.strip("'").lower()  # Normalize constraints to lowercase and strip quotes
        if constr.startswith("vram"):
            vram_constraints.append(int(constr.replace("vram", "")))
        elif constr.startswith("gpu"):
            # if the constraint starts with "gpu", it is expected to be in the format "gpu:type"
            split_constr = constr.split(":")
            if len(split_constr) > 1:
                gpu_type = split_constr[1].lower()
            else:
                print(
                    f"[ERROR] Malformed GPU constraint: '{constr}'. Supported type is gpu:<name> Setting VRAM to NA."
                )
                continue

            if gpu_type in VRAM_VALUES:
                vram_constraints.append(VRAM_VALUES[gpu_type])
            else:
                print(f"[ERROR] GPU type '{gpu_type}' not found in VRAM_VALUES. Setting VRAM to NA.")
        else:
            # if they enter a GPU name without the prefix
            if constr in VRAM_VALUES:
                vram_constraints.append(VRAM_VALUES[constr])

    if not (len(vram_constraints)) or gpu_count == 0:
        return pd.NA  # if there are no constraints, return NAType

    # TODO (Ayush): Check if we want to take max or min of the VRAM constraints
    return max(vram_constraints) * gpu_count


def _get_partition_gpu(partition: str) -> str:
    """
    Get the GPU type based on the partition name.

    This function maps specific partition names to their corresponding GPU types.

    Args:
        partition (str): The name of the partition (e.g., "superpod-a100", "umd-cscdr-gpu").

    Returns:
        str: The GPU type associated with the partition or the partition if no specific mapping exists.

    """
    temp = partition.replace("gypsum-", "")
    if partition in ["superpod-a100", "umd-cscdr-gpu", "uri-gpu", "cbio-gpu"]:
        return "a100-80g"
    if partition in ["power9-gpu", "power9-gpu-preempt"]:
        return "v100"
    if partition in ["ials-gpu"]:
        return "2080_ti"
    if partition in ["ece-gpu"]:
        return "a100-40g"
    if partition in ["lan"]:
        return "a40"
    if partition in ["astroth-gpu"]:
        return "2080"
    if partition in ["gpupod-l40s"]:
        return "l40s"
    return temp


def _get_partition_constraint(partition: str, gpu_count: int) -> int | NAType:
    """
    Get the VRAM size based on the partition name requested.

    This function returns the VRAM size in GiB for a given partition name. If the partition is not recognized,
    it returns NAType.

    Args:
        partition (str): The name of the partition (e.g., "superpod-a100", "umd-cscdr-gpu").
        gpu_count (int): The number of GPUs requested by the job.

    Returns:
        int | NAType: The VRAM size in GiB or NAType if the partition is not recognized.
    """
    gpu_type = _get_partition_gpu(partition).lower()
    if gpu_type not in VRAM_VALUES:
        # if the GPU type is not in VRAM_VALUES, return pd.NA
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
        ValueError: If an invalid GPU type is provided or if no valid nodes are found for a multivalent GPU type.
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
            raise ValueError(
                f"Invalid GPU type '{gpu}' provided. "
                f"Please choose from the supported types: {', '.join(VRAM_VALUES.keys())}."
            )

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
        # calculate all VRAM for all nodes in the node_list
        node_values = set()  # to avoid duplicates
        for node in node_list:
            node_vram = _get_multivalent_vram_based_on_node(gpu, node)
            if node_vram != 0:  # only consider nodes with non-zero VRAM
                node_values.add(_get_multivalent_vram_based_on_node(gpu, node))

        if not node_values:
            raise ValueError(f"No valid nodes found for multivalent GPU type '{gpu}' in node list: {node_list}")

        sorted_node_values = sorted(list(node_values))
        total_vram = sorted_node_values.pop(0) * gpu_count  # use the node with the minimum VRAM value
        # if the total VRAM is less than the GPU memory usage, use the VRAM from the GPU in the next larger node
        while total_vram < (gpu_mem_usage / 2**30) and sorted_node_values:
            total_vram = sorted_node_values.pop(0) * gpu_count

    return total_vram


def _adjust_vram_for_multivalent_gpus(multivalent: dict, allocated_vram: int, gpu_mem_usage: int | float) -> int:
    """
    Adjust the allocated VRAM for multivalent GPUs to meet or exceed the GPU memory usage.

    This function increases the allocated VRAM by adding the minimum VRAM for each multivalent GPU
    until the total allocated VRAM is at least as large as the required GPU memory usage.

    Args:
        multivalent (dict): Dictionary of GPU types (str) to counts (int) for multivalent GPUs.
        allocated_vram (int): Current total allocated VRAM in GiB.
        gpu_mem_usage (int | float): GPU memory usage in bytes.

    Returns:
        int: Adjusted total allocated VRAM in GiB.
    """

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


def _can_calculate_perfectly(multivalent_gpu_type: str, count: int, node_list: list[str]) -> bool:
    """
    Determine whether VRAM can be calculated precisely for a GPU type based on matching nodes.

    Args:
        multivalent_gpu_type (str): The GPU type (e.g., "a100", "v100").
        count (int): Number of GPUs of this type.
        node_list (list[str]): List of node names that the job ran on.

    Returns:
        bool: True if all matching VRAM values are the same or if there are enough distinct nodes for the GPU.
    """
    multivalent_gpu = multivalent_gpu_type.lower()
    possible_vrams = _get_possible_vram_values(multivalent_gpu, node_list)
    return len(possible_vrams) == count or len(set(possible_vrams)) == 1


def _calculate_precise_vram(multivalent_gpu_type: str, count: int, node_list: list[str]) -> int:
    """
    Calculate total VRAM for a multivalent GPU precisely.

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
    # Determine which GPUs are multivalent and which are not
    multivalent = {gpu.lower(): count for gpu, count in gpu_types.items() if gpu.lower() in MULTIVALENT_GPUS}
    non_multivalent = {gpu.lower(): count for gpu, count in gpu_types.items() if gpu.lower() not in MULTIVALENT_GPUS}

    allocated_vram = 0

    # Case 1: All GPUs are not multivalent so we know their exact VRAM only based on the GPUType.
    if len(multivalent) == 0:
        for gpu, count in non_multivalent.items():
            if gpu in VRAM_VALUES:
                allocated_vram += VRAM_VALUES[gpu] * count
            else:
                print(f"[ERROR] GPU type '{gpu}' not found in VRAM_VALUES. Setting VRAM to 0.")
        return allocated_vram

    # Case 2: All GPUs multivalent. Number of GPUs is either equal or more than the number of GPU nodes.
    if len(non_multivalent) == 0:
        gpus_with_exact_values = dict()  # to keep track of GPUs for which we can calculate exact VRAM

        for gpu, count in multivalent.items():
            if _can_calculate_perfectly(gpu, count, node_list):
                # If we can calculate perfectly, use the VRAM from the matching nodes
                vram_value = _calculate_precise_vram(gpu, count, node_list)
                allocated_vram += vram_value
                gpus_with_exact_values[gpu] = vram_value

            else:
                # Estimate using the minimum VRAM for multivalent GPUs
                allocated_vram += min(MULTIVALENT_GPUS[gpu]) * count

        # if the estimate is less than the usage and not all GPU VRAMs were calculated exactly, update it
        if allocated_vram < gpu_mem_usage / 2**30 and len(gpus_with_exact_values) < len(multivalent):
            allocated_vram = _adjust_vram_for_multivalent_gpus(multivalent, allocated_vram, gpu_mem_usage)

        return allocated_vram

    # Case 3: Mixed multivalent and non-multivalent GPUs

    # Add VRAM for non-multivalent GPUs
    for gpu, count in non_multivalent.items():
        if gpu in VRAM_VALUES:
            allocated_vram += VRAM_VALUES[gpu] * count
        else:
            print(f"[ERROR] GPU type '{gpu}' not found in VRAM_VALUES. Setting VRAM to 0.")
            allocated_vram += 0

    # for each multivalent GPU, find its corresponding node and calculate its VRAM
    node_idx = 0
    for gpu, count in multivalent.items():
        for _ in range(count):
            if node_idx < len(node_list):
                allocated_vram += _get_multivalent_vram_based_on_node(gpu, node_list[node_idx])
                node_idx += 1
            else:
                allocated_vram += min(MULTIVALENT_GPUS[gpu])

    if allocated_vram >= gpu_mem_usage / 2**30:
        # If the allocated VRAM is sufficient, no need to adjust
        return allocated_vram

    # If the allocated VRAM is still less than the GPU memory usage, adjust the VRAM
    allocated_vram = _adjust_vram_for_multivalent_gpus(multivalent, allocated_vram, gpu_mem_usage)
    return allocated_vram


def _get_approx_allocated_vram(
    job_id: int, gpu_types: list[str] | dict[str, int], node_list: list[str], gpu_count: int, gpu_mem_usage: int
) -> int:
    """
    Get the total allocated VRAM for a job based on its GPU type and node list.

    This function estimates the total VRAM allocated for a job based on the GPU types used
    and the nodes that the job ran on.

    Args:
        job_id (int): Unique identifier for the job.
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
        ValueError: If an invalid GPU type is provided.

    Notes:
        - When `gpu_types` is a dictionary, the function calculates VRAM based on the counts of each GPU type.
        - For multivalent GPUs, the VRAM is determined based on the nodes where the GPUs are located.
        - If the exact number of GPUs is not known, the function uses the minimum VRAM value among the available GPUs.
    """

    # Handle cases with one type of GPU
    if len(gpu_types) == 1:
        try:
            total_vram = _calculate_approx_vram_single_gpu_type(gpu_types, node_list, gpu_count, gpu_mem_usage)
        # TODO (Ayush): Update this based on decision to raise exceptions or not
        except ValueError as e:
            raise ValueError(
                f"Error calculating VRAM for GPU type '{gpu_types}' for job {job_id}. "
                f"Please ensure the GPU type is valid."
            ) from e
        return total_vram

    # Calculate approximate allocated VRAM for jobs with multiple GPUTypes using the new GPUType format
    if isinstance(gpu_types, dict):
        gpu_types = {gpu.lower(): count for gpu, count in gpu_types.items()}
        total_vram = _calculate_alloc_vram_multiple_gpu_types_with_count(gpu_types, node_list, gpu_mem_usage)
        return total_vram

    # Handle cases with GPU types in a list

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
                    print(f"[ERROR] GPU type '{gpu}' not found in VRAM_VALUES. Setting VRAM to 0.")
                    total_vram += 0
        return total_vram

    # Using the old data format, calculate allocated VRAM for jobs with multiple GPU Types when
    # the number of GPUs is different from number of GPUTypes.
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
                print(f"[ERROR] GPU type '{gpu}' not found in VRAM_VALUES. Setting VRAM to 0.")
                allocated_vrams.add(0)

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
    res.loc[:, "ArrayID"] = res["ArrayID"].fillna(-1)
    res.loc[:, "Interactive"] = res["Interactive"].fillna("non-interactive")
    res.loc[:, "Constraints"] = (
        res["Constraints"].fillna("").apply(lambda x: [] if isinstance(x, str) and x == "" else list(x))
    )
    res.loc[:, "GPUType"] = (
        res["GPUType"]
        .fillna("")
        .apply(
            lambda x: (["cpu"] if (isinstance(x, str) and x == "") else x.tolist() if isinstance(x, np.ndarray) else x)
        )
    )
    res.loc[:, "GPUs"] = res["GPUs"].fillna(0)


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

    data = input_df.drop(columns=["UUID", "EndTime", "Nodes", "Preempted"], axis=1, inplace=False)

    cond_gpu_type = (
        data["GPUType"].notna() | include_cpu_only_jobs
    )  # filter out GPUType is null, except when include_CPU_only_job is True
    cond_gpus = (
        data["GPUs"].notna() | include_cpu_only_jobs
    )  # filter out GPUs is null, except when include_CPU_only_job is True
    cond_failed_cancelled_jobs = (
        ((data["Status"] != StatusEnum.FAILED.value) & (data["Status"] != StatusEnum.CANCELLED.value))
        | include_failed_cancelled_jobs
    )  # filter out failed or cancelled jobs, except when include_fail_cancel_jobs is True

    res = data[
        cond_gpu_type
        & cond_gpus
        & cond_failed_cancelled_jobs
        & (data["Elapsed"] >= min_elapsed_seconds)  # filter in unit of second, not timedelta object
        & (data["Account"] != AdminsAccountEnum.ROOT.value)
        & (data["Partition"] != PartitionEnum.BUILDING.value)
        & (data["QOS"] != QOSEnum.UPDATES.value)
    ].copy()

    _fill_missing(res)

    first_non_null = res["GPUType"].dropna().iloc[0]
    # Log the format of GPUType being used
    if isinstance(first_non_null, dict):
        print("[Preprocessing] Running with new database format: GPU types as dictionary.")
    elif isinstance(first_non_null, list):
        print("[Preprocessing] Running with old database format: GPU types as list.")

    # Type casting for columns involving time
    time_columns = ["StartTime", "SubmitTime"]
    for col in time_columns:
        res[col] = pd.to_datetime(res[col], errors="coerce")

    timedelta_columns = ["TimeLimit", "Elapsed"]
    for col in timedelta_columns:
        res[col] = pd.to_timedelta(res[col], unit="s", errors="coerce")

    # Added parameters for calculating VRAM metrics
    res.loc[:, "Queued"] = res["StartTime"] - res["SubmitTime"]
    res.loc[:, "vram_constraint"] = res.apply(
        lambda row: _get_vram_constraint(row["Constraints"], row["GPUs"], row["GPUMemUsage"]), axis=1
    ).astype(pd.Int64Dtype())  # Use Int64Dtype to allow for nullable integers
    res.loc[:, "partition_constraint"] = res.apply(
        lambda row: _get_partition_constraint(row["Partition"], row["GPUs"]), axis=1
    ).astype(pd.Int64Dtype())  # Use Int64Dtype to allow for nullable integers
    res.loc[:, "requested_vram"] = res.apply(
        lambda row: _get_requested_vram(row["vram_constraint"], row["partition_constraint"]), axis=1
    ).astype(pd.Int64Dtype())  # Use Int64Dtype to allow for nullable integers
    res.loc[:, "allocated_vram"] = res.apply(
        lambda row: _get_approx_allocated_vram(
            row["JobID"], row["GPUType"], row["NodeList"], row["GPUs"], row["GPUMemUsage"]
        ),
        axis=1,
    )
    res.loc[:, "user_jobs"] = res.groupby("User")["User"].transform("size")
    res.loc[:, "account_jobs"] = res.groupby("Account")["Account"].transform("size")

    # Convert columns to categorical

    for col, enum_obj in ATTRIBUTE_CATEGORIES.items():
        enum_values = [e.value for e in enum_obj]
        unique_values = res[col].unique().tolist()
        all_categories = list(set(enum_values) | set(unique_values))
        res[col] = pd.Categorical(res[col], categories=all_categories, ordered=False)

    # Raise warning if GPUMemUsage or CPUMemUsage having infinity values
    mem_usage_columns = ["CPUMemUsage", "GPUMemUsage"]
    for col_name in mem_usage_columns:
        filtered = res[res[col_name] == np.inf].copy()
        if len(filtered) > 0:
            message = f"Some entries in {col_name} having infinity values. This may be caused by an overflow."
            warnings.warn(message=message, stacklevel=2, category=UserWarning)
    return res
