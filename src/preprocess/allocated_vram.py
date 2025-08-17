import pandas as pd
import re

from ..config.constants import (
    VRAM_VALUES,
    MULTIVALENT_GPUS,
)
from ..config.enum_constants import PreprocessingErrorTypeEnum
from .errors import JobProcessingError


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

    # Case 2.2.1: Handle cases where each GPU has a different type.
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

    # Case 2.2.2: Handle cases where the number of GPUs is different from number of GPUTypes.
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
