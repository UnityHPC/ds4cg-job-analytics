from ..config.enum_constants import PreprocessingErrorTypeEnum
from ..config.constants import (
    VRAM_VALUES,
    MONO_GPU_PARTITION_GPU_TYPE,
)
from pandas.api.typing import NAType
import pandas as pd
from .errors import JobPreprocessingError


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
        JobPreprocessingError: If a malformed constraint is encountered or if an unknown GPU type is specified.
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
                raise JobPreprocessingError(PreprocessingErrorTypeEnum.MALFORMED_CONSTRAINT, constr)

            gpu_type = split_constr[1].lower()

            if gpu_type in VRAM_VALUES:
                vram_constraints.append(VRAM_VALUES[gpu_type])
            else:
                raise JobPreprocessingError(PreprocessingErrorTypeEnum.UNKNOWN_GPU_TYPE, gpu_type)
        else:
            # if they enter a GPU name without the prefix
            if constr in VRAM_VALUES:
                vram_constraints.append(VRAM_VALUES[constr])

    if not (len(vram_constraints)):
        return pd.NA  # if no VRAM constraints are provided or no GPUs are requested return pd.NA

    return max(vram_constraints) * gpu_count


def _get_monogpu_partition_gpu_type(partition: str) -> str | None:
    """
    Get the GPU type based on the partition if it only has one type of GPU.

    This function relies on the mapping of mono-GPU partition names to their corresponding GPU types.

    Args:
        partition (str): The name of the partition (e.g., "superpod-a100", "umd-cscdr-gpu").

    Returns:
        str | None: The GPU type associated with the partition or None if no specific mapping exists.
    """
    return MONO_GPU_PARTITION_GPU_TYPE.get(partition.lower(), None)


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
    gpu_type = _get_monogpu_partition_gpu_type(partition)
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
