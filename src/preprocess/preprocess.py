import pandas as pd
import numpy as np
import re

from ..config.constants import (
    VRAM_VALUES,
    DEFAULT_MIN_ELAPSED_SECONDS,
    ATTRIBUTE_CATEGORIES,
    MULTIVALENT_GPUS,
)
from ..config.enum_constants import StatusEnum, AdminsAccountEnum, PartitionEnum, QOSEnum


def get_vram_from_node(gpu_type: str, node: str) -> int:
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

        TODO: Consider reading this information from a config file or database.
    """
    if gpu_type not in MULTIVALENT_GPUS:
        # if the GPU is not multivalent we do not need to check the node
        return 0

    if gpu_type == "a100":
        if node.startswith("ece-gpu"):
            return 40
        elif re.match("^(gpu0(1[3-9]|2[0-4]))|(gpu042)|(umd-cscdr-gpu00[1-2])|(uri-gpu00[1-8])$", node):
            return 80
        else:
            # if the node does not match any of the patterns, it is not a valid node for this GPU type
            # so we return 0
            return 0
    elif gpu_type == "v100":
        if re.match("^(gpu00[1-7])|(power9-gpu009)|(power9-gpu01[0-6])$", node):
            return 16
        elif re.match("^(gpu01[1-2])|(power9-gpu00[1-8])$", node):
            return 32
        else:
            # if the node does not match any of the patterns, it is not a valid node for this GPU type
            # so we return 0
            return 0


def get_vram_constraints(constraints: list[str], gpu_count: int, gpu_mem_usage: int) -> int | None:
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
        int | None: Maximum VRAM amount in GiB obtained based on the provided constraints, multiplied by the
                    number of GPUs. Returns None if no VRAM constraints are present.

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
        return None  # if nothing is requested, return None

    return max(vram_constraints) * gpu_count


def get_approx_allocated_vram(gpu_types: list[str], node_list: list[str], gpu_count: int, gpu_mem_usage: int) -> int:
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
        gpu = gpu_types[0]
        if gpu not in MULTIVALENT_GPUS:
            # if the GPU is not multivalent, return the VRAM value for that GPU
            return VRAM_VALUES[gpu] * gpu_count

        # calculate VRAM for multivalent GPUs
        total_vram = 0

        # if all GPUs are on the same node, multiply the VRAM of that node by the number of GPUs
        if len(node_list) == 1:
            node = node_list[0]
            total_vram = get_vram_from_node(gpu, node) * gpu_count

        # if all GPUs are on different nodes, sum the VRAM of each node
        # and return the total VRAM
        elif len(node_list) == gpu_count:
            for node in node_list:
                total_vram += get_vram_from_node(gpu, node)

        # if there are multiple nodes, but not all GPUs are on different nodes
        # we need to calculate the total VRAM based on the minimum VRAM of the nodes
        else:
            # calculate all VRAM for all nodes in the node_list
            node_values = set()  # to avoid duplicates
            for node in node_list:
                node_values.add(get_vram_from_node(gpu, node))

            node_values = sorted(list(node_values))
            total_vram = node_values.pop(0) * gpu_count  # use the node with the minimum VRAM value
            # if the total VRAM is less than the GPU memory usage, use the VRAM from the GPU in the larger node
            while total_vram < (gpu_mem_usage / 2**30) and node_values:
                total_vram = node_values.pop(0) * gpu_count

        return total_vram

    # Calculate allocated VRAM when there are multiple GPU types in a job
    if len(gpu_types) == gpu_count:
        total_vram = 0
        for gpu in gpu_types:
            if gpu in MULTIVALENT_GPUS:
                for node in node_list:
                    total_vram += get_vram_from_node(gpu, node)
            else:
                total_vram += VRAM_VALUES[gpu]
        return total_vram

    # estimate VRAM for multiple GPUs where exact number isn't known
    # TODO: update this based on the updated GPU types which specify exact number of GPUs
    allocated_vrams = []
    for gpu in gpu_types:
        if gpu in MULTIVALENT_GPUS:
            for node in node_list:
                allocated_vrams.append(get_vram_from_node(gpu, node))
        else:
            allocated_vrams.append(VRAM_VALUES[gpu])

    vram_values = sorted(list(allocated_vrams))
    total_vram = vram_values.pop(0) * gpu_count  # use the GPU with the minimum VRAM value
    # if the total VRAM is less than the GPU memory usage, use the VRAM from the next smallest GPU
    while total_vram < (gpu_mem_usage / 2**30) and vram_values:
        total_vram = node_values.pop(0) * gpu_count
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
    mask_constraints_null = res["Constraints"].isna()
    res.loc[mask_constraints_null, "Constraints"] = res.loc[mask_constraints_null, "Constraints"].apply(
        lambda _: np.array([])
    )
    res.loc[:, "GPUType"] = res["GPUType"].fillna("").apply(lambda x: np.array(["cpu"]) if x == "" else np.array(x))
    res.loc[:, "GPUs"] = res["GPUs"].fillna(0)


def preprocess_data(
    input_df: pd.DataFrame,
    min_elapsed_seconds: int = DEFAULT_MIN_ELAPSED_SECONDS,
    include_failed_cancelled_jobs=False,
    include_cpu_only_jobs=False,
) -> pd.DataFrame:
    """
    Preprocess dataframe, filtering out unwanted rows and columns, filling missing values and converting types.
    
    This function will take in a dataframe and create a new dataframe satisfying criterias,
    original dataframe is intact.

    Args:
        data (pd.DataFrame): The input dataframe containing job data.
        min_elapsed_seconds (int, optional): Minimum elapsed time in seconds to keep a job record. Defaults to 600.
        include_failed_cancelled_jobs (bool, optional): Whether to include jobs with status FAILED or CANCELLED.
        include_cpu_only_jobs (bool, optional): Whether to include jobs that do not use GPUs (CPU-only jobs).

    Returns:
        pd.DataFrame: The preprocessed dataframe
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
    # type casting for columns involving time
    time_columns = ["StartTime", "SubmitTime"]
    for col in time_columns:
        res[col] = pd.to_datetime(res[col], errors="coerce")

    timedelta_columns = ["TimeLimit", "Elapsed"]
    for col in timedelta_columns:
        res[col] = pd.to_timedelta(res[col], unit="s", errors="coerce")

    # Added parameters for calculating VRAM metrics
    res.loc[:, "Queued"] = res["StartTime"] - res["SubmitTime"]
    res.loc[:, "requested_vram"] = res.apply(
        lambda row: get_vram_constraints(row["Constraints"], row["GPUs"], row["GPUMemUsage"]), axis=1
    )
    res.loc[:, "allocated_vram"] = res.apply(
        lambda row: get_approx_allocated_vram(row["GPUType"], row["NodeList"], row["GPUs"], row["GPUMemUsage"]), axis=1
    )
    res.loc[:, "user_jobs"] = res.groupby("User")["User"].transform("size")
    res.loc[:, "account_jobs"] = res.groupby("Account")["Account"].transform("size")

    # convert columns to categorical

    for col, enum_obj in ATTRIBUTE_CATEGORIES.items():
        enum_values = [e.value for e in enum_obj]
        unique_values = res[col].unique().tolist()
        all_categories = list(set(enum_values) | set(unique_values))
        res[col] = pd.Categorical(res[col], categories=all_categories, ordered=False)
    return res
