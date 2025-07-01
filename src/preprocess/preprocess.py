import pandas as pd
import numpy as np
from ..config.constants import (
    VRAM_VALUES,
    DEFAULT_MIN_ELAPSED_SECONDS,
    ATTRIBUTE_CATEGORIES,
    MULTIVALENT_GPUS,
    GET_VRAM_FROM_NODE,
)
from ..config.enum_constants import StatusEnum, AdminsAccountEnum, PartitionEnum, QOSEnum


def get_vram_constraints(constraints: list[str], gpu_count: int, gpu_mem_usage: int) -> int | None:
    """Get the VRAM assigned for a job based on its constraints and GPU usage.
    This function extracts VRAM requests from the job constraints and returns the maximum requested VRAM from the
    constraints. For GPU names that correspond to multiple VRAM values, take the minimum value that is not smaller
    than the amount of VRAM used by that job.

    Args:
        constraints (list[str]): List of constraints from the job, which may include VRAM requests.
        gpu_count (int): Number of GPUs requested by the job.
        gpu_mem_usage (int): GPU memory usage in bytes.

    Returns:
        int | None: Maximum requested VRAM in GiB for the job, multiplied by the number of GPUs.
        Returns None if no VRAM is requested.

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


def get_approx_allocated_vram(gpu_type: list[str], node_list: list[str], gpu_count: int, gpu_mem_usage: int) -> int:
    """
    Get the total allocated VRAM for a job based on its GPU type and node list.
    This function estimates the total VRAM allocated for a job based on the GPU types used
    and the nodes where the job ran.

    Args:
        gpu_type (list[str]): List of GPU types used in the job.
        node_list (list[str]): List of nodes where the job ran.
        gpu_count (int): Number of GPUs requested by the job.
        gpu_mem_usage (int): GPU memory usage in bytes.

    Returns:
        int: Total allocated (estimate) VRAM for the job in GiB (gibibyte).
    """

    # one type of GPU
    if len(gpu_type) == 1:
        gpu = gpu_type[0]
        if gpu not in MULTIVALENT_GPUS:
            # if the GPU is not multivalent, return the VRAM value for that GPU
            return VRAM_VALUES[gpu] * gpu_count
        
        # calculate VRAM for multivalent GPUs
        total_vram = 0
        if len(node_list) == 1:
            node = node_list[0]
            total_vram = GET_VRAM_FROM_NODE[gpu](node) * gpu_count

        elif len(node_list) == gpu_count:
            for node in node_list:
                total_vram += GET_VRAM_FROM_NODE[gpu](node)

        else:
            # calculate all VRAM for all nodes in the node_list
            node_values = set() # to avoid duplicates
            for node in node_list:
                node_values.add(GET_VRAM_FROM_NODE[gpu](node))
            
            if not node_values:
                return None
            
            node_values = sorted(list(node_values))
            total_vram = node_values.pop(0) * gpu_count # use the node with the minimum VRAM value
            # if the total VRAM is less than the GPU memory usage, use the VRAM from the GPU in the larger ndoe 
            while total_vram < (gpu_mem_usage / 2**30) and node_values:
                total_vram = node_values.pop(0) * gpu_count

        return total_vram

    # add VRAM for multiple distinct GPUs
    if len(gpu_type) == gpu_count:
        total_vram = 0
        for gpu in gpu_type:
            if gpu in MULTIVALENT_GPUS:
                for node in node_list:
                    total_vram += GET_VRAM_FROM_NODE[gpu](node)
            else:
                total_vram += VRAM_VALUES[gpu]
        return total_vram

    # estimate VRAM for multiple GPUs where exact number isn't known
    # TODO: update this based on the updated GPU types which specify exact number of GPUs
    allocated_vrams = []
    for gpu in gpu_type:
        if gpu in MULTIVALENT_GPUS:
            for node in node_list:
                allocated_vrams.append(GET_VRAM_FROM_NODE[gpu](node))
        else:
            allocated_vrams.append(VRAM_VALUES[gpu])

    return min(allocated_vrams) * gpu_count


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
    res.loc[:, "Constraints"] = res["Constraints"].fillna("").apply(lambda x: np.array(list(x)))
    res.loc[:, "GPUType"] = (
        res["GPUType"].fillna("").apply(lambda x: np.array(["cpu"]) if x == "" else np.array(x))
    )
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
