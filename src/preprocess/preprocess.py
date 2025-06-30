import pandas as pd
import numpy as np
from ..config.constants import (
    RAM_MAP,
    DEFAULT_MIN_ELAPSED_SECONDS,
    ATTRIBUTE_CATEGORIES,
    VARIABLE_GPUS,
    GET_VRAM_FROM_NODE,
)
from ..config.enum_constants import StatusEnum, AdminsAccountEnum, PartitionEnum, QOSEnum


def get_requested_vram(constraints: list[str], num_gpus: int, gpu_mem_usage: int) -> int:
    """Get the requested VRAM for a job based on its constraints and GPU usage.
    This function extracts VRAM requests from the job constraints and returns the maximum requested VRAM.

    Args:
        constraints (list[str]): List of constraints from the job, which may include VRAM requests.
        num_gpus (int): Number of GPUs requested by the job.
        gpu_mem_usage (int): GPU memory usage in bytes.

    Returns:
        int: Maximum requested VRAM in GB for the job, multiplied by the number of GPUs.

    """
    gpu_mem_usage_gb = gpu_mem_usage / (2**30)
    requested_vrams = []
    for constr in constraints:
        constr = constr.strip("'")
        if constr.startswith("vram"):
            requested_vrams.append(int(constr.replace("vram", "")))
        elif constr.startswith("gpu"):
            gpu_type = constr.split(":")[1]
            if gpu_type in VARIABLE_GPUS and gpu_mem_usage_gb > RAM_MAP[gpu_type] * num_gpus:
                # assume they want the maximum vram for this GPU type if their usage exceeds the default
                # for a V100, we assume they want 16GB by default
                requested_vrams.append(max(VARIABLE_GPUS[gpu_type]))
            else:
                requested_vrams.append(RAM_MAP[gpu_type])

    if not (len(requested_vrams)):
        return 0

    return max(requested_vrams) * num_gpus


def get_estimated_allocated_vram(gpu_type: list[str], node_list: list[str], num_gpus: int) -> int:
    """
    Get the total allocated VRAM for a job based on its GPU type and node list.
    This function estimates the total VRAM allocated for a job based on the GPU types used
    and the nodes where the job ran.

    Args:
        gpu_type (list[str]): List of GPU types used in the job.
        node_list (list[str]): List of nodes where the job ran.
        num_gpus (int): Number of GPUs requested by the job.

    Returns:
        int: Total allocated (estimate) VRAM for the job.
    """

    # single GPU
    if len(gpu_type) == 1:
        gpu = gpu_type[0]
        if gpu in VARIABLE_GPUS:
            total_vram = 0
            if len(node_list) > 1:
                for node in node_list:
                    total_vram += GET_VRAM_FROM_NODE[gpu](node)
            else:
                node = node_list[0]
                total_vram = GET_VRAM_FROM_NODE[gpu](node) * num_gpus

            return total_vram
        else:
            return RAM_MAP[gpu] * num_gpus

    # add VRAM for multiple distinct GPUs
    if len(gpu_type) == num_gpus:
        total_vram = 0
        for gpu in gpu_type:
            if gpu in VARIABLE_GPUS:
                for node in node_list:
                    total_vram += GET_VRAM_FROM_NODE[gpu](node)
            else:
                total_vram += RAM_MAP[gpu]
        return total_vram

    # estimate VRAM for multiple GPUs where exact number isn't known
    allocated_vrams = []
    for gpu in gpu_type:
        if gpu in VARIABLE_GPUS:
            for node in node_list:
                allocated_vrams.append(GET_VRAM_FROM_NODE[gpu](node))
        else:
            allocated_vrams.append(RAM_MAP[gpu])

    return min(allocated_vrams) * num_gpus


def _fill_missing(res: pd.DataFrame) -> None:
    """
    Intended for internal use inside preprocess_data() only. Fill missing values in the DataFrame with default values.

    Args:
        res (pd.DataFrame): The DataFrame to fill missing values in.

    Returns:
        None: The function modifies the DataFrame in place.
    """

    #! all Nan value are np.nan
    # fill default values for specific columns
    res.loc[:, "ArrayID"] = res["ArrayID"].fillna(-1)
    res.loc[:, "Interactive"] = res["Interactive"].fillna("non-interactive")
    res.loc[:, "Constraints"] = res["Constraints"].fillna("").apply(lambda x: np.array(list(x)))
    res.loc[:, "GPUType"] = (
        res["GPUType"].fillna("").apply(lambda x: np.array(["cpu"]) if isinstance(x, str) else np.array(x))
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

    data = input_df.drop(columns=["UUID", "EndTime", "Nodes"], axis=1, inplace=False)

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

    # Added parameters, similar to Benjamin code
    res.loc[:, "Queued"] = res["StartTime"] - res["SubmitTime"]
    res.loc[:, "requested_vram"] = res.apply(
        lambda row: get_requested_vram(row["Constraints"], row["GPUs"], row["GPUMemUsage"]), axis=1
    )
    res.loc[:, "allocated_vram"] = res.apply(
        lambda row: get_estimated_allocated_vram(row["GPUType"], row["NodeList"], row["GPUs"]), axis=1
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
