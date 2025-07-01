import pandas as pd
from ..config.constants import RAM_MAP, DEFAULT_MIN_ELAPSED_SECONDS, ATTRIBUTE_CATEGORIES
from ..config.enum_constants import StatusEnum, AdminsAccountEnum, PartitionEnum, QOSEnum


def get_requested_vram(constraints: list[str]) -> int:
    requested_vrams = []
    for constr in constraints:
        constr = constr.strip("'")
        if constr.startswith("vram"):
            requested_vrams.append(int(constr.replace("vram", "")))
        elif constr.startswith("gpu"):
            gpu_type = constr.split(":")[1]
            requested_vrams.append(RAM_MAP[gpu_type])
    if not (len(requested_vrams)):
        return 0
    return max(requested_vrams)


def get_allocated_vram(gpu_type: list[str]) -> int:
    return min(RAM_MAP[x] for x in gpu_type)


def _fill_missing(res: pd.DataFrame) -> None:
    """
    Intended for internal use inside preprocess_data() only. Fill missing values in the DataFrame with default values.

    Args:
        res (pd.DataFrame): The DataFrame to fill missing values in.

    Returns:
        None: The function modifies the DataFrame in place.
    """

    res.loc[:, "ArrayID"] = res["ArrayID"].fillna(-1)
    res.loc[:, "Interactive"] = res["Interactive"].fillna("non-interactive")
    res.loc[:, "Constraints"]= res["Constraints"].fillna("").apply(lambda x: [] if x == "" else x)
    res.loc[:, "GPUType"] = res["GPUType"].fillna("").apply(lambda x: ["cpu"] if x == "" else x)
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

    # Added parameters, similar to Benjamin code
    res.loc[:, "Queued"] = res["StartTime"] - res["SubmitTime"]
    res.loc[:, "requested_vram"] = res["Constraints"].apply(lambda c: get_requested_vram(c))
    res.loc[:, "allocated_vram"] = res["GPUType"].apply(lambda x: get_allocated_vram(x))
    res.loc[:, "user_jobs"] = res.groupby("User")["User"].transform("size")
    res.loc[:, "account_jobs"] = res.groupby("Account")["Account"].transform("size")

    # convert columns to categorical

    for col, enum_obj in ATTRIBUTE_CATEGORIES.items():
        enum_values = [e.value for e in enum_obj]
        unique_values = res[col].unique().tolist()
        all_categories = list(set(enum_values) | set(unique_values))
        res[col] = pd.Categorical(res[col], categories=all_categories, ordered=False)
    return res
