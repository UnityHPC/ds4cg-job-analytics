"""

The following describes some preprocessing criteria for the job data:

- Attributes omitted in preprocessing the provided data:
UUID
Nodes : NodesList have more specific information
EndTime : can be calculated from StartTime and Elapsed

- Providing the option of including or omitting certain types of jobs:
  - Keeping CPU jobs:
      - If GPUType is null, the value will be filled with "CPU"
      - If GPUs is null or is 0, the value will be 0.
- Keeping jobs where the status is "Failed" or "Cancelled"


- Records with the following conditions are omitted:
   - Elapsed is less than the minimum threshold
   - account is root
   - partion is building
   - QOS is updates

- Null attributes in the data are filled with default values as described below:
  - ArrayID is set to -1
  - Interactive is set to "non-interactive"
  - Constraints is set to an empty numpy array
  - GPUs is be set to 0 (when CPU jobs are kept)
  - GPUType is set to an empty numpy array (when CPU jobs are kept)

- Type of the following attributes is be set as `datetime`: StartTime, SubmitTime
- Type of the following attributes is be set as `timedelta`: TimeLimit, Elapsed
- Type of the following attributes is set as `Categorical`: Interactive, Status, ExitCode, QOS, Partition, Account

"""

import pandas as pd
import numpy as np
from ..config.constants import RAM_MAP, DEFAULT_MIN_ELAPSED_SECONDS, ATTRIBUTE_CATEGORIES
from ..config.enum_constants import StatusEnum


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

    mask_constraints_null = res["Constraints"].isnull()
    res.loc[mask_constraints_null, "Constraints"] = res.loc[mask_constraints_null, "Constraints"].apply(
        lambda _: np.array([])
    )

    mask_GPUType_null = res["GPUType"].isnull()
    res.loc[mask_GPUType_null, "GPUType"] = res.loc[mask_GPUType_null, "GPUType"].apply(lambda _: np.array(["cpu"]))

    res.loc[:, "GPUs"] = res["GPUs"].fillna(0)


def preprocess_data(
    input_df: pd.DataFrame,
    min_elapsed_seconds: int = DEFAULT_MIN_ELAPSED_SECONDS,
    include_failed_cancelled_jobs=False,
    include_CPU_only_jobs=False,
) -> pd.DataFrame:
    """
    Preprocess dataframe, filtering out unwanted rows and columns, filling missing values and converting types.
    This function will take in a dataframe and create a new dataframe satisfying criterias,
    original dataframe is intact.

    Args:
        data (pd.DataFrame): The input dataframe containing job data.
        min_elapsed_seconds (int, optional): Minimum elapsed time in seconds to keep a job record. Defaults to 600.
        include_failed_cancelled_jobs (bool, optional): Whether to include jobs with status FAILED or CANCELLED.
        include_CPU_only_jobs (bool, optional): Whether to include jobs that do not use GPUs (CPU-only jobs).

    Returns:
        pd.DataFrame: The preprocessed dataframe
    """

    data = input_df.drop(columns=["UUID", "EndTime", "Nodes"], axis=1, inplace=False)

    cond_GPU_Type = (
        data["GPUType"].notnull() | include_CPU_only_jobs
    )  # filter out GPUType is null, except when include_CPU_only_job is True
    cond_GPUs = (
        data["GPUs"].notnull() | include_CPU_only_jobs
    )  # filter out GPUs is null, except when include_CPU_only_job is True
    cond_failed_cancelled_jobs = (
        ((data["Status"] != StatusEnum.FAILED.value) & (data["Status"] != StatusEnum.CANCELLED.value))
        | include_failed_cancelled_jobs
    )  # filter out failed or cancelled jobs, except when include_fail_cancel_jobs is True

    res = data[
        cond_GPU_Type
        & cond_GPUs
        & cond_failed_cancelled_jobs
        & (data["Elapsed"] >= min_elapsed_seconds)  # filter in unit of second, not timedelta object
        & (data["Account"] != "root")
        & (data["Partition"] != "building")
        & (data["QOS"] != "updates")
    ].copy()

    _fill_missing(res)
    #! type casting for columns involving time
    time_columns = ["StartTime", "SubmitTime"]
    for col in time_columns:
        res[col] = pd.to_datetime(res[col], errors="coerce")

    timedelta_columns = ["TimeLimit", "Elapsed"]
    for col in timedelta_columns:
        res[col] = pd.to_timedelta(res[col], unit="s", errors="coerce")

    #!Added parameters, similar to Benjamin code
    res.loc[:, "Queued"] = res["StartTime"] - res["SubmitTime"]
    res.loc[:, "requested_vram"] = res["Constraints"].apply(lambda c: get_requested_vram(c))
    res.loc[:, "allocated_vram"] = res["GPUType"].apply(lambda x: get_allocated_vram(x))
    res.loc[:, "user_jobs"] = res.groupby("User")["User"].transform("size")
    res.loc[:, "account_jobs"] = res.groupby("Account")["Account"].transform("size")

    #! convert columns to categorical

    for col, enum_obj in ATTRIBUTE_CATEGORIES.items():
        enum_values = [e.value for e in enum_obj]
        unique_values = res[col].unique().tolist()
        all_categories = list(set(enum_values) | set(unique_values))
        res[col] = pd.Categorical(res[col], categories=all_categories, ordered=False)
    return res
