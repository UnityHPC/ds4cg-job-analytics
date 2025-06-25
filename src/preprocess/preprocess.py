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

# TODO: convert the partition to Categorical type

import pandas as pd
import numpy as np


ram_map = {
    "a100": 80,
    "v100": 16,
    "a40": 48,
    "gh200": 95,
    "rtx_8000": 48,
    "2080_ti": 11,
    "1080_ti": 11,
    "2080": 8,
    "h100": 80,
    "l4": 23,
    "m40": 23,
    "l40s": 48,
    "titan_x": 12,
    "a16": 16,
    "cpu": 0,
}

CATEGORY_PARTITION = [
    "building",
    "arm-gpu",
    "arm-preempt",
    "cpu",
    "cpu-preempt",
    "gpu",
    "gpu-preempt",
    "power9",
    "power9-gpu",
    "power9-gpu-preempt",
    "astroth-cpu",
    "astroth-gpu",
    "cbio-cpu",
    "cbio-gpu",
    "ceewater_casey-cpu",
    "ceewater_cjgleason-cpu",
    "ceewater_kandread-cpu",
    "ece-gpu",
    "fsi-lab",
    "gaoseismolab-cpu",
    "superpod-a100",
    "gpupod-l40s",
    "ials-gpu",
    "jdelhommelle",
    "lan",
    "mpi",
    "power9-gpu-osg",
    "toltec-cpu",
    "umd-cscdr-arm",
    "umd-cscdr-cpu",
    "umd-cscdr-gpu",
    "uri-cpu",
    "uri-gpu",
    "uri-richamp",
    "visterra",
    "zhoulin-cpu",
]

CATEGORY_GPUTYPE = [
    "titanx",
    "m40",
    "1080ti",
    "v100",
    "2080",
    "2080ti",
    "rtx8000",
    "a100-40g",
    "a100-80g",
    "a16",
    "a40",
    "gh200",
    "l40s",
    "l4",
]


def get_requested_vram(constraints: list[str]) -> int:
    requested_vrams = []
    for constr in constraints:
        constr = constr.strip("'")
        if constr.startswith("vram"):
            requested_vrams.append(int(constr.replace("vram", "")))
        elif constr.startswith("gpu"):
            gpu_type = constr.split(":")[1]
            requested_vrams.append(ram_map[gpu_type])
    if not (len(requested_vrams)):
        return 0
    return max(requested_vrams)


# TODO: for a100 and v100, look at the nodes for vram computation; for multiple GPUTypes, return min for now
def get_allocated_vram(gpu_type: list[str]) -> int:
    return min(ram_map[x] for x in gpu_type)


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
    min_elapsed_second: int = 600,
    include_failed_cancelled_jobs=False,
    include_CPU_only_job=False,
) -> pd.DataFrame:
    """
    Preprocess dataframe, filtering out unwanted rows and columns, filling missing values and converting types.
    This function will take in a dataframe and create a new dataframe satisfying criterias,
    original dataframe is intact.

    Args:
        data (pd.DataFrame): The input dataframe containing job data.
        min_elapsed_second (int, optional): Minimum elapsed time in seconds to keep a job record. Defaults to 600.
        include_failed_cancelled_jobs (bool, optional): Whether to include jobs with status FAILED or CANCELLED.
        include_CPU_only_job (bool, optional): Whether to include jobs that do not use GPUs (CPU-only jobs).

    Returns:
        pd.DataFrame: The preprocessed dataframe
    """

    data = input_df.drop(columns=["UUID", "EndTime", "Nodes"], axis=1, inplace=False)

    cond_GPU_Type = (
        data["GPUType"].notnull() | include_CPU_only_job
    )  # filter out GPUType is null, except when include_CPU_only_job is True
    cond_GPUs = (
        data["GPUs"].notnull() | include_CPU_only_job
    )  # filter out GPUs is null, except when include_CPU_only_job is True
    cond_failed_cancelled_jobs = (
        ((data["Status"] != "FAILED") & (data["Status"] != "CANCELLED")) | include_failed_cancelled_jobs
    )  # filter out failed or cancelled jobs, except when include_fail_cancel_jobs is True

    res = data[
        cond_GPU_Type
        & cond_GPUs
        & cond_failed_cancelled_jobs
        & (data["Elapsed"] >= min_elapsed_second)  # filter in unit of second, not timedelta object
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
    # a map from columns to some of its possible values, any values not in the map will be added automatically
    custom_category_map = {
        "Interactive": ["non-interactive", "shell"],
        "QOS": ["normal", "updates", "short", "long"],
        "Status": [
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "PREEMPTED",
            "OUT_OF_MEMORY",
            "PENDING",
            "NODE_FAIL",
        ],
        "ExitCode": ["SUCCESS", "ERROR", "SIGNALED"],
        "Account": ["root"],
    }
    for col in custom_category_map:
        all_categories = list(set(custom_category_map[col] + list(res[col].unique())))
        res[col] = pd.Categorical(res[col], categories=all_categories, ordered=False)

    #! some category that we have access to all possible values on Unity documentation
    res["Partition"] = pd.Categorical(res["Partition"], categories=CATEGORY_PARTITION, ordered=False)

    return res
