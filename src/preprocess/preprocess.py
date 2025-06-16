"""
Ignore columns:
UUID

add optional parameter for keeping/ deleting columns:
If include:
GPUType is null, keep this and replace as CPU
GPUs is null or is 0, keep this replace as 0
status os Failed or Cancelled, don't drop this

ignore record which has:
Elapsed < min_elapsed
account is root
partion is building
QOS is updates
Need to fill in missing values for arrayID, interactive, and constraints

Type of starttime, endtime, submit time will be converted into datetime
Typoe of Queued Time, TimeLimit, Elapsed will be converted into timedelta

"""

import pandas as pd

#! add requested_vram, allocated_vram, user_jobs, account_jobs

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


# TODO: check if Elapsed Time is always = StartTime - EndTime
def get_requested_vram(constraints: list[str] | str) -> int:
    if isinstance(constraints, str) and constraints == "":
        return 0
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
    else:
        return min(requested_vrams)


def get_allocated_vram(gpu_type: list[str] | str) -> int:
    if isinstance(gpu_type, str):
        return 0
    else:
        return min(ram_map[x] for x in gpu_type)


def fill_missing(res: pd.DataFrame) -> None:
    # modify object inplace
    #! all Nan value are np.nan
    #!important note: not filled the null values of constraints since Benjaminc code already handled it
    res.loc[:, "ArrayID"] = res["ArrayID"].fillna(-1)  # null then fills -1
    res.loc[:, "Interactive"] = res["Interactive"].fillna("non-interactive")
    res.loc[:, "Constraints"] = res["Constraints"].fillna("")  # null then fills empty list
    if "GPUType" in res.columns:
        res.loc[:, "GPUType"] = res["GPUType"].fillna("cpu")  # Nan in GPUType is CPU
    if "GPUs" in res.columns:
        res.loc[:, "GPUs"] = res["GPUs"].fillna(0)  # Nan in GPUs is 0


def preprocess_data(
    data: pd.DataFrame, min_elapsed_second: int = 600, include_failed_cancelled_jobs=False, include_CPU_only_job=False
) -> pd.DataFrame:
    data.drop(columns=["UUID"], axis=1, inplace=True)

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
    ]

    # type casting for columns involving time
    time_columns = ["StartTime", "EndTime", "SubmitTime"]
    for col in time_columns:
        res.loc[:, col] = pd.to_datetime(res[col], errors="coerce")

    timedelta_columns = ["TimeLimit", "Elapsed"]
    for col in timedelta_columns:
        res.loc[:, col] = pd.to_timedelta(res[col], unit="s", errors="coerce")

    #!Added parameters, similar to Benjamin code
    res["Queued"] = res["StartTime"] - res["SubmitTime"]
    res["requested_vram"] = res["Constraints"].apply(lambda c: get_requested_vram(c))
    # res["allocated_vram"] = res["GPUType"].apply(lambda x: get_allocated_vram(x))
    # res["user_jobs"] = res.groupby("User")["User"].transform("size")
    # res["account_jobs"] = res.groupby("Account")["Account"].transform("size")
    # res["queued_seconds"] = res["Queued"].apply(lambda x: x.total_seconds())
    # res["total_seconds"] = res["Elapsed"] + res["queued_seconds"]
    fill_missing(res)

    return res
