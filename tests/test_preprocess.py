from src import preprocess_data, fill_missing
import pandas as pd


def test_pre_process_data_fill_missing_small_interactive(small_sample_data):
    data = small_sample_data
    fill_missing(data)
    assert data["Interactive"].isnull().sum() == 0
    assert data["Interactive"].tolist() == ["non-interactive", "Matlab", "non-interactive", "Matlab"]


def test_pre_process_data_fill_missing_small_constraints(small_sample_data):
    data = small_sample_data
    fill_missing(data)
    assert data["Constraints"].isnull().sum() == 0
    assert data["Constraints"].tolist() == ["", ["some constraints"], "", ["some constraints"]]


def test_pre_process_data_fill_missing_small_GPUType(small_sample_data):
    data = small_sample_data
    fill_missing(data)
    assert data["GPUType"].isnull().sum() == 0
    assert data["GPUType"].tolist() == ["cpu", "v100", "cpu", "v100"]


def test_pre_process_data_fill_missing_small_GPUs(small_sample_data):
    data = small_sample_data
    fill_missing(data)
    assert data["GPUs"].isnull().sum() == 0
    assert data["GPUs"].tolist() == [0, 1, 0, 4]


def test_pre_process_data_fill_missing_small_arrayID(small_sample_data):
    data = small_sample_data
    fill_missing(data)
    assert data["ArrayID"].isnull().sum() == 0
    assert data["ArrayID"].tolist() == [-1, 1, 2, -1]


def test_preprocess_data_filtred_columns_total_data(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600)
    assert "UUID" not in data.columns
    assert "EndTime" not in data.columns
    assert "Nodes" not in data.columns


def test_pre_preocess_data_filtered_GPU_total_data(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600)
    GPUTypeNull = data["GPUType"].isnull()
    GPUNull = data["GPUs"].isnull()
    assert not any(GPUTypeNull)
    assert not any(GPUNull)


def test_pre_process_data_filtered_status_total_data(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600)
    statusFailed = data["Status"] == "FAILED"
    statusCancelled = data["Status"] == "CANCELLED"
    assert not any(statusFailed)
    assert not any(statusCancelled)


def test_pre_process_data_filtered_elapsed_total_data(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=300)
    elapsedLessThanMin = data["Elapsed"] < pd.to_timedelta(
        300, unit="s"
    )  # final version of Elapsed column is timedelta so convert for comparison
    assert not any(elapsedLessThanMin)


def test_pre_process_data_filtered_root_account_total_data(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600)
    partitionBuilding = data["Partition"] == "building"
    qosUpdates = data["QOS"] == "updates"
    accountRoot = data["Account"] == "root"
    assert not any(accountRoot)
    assert not any(qosUpdates)
    assert not any(partitionBuilding)


def test_pre_preprocess_data_include_CPU_job(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600, include_CPU_only_job=True)
    assert (data["GPUType"] == "cpu").sum() == 2
    assert data["GPUs"].value_counts()[0] == 2


def test_pre_process_data_include_FAILED_CANCELLED_job(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600, include_failed_cancelled_jobs=True)
    assert data["Status"].value_counts()["FAILED"] == 1
    assert data["Status"].value_counts()["CANCELLED"] == 0


def test_pre_process_data_include_all(load_mock_data_1):
    data = preprocess_data(
        data=load_mock_data_1, min_elapsed_second=600, include_failed_cancelled_jobs=True, include_CPU_only_job=True
    )
    assert len(data) == 11
    assert (data["GPUType"] == "cpu").sum() == 6
    assert data["GPUs"].value_counts()[0] == 6
    assert data["Status"].value_counts()["FAILED"] == 3
    assert data["Status"].value_counts()["CANCELLED"] == 2
    assert data["Status"].value_counts()["COMPLETED"] == 6


def test_pre_process_data_fill_missing_Interactive_mock2(load_mock_data_2):
    data = preprocess_data(
        data=load_mock_data_2, min_elapsed_second=100, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )
    interactive_stat = data["Interactive"].value_counts()
    assert interactive_stat["matlab"] == 2
    assert interactive_stat["shell"] == 1
    assert interactive_stat["jupyter"] == 1
    assert interactive_stat["non-interactive"] == 7


def test_pre_process_data_fill_missing_ArrayID_mock2(load_mock_data_2):
    data = preprocess_data(
        data=load_mock_data_2, min_elapsed_second=100, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )
    array_id_stat = data["ArrayID"].value_counts()
    assert array_id_stat[-1] == 7


def test_pre_process_data_fill_missing_GPUType_mock2(load_mock_data_2):
    data = preprocess_data(
        data=load_mock_data_2, min_elapsed_second=100, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )
    GPUs_stat = data["GPUs"].value_counts()
    assert (data["GPUType"] == "cpu").sum() == 4
    assert GPUs_stat[0] == 4


def test_pre_process_data_filter_min_esplapes_mock2(load_mock_data_2):
    data = preprocess_data(
        data=load_mock_data_2, min_elapsed_second=700, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )
    assert len(data) == 8


def test_category_interactive(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600)
    assert data["Interactive"].dtype == "category"


def test_category_QOS(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600)
    assert data["QOS"].dtype == "category"
    expected = {"short", "long", "normal"}
    assert expected.issubset(set(data["QOS"].cat.categories))


def test_category_exit_code(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600)
    assert data["ExitCode"].dtype == "category"
    expected = {"SUCCESS", "ERROR", "SIGNALED"}
    assert expected.issubset(set(data["ExitCode"].cat.categories))


def test_category_partition(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600)
    assert data["Partition"].dtype == "category"
    expected = {"building", "gpu", "cpu"}
    assert expected.issubset(set(data["Partition"].cat.categories))


def test_category_account(load_mock_data_1):
    data = preprocess_data(data=load_mock_data_1, min_elapsed_second=600)
    assert data["Account"].dtype == "category"


def test_pre_proess_timedelta_conversion(load_mock_data_1):
    data = preprocess_data(
        data=load_mock_data_1, min_elapsed_second=600, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )
    time_limit = data["TimeLimit"]
    assert time_limit.dtype == "timedelta64[ns]"
    assert time_limit[0].total_seconds() == 24000
    assert time_limit[10].total_seconds() == 480
