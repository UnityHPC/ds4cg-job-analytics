from src.preprocess import preprocess_data, _fill_missing
import pandas as pd


def _helper_filter_irrelevants_record(input_df: pd.DataFrame, min_elapsed: int) -> pd.DataFrame:
    """
    Private function to help generate expected ground truth dataframe for test.
    Given a ground truth dataframe, this will create a new dataframe without records meeting the following criteria:
    - QOS is updates
    - Account is root
    - Partition is building
    - Elasped time is less than min_elapsed

    Args:
        input_df (pd.DataFrame): Input dataframe to filter. Note that the Elapsed field should be in unit seconds.
        min_elapsed (int): Minimum elapsed time in seconds.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """

    res = input_df[
        (input_df["Elapsed"] >= min_elapsed)
        & (input_df["Account"] != "root")
        & (input_df["Partition"] != "building")
        & (input_df["QOS"] != "updates")
    ]

    return res


def test_pre_process_data_fill_missing_small_interactive(small_sample_data):
    data = small_sample_data
    _fill_missing(data)
    assert data["Interactive"].isnull().sum() == 0
    assert data["Interactive"].tolist() == ["non-interactive", "Matlab", "non-interactive", "Matlab"]


def test_pre_process_data_fill_missing_small_constraints(small_sample_data):
    data = small_sample_data
    _fill_missing(data)
    assert data["Constraints"].isnull().sum() == 0
    temp = data["Constraints"].tolist()
    assert temp[1] == temp[3] == ["some constraints"]
    assert len(temp[0]) == len(temp[2]) == 0


def test_pre_process_data_fill_missing_small_GPUType(small_sample_data):
    data = small_sample_data
    _fill_missing(data)
    assert data["GPUType"].isnull().sum() == 0
    print(data["GPUType"].tolist() == [["cpu"], ["v100"], ["cpu"], ["v100"]])


def test_pre_process_data_fill_missing_small_GPUs(small_sample_data):
    data = small_sample_data
    _fill_missing(data)
    assert data["GPUs"].isnull().sum() == 0
    assert data["GPUs"].tolist() == [0, 1, 0, 4]


def test_pre_process_data_fill_missing_small_arrayID(small_sample_data):
    data = small_sample_data
    _fill_missing(data)
    assert data["ArrayID"].isnull().sum() == 0
    assert data["ArrayID"].tolist() == [-1, 1, 2, -1]


def test_pre_process_data_filtred_columns(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600)
    assert "UUID" not in data.columns
    assert "EndTime" not in data.columns
    assert "Nodes" not in data.columns


def test_pre_pre_process_data_filtered_GPU(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600)
    GPUTypeNull = data["GPUType"].isnull()
    GPUNull = data["GPUs"].isnull()
    assert not any(GPUTypeNull)
    assert not any(GPUNull)


def test_pre_process_data_filtered_status(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600)
    statusFailed = data["Status"] == "FAILED"
    statusCancelled = data["Status"] == "CANCELLED"
    assert not any(statusFailed)
    assert not any(statusCancelled)


def test_pre_process_data_filtered_elapsed(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=300)
    elapsedLessThanMin = data["Elapsed"] < pd.to_timedelta(
        300, unit="s"
    )  # final version of Elapsed column is timedelta so convert for comparison
    assert not any(elapsedLessThanMin)


def test_pre_process_data_filtered_root_account(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600)
    partitionBuilding = data["Partition"] == "building"
    qosUpdates = data["QOS"] == "updates"
    accountRoot = data["Account"] == "root"
    assert not any(accountRoot)
    assert not any(qosUpdates)
    assert not any(partitionBuilding)


def test_pre_preprocess_data_include_CPU_job(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600, include_CPU_only_job=True)
    ground_truth = _helper_filter_irrelevants_record(load_mock_data, 600)
    expected_cpu_type = len(
        ground_truth[
            ground_truth["GPUType"].isnull()
            & (ground_truth["Status"] != "FAILED")
            & (ground_truth["Status"] != "CANCELLED")
        ]
    )
    expected_gpus_count_0 = len(
        ground_truth[
            ground_truth["GPUs"].isnull()
            & (ground_truth["Status"] != "FAILED")
            & (ground_truth["Status"] != "CANCELLED")
        ]
    )
    assert sum(x == ["cpu"] for x in data["GPUType"]) == expected_cpu_type
    assert data["GPUs"].value_counts()[0] == expected_gpus_count_0


def test_pre_process_data_include_FAILED_CANCELLED_job(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600, include_failed_cancelled_jobs=True)
    ground_truth = _helper_filter_irrelevants_record(load_mock_data, 600)
    expect_failed_status = len(
        ground_truth[
            (ground_truth["Status"] == "FAILED")
            & (ground_truth["GPUType"].notnull())
            & (ground_truth["GPUs"].notnull())
        ]
    )
    expect_cancelled_status = len(
        ground_truth[
            (ground_truth["Status"] == "CANCELLED")
            & (ground_truth["GPUType"].notnull())
            & (ground_truth["GPUs"].notnull())
        ]
    )
    assert data["Status"].value_counts()["FAILED"] == expect_failed_status
    assert data["Status"].value_counts()["CANCELLED"] == expect_cancelled_status


def test_pre_process_data_include_all(load_mock_data):
    data = preprocess_data(
        input_df=load_mock_data, min_elapsed_second=600, include_failed_cancelled_jobs=True, include_CPU_only_job=True
    )
    ground_truth = _helper_filter_irrelevants_record(load_mock_data, 600)

    expect_failed_status = len(ground_truth[(ground_truth["Status"] == "FAILED")])
    expect_cancelled_status = len(ground_truth[(ground_truth["Status"] == "CANCELLED")])
    expect_completed_status = len(ground_truth[(ground_truth["Status"] == "COMPLETED")])
    expect_GPUType_null = len(ground_truth[(ground_truth["GPUType"].isnull())])
    expect_GPUs_null = len(ground_truth[(ground_truth["GPUs"].isnull())])

    assert len(data) == len(ground_truth)
    assert sum(x == ["cpu"] for x in data["GPUType"]) == expect_GPUType_null
    assert data["GPUs"].value_counts()[0] == expect_GPUs_null
    assert data["Status"].value_counts()["FAILED"] == expect_failed_status
    assert data["Status"].value_counts()["CANCELLED"] == expect_cancelled_status
    assert data["Status"].value_counts()["COMPLETED"] == expect_completed_status


def test_pre_process_data_fill_missing_Interactive(load_mock_data):
    data = preprocess_data(
        input_df=load_mock_data, min_elapsed_second=100, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )
    ground_truth = _helper_filter_irrelevants_record(load_mock_data, 100)

    expect_matlab = len(ground_truth[(ground_truth["Interactive"] == "matlab")])
    expect_shell = len(ground_truth[(ground_truth["Interactive"] == "shell")])
    expect_jupyter = len(ground_truth[(ground_truth["Interactive"] == "jupyter")])
    expect_non_interactive = len(ground_truth[(ground_truth["Interactive"].isnull())])

    interactive_stat = data["Interactive"].value_counts()
    assert interactive_stat["matlab"] == expect_matlab
    assert interactive_stat["shell"] == expect_shell
    assert interactive_stat["jupyter"] == expect_jupyter
    assert interactive_stat["non-interactive"] == expect_non_interactive


def test_pre_process_data_fill_missing_ArrayID(load_mock_data):
    data = preprocess_data(
        input_df=load_mock_data, min_elapsed_second=100, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )
    ground_truth = _helper_filter_irrelevants_record(load_mock_data, 100)
    expect_array_id_null = len(ground_truth[(ground_truth["ArrayID"].isnull())])
    array_id_stat = data["ArrayID"].value_counts()
    assert array_id_stat[-1] == expect_array_id_null


def test_pre_process_data_fill_missing_GPUType(load_mock_data):
    data = preprocess_data(
        input_df=load_mock_data, min_elapsed_second=100, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )

    ground_truth = _helper_filter_irrelevants_record(load_mock_data, 100)
    expect_GPUType_null = len(ground_truth[(ground_truth["GPUType"].isnull())])
    expect_GPUs_null = len(ground_truth[(ground_truth["GPUs"].isnull())])
    GPUs_stat = data["GPUs"].value_counts()

    assert sum(x == ["cpu"] for x in data["GPUType"]) == expect_GPUType_null
    assert GPUs_stat[0] == expect_GPUs_null


def test_pre_process_data_fill_missing_Constraints(load_mock_data):
    data = preprocess_data(
        input_df=load_mock_data, min_elapsed_second=100, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )
    ground_truth = _helper_filter_irrelevants_record(load_mock_data, 100)
    expect_constraints_null = len(ground_truth[(ground_truth["Constraints"].isnull())])
    assert sum(len(x) == 0 for x in data["Constraints"]) == expect_constraints_null


def test_pre_process_data_filter_min_esplapes_mock2(load_mock_data):
    data = preprocess_data(
        input_df=load_mock_data, min_elapsed_second=700, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )

    ground_truth = _helper_filter_irrelevants_record(load_mock_data, 700)
    assert len(data) == len(ground_truth)


def test_category_interactive(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600)
    assert data["Interactive"].dtype == "category"


def test_category_QOS(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600)
    assert data["QOS"].dtype == "category"
    expected = {"short", "long", "normal"}
    assert expected.issubset(set(data["QOS"].cat.categories))


def test_category_exit_code(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600)
    assert data["ExitCode"].dtype == "category"
    expected = {"SUCCESS", "ERROR", "SIGNALED"}
    assert expected.issubset(set(data["ExitCode"].cat.categories))


def test_category_partition(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600)
    assert data["Partition"].dtype == "category"
    expected = {"building", "gpu", "cpu"}
    assert expected.issubset(set(data["Partition"].cat.categories))


def test_category_account(load_mock_data):
    data = preprocess_data(input_df=load_mock_data, min_elapsed_second=600)
    assert data["Account"].dtype == "category"


def test_pre_proess_timedelta_conversion(load_mock_data):
    data = preprocess_data(
        input_df=load_mock_data, min_elapsed_second=600, include_CPU_only_job=True, include_failed_cancelled_jobs=True
    )
    ground_truth = _helper_filter_irrelevants_record(load_mock_data, min_elapsed=600)
    maxLen = len(ground_truth)
    time_limit = data["TimeLimit"]
    assert time_limit.dtype == "timedelta64[ns]"
    assert time_limit[0].total_seconds() == ground_truth["TimeLimit"][0]
    assert time_limit[maxLen - 1].total_seconds() == ground_truth["TimeLimit"][maxLen - 1]
