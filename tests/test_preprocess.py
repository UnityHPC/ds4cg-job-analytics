import pandas as pd
import pytest
from pandas.api.typing import NAType


from src.config.enum_constants import (
    AdminsAccountEnum,
    ExitCodeEnum,
    InteractiveEnum,
    QOSEnum,
    StatusEnum,
    AdminPartitionEnum,
    ExcludedColumnsEnum,
    RequiredColumnsEnum,
    OptionalColumnsEnum,
)
from src.preprocess import preprocess_data
from src.preprocess.preprocess import _get_partition_constraint, _get_requested_vram, _get_vram_constraint
from .conftest import preprocess_mock_data


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
@pytest.mark.parametrize("column", [member.value for member in ExcludedColumnsEnum])
def test_preprocess_data_filtred_columns(mock_data_frame: pd.DataFrame, column: str) -> None:
    """
    Test that the preprocessed data does not contain irrelevant columns.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    assert column not in data.columns


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_filtered_gpu(mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data does not contain null GPUType and GPUs.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    is_gpu_type_null = data["GPUType"].isna()
    is_gpu_null = data["GPUs"].isna()
    assert not any(is_gpu_type_null)
    assert not any(is_gpu_null)


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_filtered_status(mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data does not contain FAILED or CANCELLED jobs.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    status_failed = data["Status"] == StatusEnum.FAILED.value
    status_cancelled = data["Status"] == StatusEnum.CANCELLED.value
    assert not any(status_failed)
    assert not any(status_cancelled)


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_filtered_min_elapsed_1(mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data does not contain jobs with elapsed time below the threshold (300 seconds).
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=300)
    elapsed_below_threshold = data["Elapsed"] < pd.to_timedelta(
        300, unit="s"
    )  # final version of Elapsed column is timedelta so convert for comparison
    assert not any(elapsed_below_threshold)


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_filter_min_elapsed_2(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data contains only jobs with elapsed time above the threshold (700 seconds).
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=700,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    ground_truth = preprocess_mock_data(
        mock_data_path,
        min_elapsed_seconds=700,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    assert len(data) == len(ground_truth), (
        f"JobIDs in data: {data['JobID'].tolist()}, JobIDs in ground_truth: {ground_truth['JobID'].tolist()}"
    )


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_filtered_root_account(mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data does not contain jobs with root account, partition building, or qos updates.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    partition_building = data["Partition"] == AdminPartitionEnum.BUILDING.value
    qos_updates = data["QOS"] == QOSEnum.UPDATES.value
    account_root = data["Account"] == AdminsAccountEnum.ROOT.value
    assert not any(account_root)
    assert not any(qos_updates)
    assert not any(partition_building)


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_include_cpu_job(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data includes CPU-only jobs when specified.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600, include_cpu_only_jobs=True)
    ground_truth = preprocess_mock_data(mock_data_path, include_cpu_only_jobs=True)
    expected_cpu_type = len(ground_truth[ground_truth["GPUType"].isna()])
    expected_gpus_count_0 = len(ground_truth[ground_truth["GPUs"].isna()])
    assert sum(pd.isna(x) for x in data["GPUType"]) == expected_cpu_type
    assert sum(x == 0 for x in data["GPUs"]) == expected_gpus_count_0

    # Check that GPUType is NA for CPU-only jobs
    assert all(isinstance(row, list | dict) for row in data["GPUType"] if not pd.isna(row))


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_include_failed_cancelled_job(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data includes FAILED and CANCELLED jobs when specified.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600, include_failed_cancelled_jobs=True)
    ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600, include_failed_cancelled_jobs=True)
    expect_failed_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.FAILED.value)])
    expect_cancelled_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.CANCELLED.value)])
    assert sum(x == StatusEnum.FAILED.value for x in data["Status"]) == expect_failed_status
    assert sum(x == StatusEnum.CANCELLED.value for x in data["Status"]) == expect_cancelled_status


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_include_custom_qos_values(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600, include_custom_qos_jobs=True)
    ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600, include_custom_qos_jobs=True)
    filtered_ground_truth = ground_truth[
        (ground_truth["Status"] != "CANCELLED") & (ground_truth["Status"] != "FAILED")
    ].copy()
    assert len(data) == len(filtered_ground_truth), (
        f"JobIDs in data: {data['JobID'].tolist()}, JobIDs in ground_truth: {filtered_ground_truth['JobID'].tolist()}"
    )
    expect_ids = filtered_ground_truth["JobID"].to_list()
    for id in data["JobID"]:
        assert id in expect_ids


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_all_boolean_args_being_true(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data includes all jobs when both CPU-only and FAILED/CANCELLED jobs are specified.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=600,
        include_failed_cancelled_jobs=True,
        include_cpu_only_jobs=True,
        include_custom_qos_jobs=True,
    )
    ground_truth = preprocess_mock_data(
        mock_data_path,
        min_elapsed_seconds=600,
        include_cpu_only_jobs=True,
        include_custom_qos_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    expect_failed_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.FAILED.value)])
    expect_cancelled_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.CANCELLED.value)])
    expect_completed_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.COMPLETED.value)])
    expect_gpu_type_null = len(ground_truth[(ground_truth["GPUType"].isna())])
    expect_gpus_null = len(ground_truth[(ground_truth["GPUs"].isna())])

    assert len(data) == len(ground_truth), (
        f"JobIDs in data: {data['JobID'].tolist()}, JobIDs in ground_truth: {ground_truth['JobID'].tolist()}"
    )
    assert sum(pd.isna(x) for x in data["GPUType"]) == expect_gpu_type_null
    assert sum(x == 0 for x in data["GPUs"]) == expect_gpus_null
    assert sum(x == StatusEnum.FAILED.value for x in data["Status"]) == expect_failed_status
    assert sum(x == StatusEnum.CANCELLED.value for x in data["Status"]) == expect_cancelled_status
    assert sum(x == StatusEnum.COMPLETED.value for x in data["Status"]) == expect_completed_status


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_fill_missing_interactive(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data fills missing interactive job types with 'non-interactive' correctly.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    ground_truth = preprocess_mock_data(
        mock_data_path,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )

    expect_non_interactive = len(ground_truth[(ground_truth["Interactive"].isna())])

    assert sum(x == InteractiveEnum.NON_INTERACTIVE.value for x in data["Interactive"]) == expect_non_interactive


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_fill_missing_array_id(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data fills missing ArrayID with -1 correctly.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    ground_truth = preprocess_mock_data(
        mock_data_path,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    expect_array_id_null = len(ground_truth[(ground_truth["ArrayID"].isna())])
    assert sum(x == -1 for x in data["ArrayID"]) == expect_array_id_null


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_fill_missing_gpu_type(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data fills missing GPUType with pd.NA correctly.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )

    ground_truth = preprocess_mock_data(
        mock_data_path,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    expect_gpu_type_null = len(ground_truth[(ground_truth["GPUType"].isna())])
    expect_gpus_null = len(ground_truth[(ground_truth["GPUs"] == 0) | (ground_truth["GPUs"].isna())])
    actual_count_gpu_0 = sum(x == 0 for x in data["GPUs"])
    assert sum(pd.isna(x) for x in data["GPUType"]) == expect_gpu_type_null
    assert actual_count_gpu_0 == expect_gpus_null, (
        f"Expected {expect_gpus_null} null GPUs, but found {actual_count_gpu_0} null GPUs."
    )


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_data_fill_missing_constraints(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data fills missing Constraints with empty numpy array correctly.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    ground_truth = preprocess_mock_data(
        mock_data_path,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    expect_constraints_null = len(ground_truth[(ground_truth["Constraints"].isna())])

    assert sum(len(x) == 0 for x in data["Constraints"]) == expect_constraints_null


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_category_interactive(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data has 'Interactive' as a categorical variable and check values contained within it.
    """

    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600)
    expected = set(ground_truth["Interactive"].dropna().to_numpy()) | set([e.value for e in InteractiveEnum])

    assert data["Interactive"].dtype == "category"
    assert expected.issubset(set(data["Interactive"].cat.categories))


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_category_qos(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data has 'QOS' as a categorical variable and check values contained within it.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600)
    expected = set(ground_truth["QOS"].dropna().to_numpy()) | set([e.value for e in QOSEnum])

    assert data["QOS"].dtype == "category"
    assert expected.issubset(set(data["QOS"].cat.categories))


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_category_exit_code(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data has 'ExitCode' as a categorical variable and check values contained within it.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)

    ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600)
    expected = set(ground_truth["ExitCode"].dropna().to_numpy()) | set([e.value for e in ExitCodeEnum])

    assert data["ExitCode"].dtype == "category"
    assert expected.issubset(set(data["ExitCode"].cat.categories))


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_category_partition(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data has 'Partition' as a categorical variable and check values contained within it.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)

    ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600)
    expected = set(ground_truth["Partition"].dropna().to_numpy()) | set([e.value for e in AdminPartitionEnum])

    assert data["Partition"].dtype == "category"
    assert expected.issubset(set(data["Partition"].cat.categories))


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_category_account(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data has 'Account' as a categorical variable and check values contained within it.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)

    ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600)
    expected = set(ground_truth["Account"].dropna().to_numpy()) | set([e.value for e in AdminsAccountEnum])

    assert data["Account"].dtype == "category"
    assert expected.issubset(set(data["Account"].cat.categories))


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_timedelta_conversion(mock_data_path: str, mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the preprocessed data converts elapsed time to timedelta.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=600,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    ground_truth = preprocess_mock_data(
        mock_data_path, min_elapsed_seconds=600, include_cpu_only_jobs=True, include_failed_cancelled_jobs=True
    )
    time_limit = data["TimeLimit"]
    assert time_limit.dtype == "timedelta64[ns]"  # assert correct type

    time_limit_list = time_limit.tolist()
    ground_truth_time_limit = ground_truth["TimeLimit"].tolist()
    for i, timedelta in enumerate(time_limit_list):
        assert timedelta.total_seconds() / 60 == ground_truth_time_limit[i]


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_gpu_type(mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the GPUType column is correctly filled and transformed during preprocessing.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        include_cpu_only_jobs=True,
    )

    # Check that GPUType is filled with NA for CPU-only jobs
    assert all(pd.isna(row) for row in data.loc[data["GPUType"].isna(), "GPUType"])


def test_get_partition_constraint_known() -> None:
    """
    Test that _get_partition_constraint returns the correct VRAM constraint for known partitions.
    """
    # Known partition, e.g. "superpod-a100" maps to a100-80g (80 GiB)
    assert _get_partition_constraint("superpod-a100", 2) == 160
    # Known partition, e.g. "ece-gpu" maps to a100-40g (40 GiB)
    assert _get_partition_constraint("ece-gpu", 1) == 40
    # Known partition, e.g. "lan" maps to a40 (48 GiB)
    assert _get_partition_constraint("lan", 2) == 96


def test_get_partition_constraint_unknown() -> None:
    """
    Test that _get_partition_constraint returns pd.NA for unknown partitions.
    """
    # Unknown partition returns pd.NA
    assert pd.isna(_get_partition_constraint("unknown-partition", 1))


def test_get_requested_vram_cases() -> None:
    """
    Test that _get_requested_vram handles various cases correctly.
    """
    # Both constraints are int, choose partition constraint
    assert _get_requested_vram(100, 80) == 80
    assert _get_requested_vram(80, 100) == 100
    # One constraint is NA
    assert _get_requested_vram(pd.NA, 80) == 80
    assert _get_requested_vram(100, pd.NA) == 100
    # Both constraints are NA
    assert pd.isna(_get_requested_vram(pd.NA, pd.NA))


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_partition_constraint_and_requested_vram_on_mock_data(mock_data_frame: pd.DataFrame) -> None:
    """
    Test that the partition_constraint and requested_vram columns are correctly computed in the preprocessed data.
    """
    # Run preprocess_data on the mock data
    processed = preprocess_data(
        mock_data_frame, min_elapsed_seconds=0, include_cpu_only_jobs=True, include_failed_cancelled_jobs=True
    )

    # Check that partition_constraint and requested_vram columns exist
    assert "partition_constraint" in processed.columns
    assert "requested_vram" in processed.columns

    # For each row, check that requested_vram is set to partition_constraint if both are not NA.
    for _idx, row in processed.iterrows():
        part_con = _get_partition_constraint(row["Partition"], row["GPUs"])
        constraint_val = _get_vram_constraint(row["Constraints"], row["GPUs"])
        # Compute expected requested_vram
        expected: int | NAType
        if pd.isna(part_con) and pd.isna(constraint_val):
            expected = pd.NA
        elif pd.isna(part_con):
            expected = constraint_val
        else:
            expected = part_con
        actual = row["requested_vram"]
        if pd.isna(expected):
            assert pd.isna(actual)
        else:
            assert actual == expected


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
@pytest.mark.parametrize("missing_col", [col.value for col in RequiredColumnsEnum])
def test_preprocess_missing_required_columns(mock_data_frame: pd.DataFrame, missing_col: str) -> None:
    """
    Test handling the dataframe when missing one of the ENFORCE_COLUMNS in constants.py.

    Expect to raise KeyError for any of these columns if they are missing in the dataframe.
    """
    cur_df = mock_data_frame.drop(missing_col, axis=1, inplace=False)
    with pytest.raises(KeyError, match=f"Column {missing_col} does not exist in dataframe."):
        _res = preprocess_data(cur_df)


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
@pytest.mark.parametrize("missing_col", [col.value for col in OptionalColumnsEnum])
def test_preprocess_missing_optional_columns(
    mock_data_frame: pd.DataFrame, missing_col: str, recwarn: pytest.WarningsRecorder
) -> None:
    """
    Test handling the dataframe when missing one of the columns.

    These columns are not in ENFORCE_COLUMNS so only warnings are expected to be raised.
    """
    cur_df = mock_data_frame.drop(missing_col, axis=1, inplace=False)

    expect_warning_msg = (
        f"Column '{missing_col}' is missing from the dataframe. "
        "This may impact filtering operations and downstream processing."
    )
    _res = preprocess_data(cur_df)

    # Check that a warning was raised with the expected message
    assert len(recwarn) == 1
    assert str(recwarn[0].message) == expect_warning_msg


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_preprocess_empty_dataframe_warning(mock_data_frame: pd.DataFrame, recwarn: pytest.WarningsRecorder) -> None:
    """
    Test handling when preprocess_data results in an empty dataframe.

    Expect a UserWarning to be raised with the appropriate message.
    Also verify that columns added and type-casted in _cast_type_and_add_columns have correct data types.
    """
    # Make a copy of mock_data_frame and remove all entries to make it empty
    empty_df = mock_data_frame.copy()
    empty_df = empty_df.iloc[0:0]
    # Should trigger the warning since the dataframe is empty
    result = preprocess_data(empty_df)

    # Check that the result is still empty
    assert result.empty

    # Check that a warning was raised about empty dataframe
    assert len(recwarn) == 1
    assert str(recwarn[0].message) == "Dataframe results from database and filtering is empty."

    # Test that columns added in _cast_type_and_add_columns have correct types
    # New columns added for empty dataframes
    assert "Queued" in result.columns
    assert result["Queued"].dtype == "timedelta64[ns]"

    assert "vram_constraint" in result.columns
    assert result["vram_constraint"].dtype == pd.Int64Dtype()

    assert "allocated_vram" in result.columns
    assert result["allocated_vram"].dtype == pd.Int64Dtype()

    # Test that time columns were converted to datetime (if they exist)
    time_columns = ["StartTime", "SubmitTime"]
    for col in time_columns:
        if col in result.columns:
            assert pd.api.types.is_datetime64_any_dtype(result[col])

    # Test that duration columns were converted to timedelta (if they exist)
    duration_columns = ["TimeLimit", "Elapsed"]
    for col in duration_columns:
        if col in result.columns:
            assert pd.api.types.is_timedelta64_dtype(result[col])

    # Test that categorical columns have correct dtype (if they exist)
    categorical_columns = ["Interactive", "QOS", "ExitCode", "Partition", "Account", "Status"]
    for col in categorical_columns:
        if col in result.columns:
            assert result[col].dtype == "category"
