from src.preprocess import preprocess_data
import pandas as pd
from src.config.enum_constants import (
    InteractiveEnum,
    QOSEnum,
    StatusEnum,
    ExitCodeEnum,
    PartitionEnum,
    AdminsAccountEnum,
)
from src.preprocess.preprocess import _get_partition_constraint, _get_requested_vram, _get_vram_constraint
import pytest


def _helper_filter_irrelevant_records(input_df: pd.DataFrame, min_elapsed_seconds: int) -> pd.DataFrame:
    """
    Private function to help generate expected ground truth dataframe for test.

    Given a ground truth dataframe, this will create a new dataframe without records meeting the following criteria:
    - QOS is updates
    - Account is root
    - Partition is building
    - Elasped time is less than min_elapsed

    Args:
        input_df (pd.DataFrame): Input dataframe to filter. Note that the Elapsed field should be in unit seconds.
        min_elapsed_seconds (int): Minimum elapsed time in seconds.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """

    res = input_df[
        (input_df["Elapsed"] >= min_elapsed_seconds)
        & (input_df["Account"] != AdminsAccountEnum.ROOT.value)
        & (input_df["Partition"] != PartitionEnum.BUILDING.value)
        & (input_df["QOS"] != QOSEnum.UPDATES.value)
    ]

    return res


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_filtered_columns(mock_data_frame):
    """
    Test that the preprocessed data does not contain irrelevant columns.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    assert "UUID" not in data.columns
    assert "EndTime" not in data.columns
    assert "Nodes" not in data.columns
    assert "Preempted" not in data.columns
    assert "partition_constraint" in data.columns
    assert "requested_vram" in data.columns


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_filtered_gpu(mock_data_frame):
    """
    Test that the preprocessed data does not contain null GPUType and GPUs.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    is_gpu_type_null = data["GPUType"].isna()
    is_gpu_null = data["GPUs"].isna()
    assert not any(is_gpu_type_null)
    assert not any(is_gpu_null)


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_filtered_status(mock_data_frame):
    """
    Test that the preprocessed data does not contain FAILED or CANCELLED jobs.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    status_failed = data["Status"] == StatusEnum.FAILED.value
    status_cancelled = data["Status"] == StatusEnum.CANCELLED.value
    assert not any(status_failed)
    assert not any(status_cancelled)


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_filtered_min_elapsed_1(mock_data_frame):
    """
    Test that the preprocessed data does not contain jobs with elapsed time below the threshold (300 seconds).
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=300)
    elapsed_below_threshold = data["Elapsed"] < pd.to_timedelta(
        300, unit="s"
    )  # final version of Elapsed column is timedelta so convert for comparison
    assert not any(elapsed_below_threshold)


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_filter_min_elapsed_2(mock_data_frame):
    """
    Test that the preprocessed data contains only jobs with elapsed time below the threshold (700 seconds).
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=700,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )

    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 700)
    assert len(data) == len(ground_truth)


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_filtered_root_account(mock_data_frame):
    """
    Test that the preprocessed data does not contain jobs with root account, partition building, or qos updates.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    partition_building = data["Partition"] == PartitionEnum.BUILDING.value
    qos_updates = data["QOS"] == QOSEnum.UPDATES.value
    account_root = data["Account"] == AdminsAccountEnum.ROOT.value
    assert not any(account_root)
    assert not any(qos_updates)
    assert not any(partition_building)


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_include_cpu_job(mock_data_frame):
    """
    Test that the preprocessed data includes CPU-only jobs when specified.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600, include_cpu_only_jobs=True)
    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 600)
    expected_cpu_type = len(
        ground_truth[
            ground_truth["GPUType"].isna()
            & (ground_truth["Status"] != StatusEnum.FAILED.value)
            & (ground_truth["Status"] != StatusEnum.CANCELLED.value)
        ]
    )
    expected_gpus_count_0 = len(
        ground_truth[
            ground_truth["GPUs"].isna()
            & (ground_truth["Status"] != StatusEnum.FAILED.value)
            & (ground_truth["Status"] != StatusEnum.CANCELLED.value)
        ]
    )
    assert sum(x == ["cpu"] for x in data["GPUType"]) == expected_cpu_type
    assert data["GPUs"].value_counts()[0] == expected_gpus_count_0


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_include_failed_cancelled_job(mock_data_frame):
    """
    Test that the preprocessed data includes FAILED and CANCELLED jobs when specified.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600, include_failed_cancelled_jobs=True)
    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 600)
    expect_failed_status = len(
        ground_truth[
            (ground_truth["Status"] == StatusEnum.FAILED.value)
            & (ground_truth["GPUType"].notna())
            & (ground_truth["GPUs"].notna())
        ]
    )
    expect_cancelled_status = len(
        ground_truth[
            (ground_truth["Status"] == StatusEnum.CANCELLED.value)
            & (ground_truth["GPUType"].notna())
            & (ground_truth["GPUs"].notna())
        ]
    )
    assert data["Status"].value_counts()[StatusEnum.FAILED.value] == expect_failed_status
    assert data["Status"].value_counts()[StatusEnum.CANCELLED.value] == expect_cancelled_status


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_include_all(mock_data_frame):
    """
    Test that the preprocessed data includes all jobs when both CPU-only and FAILED/CANCELLED jobs are specified.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=600,
        include_failed_cancelled_jobs=True,
        include_cpu_only_jobs=True,
    )
    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 600)

    expect_failed_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.FAILED.value)])
    expect_cancelled_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.CANCELLED.value)])
    expect_completed_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.COMPLETED.value)])
    expect_gpu_type_null = len(ground_truth[(ground_truth["GPUType"].isna())])
    expect_gpus_null = len(ground_truth[(ground_truth["GPUs"].isna())])

    assert len(data) == len(ground_truth)
    assert sum(x == ["cpu"] for x in data["GPUType"]) == expect_gpu_type_null
    assert data["GPUs"].value_counts()[0] == expect_gpus_null
    assert data["Status"].value_counts()[StatusEnum.FAILED.value] == expect_failed_status
    assert data["Status"].value_counts()[StatusEnum.CANCELLED.value] == expect_cancelled_status
    assert data["Status"].value_counts()[StatusEnum.COMPLETED.value] == expect_completed_status


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_fill_missing_interactive(mock_data_frame):
    """
    Test that the preprocessed data fills missing interactive job types with 'non-interactive' correctly.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 100)

    expect_non_interactive = len(ground_truth[(ground_truth["Interactive"].isna())])

    interactive_stat = data["Interactive"].value_counts()
    assert interactive_stat[InteractiveEnum.NON_INTERACTIVE.value] == expect_non_interactive


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_fill_missing_array_id(mock_data_frame):
    """
    Test that the preprocessed data fills missing ArrayID with -1 correctly.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 100)
    expect_array_id_null = len(ground_truth[(ground_truth["ArrayID"].isna())])
    array_id_stat = data["ArrayID"].value_counts()
    assert array_id_stat[-1] == expect_array_id_null


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_fill_missing_gpu_type(mock_data_frame):
    """
    Test that the preprocessed data fills missing GPUType with 'cpu' correctly.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )

    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 100)
    expect_gpu_type_null = len(ground_truth[(ground_truth["GPUType"].isna())])
    expect_gpus_null = len(ground_truth[(ground_truth["GPUs"].isna())])
    gpus_stat = data["GPUs"].value_counts()

    assert sum(x == ["cpu"] for x in data["GPUType"]) == expect_gpu_type_null
    assert gpus_stat[0] == expect_gpus_null


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_data_fill_missing_constraints(mock_data_frame):
    """
    Test that the preprocessed data fills missing Constraints with empty numpy array correctly.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=100,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 100)
    expect_constraints_null = len(ground_truth[(ground_truth["Constraints"].isna())])

    assert sum(len(x) == 0 for x in data["Constraints"]) == expect_constraints_null


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_category_interactive(mock_data_frame):
    """
    Test that the preprocessed data has 'Interactive' as a categorical variable and check values contained within it.
    """

    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 600)
    ground_truth_filtered = ground_truth[
        (ground_truth["GPUType"].notna())
        & (ground_truth["GPUs"].notna())
        & (ground_truth["Status"] != StatusEnum.FAILED.value)
        & (ground_truth["Status"] != StatusEnum.CANCELLED.value)
    ]
    expected = set(ground_truth_filtered["Interactive"].dropna().to_numpy()) | set([e.value for e in InteractiveEnum])

    assert data["Interactive"].dtype == "category"
    assert expected.issubset(set(data["Interactive"].cat.categories))


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_category_qos(mock_data_frame):
    """
    Test that the preprocessed data has 'QOS' as a categorical variable and check values contained within it.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 600)
    ground_truth_filtered = ground_truth[
        (ground_truth["GPUType"].notna())
        & (ground_truth["GPUs"].notna())
        & (ground_truth["Status"] != StatusEnum.FAILED.value)
        & (ground_truth["Status"] != StatusEnum.CANCELLED.value)
    ]
    expected = set(ground_truth_filtered["QOS"].dropna().to_numpy()) | set([e.value for e in QOSEnum])

    assert data["QOS"].dtype == "category"
    assert expected.issubset(set(data["QOS"].cat.categories))


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_category_exit_code(mock_data_frame):
    """
    Test that the preprocessed data has 'ExitCode' as a categorical variable and check values contained within it.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)

    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 600)
    ground_truth_filtered = ground_truth[
        (ground_truth["GPUType"].notna())
        & (ground_truth["GPUs"].notna())
        & (ground_truth["Status"] != StatusEnum.FAILED.value)
        & (ground_truth["Status"] != StatusEnum.CANCELLED.value)
    ]
    expected = set(ground_truth_filtered["ExitCode"].dropna().to_numpy()) | set([e.value for e in ExitCodeEnum])

    assert data["ExitCode"].dtype == "category"
    assert expected.issubset(set(data["ExitCode"].cat.categories))


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_category_partition(mock_data_frame):
    """
    Test that the preprocessed data has 'Partition' as a categorical variable and check values contained within it.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)

    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 600)
    ground_truth_filtered = ground_truth[
        (ground_truth["GPUType"].notna())
        & (ground_truth["GPUs"].notna())
        & (ground_truth["Status"] != StatusEnum.FAILED.value)
        & (ground_truth["Status"] != StatusEnum.CANCELLED.value)
    ]
    expected = set(ground_truth_filtered["Partition"].dropna().to_numpy()) | set([e.value for e in PartitionEnum])

    assert data["Partition"].dtype == "category"
    assert expected.issubset(set(data["Partition"].cat.categories))


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_category_account(mock_data_frame):
    """
    Test that the preprocessed data has 'Account' as a categorical variable and check values contained within it.
    """
    data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)

    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 600)
    ground_truth_filtered = ground_truth[
        (ground_truth["GPUType"].notna())
        & (ground_truth["GPUs"].notna())
        & (ground_truth["Status"] != StatusEnum.FAILED.value)
        & (ground_truth["Status"] != StatusEnum.CANCELLED.value)
    ]
    expected = set(ground_truth_filtered["Account"].dropna().to_numpy()) | set([e.value for e in AdminsAccountEnum])

    assert data["Account"].dtype == "category"
    assert expected.issubset(set(data["Account"].cat.categories))


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_timedelta_conversion(mock_data_frame):
    """
    Test that the preprocessed data converts elapsed time to timedelta.
    """
    data = preprocess_data(
        input_df=mock_data_frame,
        min_elapsed_seconds=600,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 600)
    max_len = len(ground_truth)
    time_limit = data["TimeLimit"]

    assert time_limit.dtype == "timedelta64[ns]"
    assert time_limit[0].total_seconds() == ground_truth["TimeLimit"][0]
    assert time_limit[max_len - 1].total_seconds() == ground_truth["TimeLimit"][max_len - 1]


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_preprocess_gpu_type(mock_data_frame):
    """
    Test that the GPUType column is correctly filled and transformed during preprocessing.
    """

    data = preprocess_data(
        input_df=mock_data_frame,
        include_cpu_only_jobs=True,
    )

    # Check that GPUType is filled with 'cpu' for CPU-only jobs
    assert all(row == ["cpu"] for row in data.loc[data["GPUType"].isna(), "GPUType"])

    # Check that numpy arrays in GPUType are converted to lists
    assert all(isinstance(row, list) for row in data["GPUType"] if not pd.isna(row))


def test_get_partition_constraint_known():
    """
    Test that _get_partition_constraint returns the correct VRAM constraint for known partitions.
    """
    # Known partition, e.g. "superpod-a100" maps to a100-80g (80 GiB)
    assert _get_partition_constraint("superpod-a100", 2) == 160
    # Known partition, e.g. "ece-gpu" maps to a100-40g (40 GiB)
    assert _get_partition_constraint("ece-gpu", 1) == 40
    # Known partition, e.g. "ials-gpu" maps to 2080ti (11 GiB)
    assert _get_partition_constraint("ials-gpu", 3) == 33
    # Known partition, e.g. "lan" maps to a40 (48 GiB)
    assert _get_partition_constraint("lan", 2) == 96


def test_get_partition_constraint_unknown():
    """
    Test that _get_partition_constraint returns pd.NA for unknown partitions.
    """
    # Unknown partition returns pd.NA
    assert pd.isna(_get_partition_constraint("unknown-partition", 1))


def test_get_requested_vram_cases():
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


@pytest.mark.parametrize("mock_data_frame", ["mock_data_frame", "mock_data_frame_new_format"], indirect=True)
def test_partition_constraint_and_requested_vram_on_mock_data(mock_data_frame):
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

    # For each row, check that requested_vram is the max of partition_constraint and Constraints (if both are not NA)
    for _idx, row in processed.iterrows():
        part_con = _get_partition_constraint(row["Partition"], row["GPUs"])
        constraint_val = _get_vram_constraint(row["Constraints"], row["GPUs"], row["GPUMemUsage"])
        # Compute expected requested_vram
        if pd.isna(part_con) and pd.isna(constraint_val):
            expected = pd.NA
        elif pd.isna(part_con):
            expected = constraint_val
        elif pd.isna(constraint_val):
            expected = part_con
        else:
            expected = max(part_con, constraint_val)
        actual = row["requested_vram"]
        if pd.isna(expected):
            assert pd.isna(actual)
        else:
            assert actual == expected
