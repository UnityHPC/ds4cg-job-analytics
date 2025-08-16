# import pandas as pd
# from src.config import PartitionInfoFetcher
# from src.config.enum_constants import (
#     AdminsAccountEnum,
#     ExitCodeEnum,
#     InteractiveEnum,
#     QOSEnum,
#     StatusEnum,
#     AdminPartitionEnum,
#     PartitionTypeEnum,
#     ExcludedColumnsEnum,
# )
# from src.config.enum_constants import OptionalColumnsEnum, RequiredColumnsEnum
# from src.config.remote_config import PartitionInfoFetcher
# from src.preprocess.preprocess import _get_partition_constraint, _get_requested_vram, _get_vram_constraint
# from .conftest import preprocess_mock_data
# import pytest

# partition_info = PartitionInfoFetcher().get_info()
# gpu_partitions = [p["name"] for p in partition_info if p["type"] == PartitionTypeEnum.GPU.value]


# @pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
# @pytest.mark.parametrize("column", [member.value for member in ExcludedColumnsEnum])
# def test_preprocess_data_filtred_columns(mock_data_frame, column):
#     """
#     Test that the preprocessed data does not contain irrelevant columns.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
#     assert column not in data.columns


# @pytest.mark.parametrize("mock_data_frame", [False, True], ids=["false_case", "true_case"], indirect=True)
# def test_preprocess_data_filtered_gpu(mock_data_frame: pd.DataFrame) -> None:
#     """
#     Test that the preprocessed data does not contain null GPUType and GPUs.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
#     is_gpu_type_null = data["GPUType"].isna()
#     is_gpu_null = data["GPUs"].isna()
#     assert not any(is_gpu_type_null)
#     assert not any(is_gpu_null)


# @pytest.mark.parametrize("mock_data_frame", [False, True], ids=["false_case", "true_case"], indirect=True)
# def test_preprocess_data_filtered_status(mock_data_frame: pd.DataFrame) -> None:
#     """
#     Test that the preprocessed data does not contain FAILED or CANCELLED jobs.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
#     status_failed = data["Status"] == StatusEnum.FAILED.value
#     status_cancelled = data["Status"] == StatusEnum.CANCELLED.value
#     assert not any(status_failed)
#     assert not any(status_cancelled)


# @pytest.mark.parametrize("mock_data_frame", [False, True], ids=["false_case", "true_case"], indirect=True)
# def test_preprocess_data_filtered_min_elapsed_1(mock_data_frame: pd.DataFrame) -> None:
#     """
#     Test that the preprocessed data does not contain jobs with elapsed time below the threshold (300 seconds).
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=300)
#     elapsed_below_threshold = data["Elapsed"] < pd.to_timedelta(
#         300, unit="s"
#     )  # final version of Elapsed column is timedelta so convert for comparison
#     assert not any(elapsed_below_threshold)


# @pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
# def test_preprocess_data_filter_min_esplaped_2(mock_data_frame, mock_data_path):
#     """
#     Test that the preprocessed data contains only jobs with elapsed time below the threshold (700 seconds).
#     """
#     data = preprocess_data(
#         input_df=mock_data_frame,
#         min_elapsed_seconds=700,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     ground_truth = preprocess_mock_data(
#         mock_data_path,
#         min_elapsed_seconds=700,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     assert len(data) == len(ground_truth), (
#         f"JobIDs in data: {data['JobID'].tolist()}, JobIDs in ground_truth: {ground_truth['JobID'].tolist()}"
#     )


# @pytest.mark.parametrize("mock_data_frame", [False, True], ids=["false_case", "true_case"], indirect=True)
# def test_preprocess_data_filtered_root_account(mock_data_frame: pd.DataFrame) -> None:
#     """
#     Test that the preprocessed data does not contain jobs with root account, partition building, or qos updates.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
#     partition_building = data["Partition"] == AdminPartitionEnum.BUILDING.value
#     qos_updates = data["QOS"] == QOSEnum.UPDATES.value
#     account_root = data["Account"] == AdminsAccountEnum.ROOT.value
#     assert not any(account_root)
#     assert not any(qos_updates)
#     assert not any(partition_building)


# @pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
# def test_preprocess_data_include_cpu_job(mock_data_frame, mock_data_path):
#     """
#     Test that the preprocessed data includes CPU-only jobs when specified.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600, include_cpu_only_jobs=True)
#     ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600, include_cpu_only_jobs=True)
#     expect_cpus_jobs = len(ground_truth[~ground_truth["Partition"].isin(gpu_partitions)])
#     assert sum(~(data["Partition"].isin(gpu_partitions))) == expect_cpus_jobs


# @pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
# def test_preprocess_data_include_failed_cancelled_job(mock_data_frame, mock_data_path):
#     """
#     Test that the preprocessed data includes FAILED and CANCELLED jobs when specified.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600, include_failed_cancelled_jobs=True)
#     ground_truth = _helper_filter_irrelevant_records(mock_data_frame, 600)
#     expect_failed_status = len(
#         ground_truth[
#             (ground_truth["Status"] == StatusEnum.FAILED.value)
#             & (ground_truth["GPUType"].notna())
#             & (ground_truth["GPUs"].notna())
#         ]
#     )
#     expect_cancelled_status = len(
#         ground_truth[
#             (ground_truth["Status"] == StatusEnum.CANCELLED.value)
#             & (ground_truth["GPUType"].notna())
#             & (ground_truth["GPUs"].notna())
#         ]
#     )
#     assert data["Status"].value_counts()[StatusEnum.FAILED.value] == expect_failed_status
#     assert data["Status"].value_counts()[StatusEnum.CANCELLED.value] == expect_cancelled_status


# @pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
# def test_all_boolean_args_being_true(mock_data_frame):
#     """
#     Test that the preprocessed data includes all jobs when CPU-only, FAILED/CANCELLED, custom QOS jobs are specified.
#     """
#     data = preprocess_data(
#         input_df=mock_data_frame,
#         min_elapsed_seconds=600,
#         include_failed_cancelled_jobs=True,
#         include_cpu_only_jobs=True,
#         include_custom_qos_jobs=True,
#     )
#     ground_truth = preprocess_mock_data(
#         mock_data_path,
#         min_elapsed_seconds=600,
#         include_cpu_only_jobs=True,
#         include_custom_qos_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     expect_failed_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.FAILED.value)])
#     expect_cancelled_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.CANCELLED.value)])
#     expect_completed_status = len(ground_truth[(ground_truth["Status"] == StatusEnum.COMPLETED.value)])
#     expect_gpu_type_null = len(ground_truth[(ground_truth["GPUType"].isna())])
#     expect_gpus_null = len(ground_truth[(ground_truth["GPUs"].isna())])

#     assert len(data) == len(ground_truth), (
#         f"JobIDs in data: {data['JobID'].tolist()}, JobIDs in ground_truth: {ground_truth['JobID'].tolist()}"
#     )
#     assert sum(x == ["cpu"] for x in data["GPUType"]) == expect_gpu_type_null
#     assert sum(x == 0 for x in data["GPUs"]) == expect_gpus_null
#     assert sum(x == StatusEnum.FAILED.value for x in data["Status"]) == expect_failed_status
#     assert sum(x == StatusEnum.CANCELLED.value for x in data["Status"]) == expect_cancelled_status
#     assert sum(x == StatusEnum.COMPLETED.value for x in data["Status"]) == expect_completed_status


# def test_preprocess_data_fill_missing_interactive(mock_data_frame):
#     """
#     Test that the preprocessed data fills missing interactive job types with 'non-interactive' correctly.
#     """
#     data = preprocess_data(
#         input_df=mock_data_frame,
#         min_elapsed_seconds=100,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     ground_truth = preprocess_mock_data(
#         mock_data_path,
#         min_elapsed_seconds=100,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )

#     expect_non_interactive = len(ground_truth[(ground_truth["Interactive"].isna())])

#     assert sum(x == InteractiveEnum.NON_INTERACTIVE.value for x in data["Interactive"]) == expect_non_interactive


# def test_preprocess_data_fill_missing_array_id(mock_data_frame):
#     """
#     Test that the preprocessed data fills missing ArrayID with -1 correctly.
#     """
#     data = preprocess_data(
#         input_df=mock_data_frame,
#         min_elapsed_seconds=100,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     ground_truth = preprocess_mock_data(
#         mock_data_path,
#         min_elapsed_seconds=100,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     expect_array_id_null = len(ground_truth[(ground_truth["ArrayID"].isna())])
#     assert sum(x == -1 for x in data["ArrayID"]) == expect_array_id_null


# def test_preprocess_data_fill_missing_gpu_type(mock_data_frame):
#     """
#     Test that the preprocessed data fills missing GPUType with pd.NA correctly.
#     """
#     data = preprocess_data(
#         input_df=mock_data_frame,
#         min_elapsed_seconds=100,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )

#     ground_truth = preprocess_mock_data(
#         mock_data_path,
#         min_elapsed_seconds=100,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     expect_gpu_type_null = len(ground_truth[(ground_truth["GPUType"].isna())])
#     expect_gpus_null = len(ground_truth[(ground_truth["GPUs"].isna())])
#     gpus_stat = data["GPUs"].value_counts()

#     assert sum(x == ["cpu"] for x in data["GPUType"]) == expect_gpu_type_null
#     assert gpus_stat[0] == expect_gpus_null, (
#         f"Expected {expect_gpus_null} null GPUs, but found {gpus_stat[0]} null GPUs."
#     )


# def test_preprocess_data_fill_missing_constraints(mock_data_frame):
#     """
#     Test that the preprocessed data fills missing Constraints with empty numpy array correctly.
#     """
#     data = preprocess_data(
#         input_df=mock_data_frame,
#         min_elapsed_seconds=100,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     ground_truth = preprocess_mock_data(
#         mock_data_path,
#         min_elapsed_seconds=100,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     expect_constraints_null = len(ground_truth[(ground_truth["Constraints"].isna())])

#     assert sum(len(x) == 0 for x in data["Constraints"]) == expect_constraints_null


# def test_category_interactive(mock_data_frame):
#     """
#     Test that the preprocessed data has 'Interactive' as a categorical variable and check values contained within it.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
#     ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600)
#     expected = set(ground_truth["Interactive"].dropna().to_numpy()) | set([e.value for e in InteractiveEnum])

#     assert data["Interactive"].dtype == "category"
#     assert expected.issubset(set(data["Interactive"].cat.categories))


# def test_category_qos(mock_data_frame):
#     """
#     Test that the preprocessed data has 'QOS' as a categorical variable and check values contained within it.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)
#     ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600)
#     expected = set(ground_truth["QOS"].dropna().to_numpy()) | set([e.value for e in QOSEnum])

#     assert data["QOS"].dtype == "category"
#     assert expected.issubset(set(data["QOS"].cat.categories))


# def test_category_exit_code(mock_data_frame):
#     """
#     Test that the preprocessed data has 'ExitCode' as a categorical variable and check values contained within it.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)

#     ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600)
#     expected = set(ground_truth["ExitCode"].dropna().to_numpy()) | set([e.value for e in ExitCodeEnum])

#     assert data["ExitCode"].dtype == "category"
#     assert expected.issubset(set(data["ExitCode"].cat.categories))


# def test_category_partition(mock_data_frame):
#     """
#     Test that the preprocessed data has 'Partition' as a categorical variable and check values contained within it.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)

#     ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600)
#     expected = set(ground_truth["Partition"].dropna().to_numpy()) | set([e.value for e in AdminPartitionEnum])

#     assert data["Partition"].dtype == "category"
#     assert expected.issubset(set(data["Partition"].cat.categories))


# def test_category_account(mock_data_frame):
#     """
#     Test that the preprocessed data has 'Account' as a categorical variable and check values contained within it.
#     """
#     data = preprocess_data(input_df=mock_data_frame, min_elapsed_seconds=600)

#     ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=600)
#     expected = set(ground_truth["Account"].dropna().to_numpy()) | set([e.value for e in AdminsAccountEnum])

#     assert data["Account"].dtype == "category"
#     assert expected.issubset(set(data["Account"].cat.categories))


# def test_preprocess_timedelta_conversion(mock_data_frame):
#     """
#     Test that the preprocessed data converts elapsed time to timedelta.
#     """
#     data = preprocess_data(
#         input_df=mock_data_frame,
#         min_elapsed_seconds=600,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     ground_truth = preprocess_mock_data(
#         mock_data_path,
#         min_elapsed_seconds=600,
#         include_cpu_only_jobs=True,
#         include_failed_cancelled_jobs=True,
#     )
#     time_limit = data["TimeLimit"]

#     assert time_limit.dtype == "timedelta64[ns]"
#     assert time_limit[0].total_seconds() / 60 == ground_truth["TimeLimit"][0]
#     assert time_limit[max_len - 1].total_seconds() / 60 == ground_truth["TimeLimit"][max_len - 1]


# @pytest.mark.parametrize("mock_data_frame", [False, True], ids=["false_case", "true_case"], indirect=True)
# def test_preprocess_gpu_type(mock_data_frame: pd.DataFrame) -> None:
#     """
#     Test that the GPUType column is correctly filled and transformed during preprocessing.
#     """
#     data = preprocess_data(
#         input_df=mock_data_frame,
#         include_cpu_only_jobs=True,
#     )

#     # Check that GPUType is filled with NA for CPU-only jobs
#     assert all(pd.isna(row) for row in data.loc[data["GPUType"].isna(), "GPUType"])

#     # Check that numpy arrays in GPUType are converted to lists
#     assert all(isinstance(row, list) for row in data["GPUType"] if not pd.isna(row))
