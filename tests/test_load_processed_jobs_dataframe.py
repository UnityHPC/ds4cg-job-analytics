import pytest
import pandas
from src.utility import load_preprocessed_jobs_dataframe_from_duckdb
from .conftest import helper_filter_irrelevant_records
from src.config.enum_constants import StatusEnum
from src.config.constants import ENFORCE_COLUMNS, ESSENTIAL_COLUMNS
from datetime import datetime, timedelta


def test_load_jobs_correct_types(mock_data_path):
    """
    Basic test on return type of function
    """
    res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=mock_data_path)
    assert isinstance(res, pandas.DataFrame)


def test_load_jobs_no_filter(mock_data_frame, mock_data_path):
    """
    Test in case there is no filtering, function should return every valid records from database.
    """
    temp = helper_filter_irrelevant_records(mock_data_frame, 0)
    ground_truth_csv = temp[
        (temp["Status"] != StatusEnum.CANCELLED.value) & (temp["Status"] != StatusEnum.FAILED.value)
    ]
    res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=mock_data_path)
    expect_num_records = len(ground_truth_csv)
    assert expect_num_records == len(res)


def test_load_jobs_filter_date_back_1(mock_data_frame, mock_data_path, recwarn):
    """
    Test for filtering by days_back
    """
    temp = helper_filter_irrelevant_records(mock_data_frame, 0)
    res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=mock_data_path, dates_back=90)
    cutoff = datetime.now() - timedelta(days=90)
    ground_truth_csv = temp[
        (temp["Status"] != StatusEnum.CANCELLED.value)
        & (temp["Status"] != StatusEnum.FAILED.value)
        & (temp["StartTime"] >= cutoff)
    ]
    expect_job_ids = ground_truth_csv["JobID"].to_numpy()
    assert len(ground_truth_csv) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


def test_load_jobs_filter_date_back_2(mock_data_frame, mock_data_path, recwarn):
    """
    Test for filtering by days_back
    """
    temp = helper_filter_irrelevant_records(mock_data_frame, 0)
    res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=mock_data_path, dates_back=150)
    cutoff = datetime.now() - timedelta(days=150)
    ground_truth_csv = temp[
        (temp["Status"] != StatusEnum.CANCELLED.value)
        & (temp["Status"] != StatusEnum.FAILED.value)
        & (temp["StartTime"] >= cutoff)
    ]
    expect_job_ids = ground_truth_csv["JobID"].to_numpy()
    assert len(ground_truth_csv) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


def test_load_jobs_filter_min_elapsed(mock_data_frame, mock_data_path, recwarn):
    """
    Test for filtering by days back and minimum elapsed time.
    """
    temp = helper_filter_irrelevant_records(mock_data_frame, 13000)
    res = load_preprocessed_jobs_dataframe_from_duckdb(
        db_path=mock_data_path, min_elapsed_seconds=13000, dates_back=90
    )
    cutoff = datetime.now() - timedelta(days=90)
    ground_truth_csv = temp[
        (temp["Status"] != StatusEnum.CANCELLED.value)
        & (temp["Status"] != StatusEnum.FAILED.value)
        & (temp["StartTime"] >= cutoff)
    ]
    expect_job_ids = ground_truth_csv["JobID"].to_numpy()
    assert len(ground_truth_csv) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


def test_load_jobs_filter_date_back_include_all(mock_data_frame, mock_data_path, recwarn):
    """
    Test for filtering by days_back, including CPU only jobs and FAILED/ CANCELLED jobs
    """
    temp = helper_filter_irrelevant_records(mock_data_frame, 0, include_cpu_only_jobs=True, include_custom_qos=True)
    res = load_preprocessed_jobs_dataframe_from_duckdb(
        db_path=mock_data_path,
        dates_back=90,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
        include_custom_qos=True,
    )
    cutoff = datetime.now() - timedelta(days=90)
    ground_truth_csv = temp[temp["StartTime"] >= cutoff]
    expect_job_ids = ground_truth_csv["JobID"].to_numpy()
    assert len(ground_truth_csv) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


def test_load_jobs_custom_query(mock_data_frame, mock_data_path, recwarn):
    """
    Test if function fetches expected records when using custom sql query.

    Warnings are expected to be raised since this select a subset of columns.
    """
    query = (
        "SELECT JobID, GPUType, Constraints, StartTime, SubmitTime, NodeList, GPUs, GPUMemUsage, Elapsed FROM Jobs "
        "WHERE Status != 'CANCELLED' AND Status !='FAILED' AND ArrayID is not NULL AND Interactive is not NULL"
    )
    res = load_preprocessed_jobs_dataframe_from_duckdb(
        db_path=mock_data_path, custom_query=query, include_cpu_only_jobs=True, include_custom_qos=True
    )
    ground_truth_csv = mock_data_frame[
        (mock_data_frame["Status"] != "CANCELLED")
        & (mock_data_frame["Status"] != "FAILED")
        & (mock_data_frame["ArrayID"].notna())
        & (mock_data_frame["Interactive"].notna())
    ].copy()
    assert len(res) == len(ground_truth_csv)
    expect_ids = ground_truth_csv["JobID"].to_list()
    for id in res["JobID"]:
        assert id in expect_ids


def test_load_jobs_custom_query_days_back_1(mock_data_frame, mock_data_path, recwarn):
    """
    Test in case custom query does not contain dates_back and dates_back parameter is given.

    Expect result will be filtered correctly by dates_back.
    """
    query = (
        "SELECT JobID, GPUType, Constraints, StartTime, SubmitTime, NodeList, GPUs, GPUMemUsage, Elapsed FROM Jobs "
        "WHERE Status != 'CANCELLED' AND Status !='FAILED' AND ArrayID is not NULL AND Interactive is not NULL"
    )
    res = load_preprocessed_jobs_dataframe_from_duckdb(
        db_path=mock_data_path, custom_query=query, include_cpu_only_jobs=True, dates_back=150
    )
    cutoff = datetime.now() - timedelta(days=150)
    ground_truth_csv = mock_data_frame[
        (mock_data_frame["Status"] != "CANCELLED")
        & (mock_data_frame["Status"] != "FAILED")
        & (mock_data_frame["ArrayID"].notna())
        & (mock_data_frame["StartTime"] >= cutoff)
    ].copy()
    assert len(res) == len(ground_truth_csv)
    expect_ids = ground_truth_csv["JobID"].to_list()
    for id in res["JobID"]:
        assert id in expect_ids


def test_load_jobs_custom_query_days_back_2(mock_data_frame, mock_data_path, recwarn):
    """
    Test in case custom_query already contains dates_back condtion and date_back parameter is given

    Expect the result will be filtered by dates_back condition in the query only and warning is correctly raised
    """
    cutoff = datetime.now() - timedelta(days=150)
    query = (
        "SELECT JobID, GPUType, Constraints, StartTime, SubmitTime, NodeList, GPUs, GPUMemUsage, Elapsed FROM Jobs "
        "WHERE Status != 'CANCELLED' AND Status !='FAILED' AND ArrayID is not NULL AND Interactive is not NULL "
        f"AND StartTime >= '{cutoff}'"
    )

    res = load_preprocessed_jobs_dataframe_from_duckdb(
        db_path=mock_data_path, custom_query=query, include_cpu_only_jobs=True, dates_back=100
    )

    ground_truth_csv = mock_data_frame[
        (mock_data_frame["Status"] != "CANCELLED")
        & (mock_data_frame["Status"] != "FAILED")
        & (mock_data_frame["ArrayID"].notna())
        & (mock_data_frame["StartTime"] >= cutoff)
    ].copy()
    expect_ids = ground_truth_csv["JobID"].to_list()
    expect_warning_msg = (
        "Parameter dates_back = 100 is passed but custom_query already contained conditions for "
        "filtering by dates_back. dates_back condition in custom_query will be used."
    )

    assert str(recwarn[0].message) == expect_warning_msg
    assert len(res) == len(ground_truth_csv)
    for id in res["JobID"]:
        assert id in expect_ids


# TODO: implement proper empty dataframe handling and run this test again
# def test_preprocess_empty_dataframe_warning(mock_data_path, recwarn):
#     """
#     Test handling the dataframe loads from database when the result is empty.

#     Expect a UserWarning to be raised with the appropriate message.
#     """
#     # Query that returns no rows
#     query = "SELECT * FROM Jobs WHERE 1=0"
#     res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=mock_data_path, custom_query=query)
#     assert res.empty
#     # Check that the warning is about empty dataframe
#     assert len(recwarn) == 1
#     assert str(recwarn[0].message) == "Dataframe results from database and filtering is empty."


def test_preprocess_key_errors_raised(mock_data_path, recwarn):
    """
    Test handling the dataframe loads from database when missing one of the ENFORCE_COLUMNS in constants.py

    Expect to raise RuntimeError for any of these columns if they are missing in the dataframe
    """
    for col in ENFORCE_COLUMNS:
        col_names = list(ENFORCE_COLUMNS)
        col_names.remove(col)
        col_str = ", ".join(col_names)
        query = f"SELECT {col_str} FROM Jobs"
        with pytest.raises(
            RuntimeError, match=f"Failed to load jobs DataFrame: 'Column {col} does not exist in dataframe.'"
        ):
            _res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=mock_data_path, custom_query=query)


def test_preprocess_warning_raised(mock_data_path, recwarn):
    """
    Test handling the dataframe loads from database when missing one of the columns

    These columns are not in ENFORCE_COLUMNS so only warnings are expected to be raised
    """
    for col in ESSENTIAL_COLUMNS:
        if col in ENFORCE_COLUMNS:
            continue
        col_names = ENFORCE_COLUMNS.copy()
        remain_cols = ESSENTIAL_COLUMNS.copy()
        remain_cols.remove(col)
        col_names = col_names.union(remain_cols)
        col_str = ", ".join(list(col_names))
        query = f"SELECT {col_str} FROM Jobs"

        expect_warning_msg = (
            f"Column {col} not exist in dataframe, this may result in unexpected results when filtering."
        )
        with pytest.warns(UserWarning, match=expect_warning_msg):
            _res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=mock_data_path, custom_query=query)
