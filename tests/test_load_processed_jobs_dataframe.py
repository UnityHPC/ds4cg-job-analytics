import pytest
import pandas
from src.utility import load_preprocessed_jobs_dataframe_from_duckdb
from .conftest import helper_filter_irrelevant_records
from src.config.enum_constants import StatusEnum
from src.config.constants import ENFORCE_COLUMNS, ESSENTIAL_COLUMNS
from datetime import datetime, timedelta


def test_load_jobs_correct_types(mock_data):
    """
    Basic test on return type of function
    """
    res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=mock_data[1])
    assert isinstance(res, pandas.DataFrame)


def test_load_jobs_no_filter(mock_data):
    """
    Test in case there is no filtering, function should return every valid records from database.
    """
    mock_csv, db_path = mock_data
    temp = helper_filter_irrelevant_records(mock_csv, 0)
    ground_truth_csv = temp[
        (temp["Status"] != StatusEnum.CANCELLED.value)
        & (temp["Status"] != StatusEnum.FAILED.value)
        & (temp["GPUType"].notna())
        & (temp["GPUs"].notna())
    ]
    res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=db_path)
    expect_num_records = len(ground_truth_csv)
    assert expect_num_records == len(res)


def test_load_jobs_filter_day_back_1(mock_data):
    """
    Test for filtering by days_back
    """
    mock_csv, db_path = mock_data
    temp = helper_filter_irrelevant_records(mock_csv, 0)
    res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=db_path, days_back=90)
    cutoff = datetime.now() - timedelta(days=90)
    ground_truth_csv = temp[
        (temp["Status"] != StatusEnum.CANCELLED.value)
        & (temp["Status"] != StatusEnum.FAILED.value)
        & (temp["GPUType"].notna())
        & (temp["GPUs"].notna())
        & (temp["StartTime"] >= cutoff)
    ]
    expect_job_ids = ground_truth_csv["JobID"].to_numpy()
    assert len(ground_truth_csv) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


def test_load_jobs_filter_day_back_2(mock_data):
    """
    Test for filtering by days_back
    """
    mock_csv, db_path = mock_data
    temp = helper_filter_irrelevant_records(mock_csv, 0)
    res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=db_path, days_back=150)
    cutoff = datetime.now() - timedelta(days=150)
    ground_truth_csv = temp[
        (temp["Status"] != StatusEnum.CANCELLED.value)
        & (temp["Status"] != StatusEnum.FAILED.value)
        & (temp["GPUType"].notna())
        & (temp["GPUs"].notna())
        & (temp["StartTime"] >= cutoff)
    ]
    expect_job_ids = ground_truth_csv["JobID"].to_numpy()
    assert len(ground_truth_csv) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


def test_load_jobs_filter_min_elapsed(mock_data):
    """
    Test for filtering by days back and minimum elapsed time.
    """
    mock_csv, db_path = mock_data
    temp = helper_filter_irrelevant_records(mock_csv, 13000)
    res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=db_path, min_elapsed_seconds=13000, days_back=90)
    cutoff = datetime.now() - timedelta(days=90)
    ground_truth_csv = temp[
        (temp["Status"] != StatusEnum.CANCELLED.value)
        & (temp["Status"] != StatusEnum.FAILED.value)
        & (temp["GPUType"].notna())
        & (temp["GPUs"].notna())
        & (temp["StartTime"] >= cutoff)
    ]
    expect_job_ids = ground_truth_csv["JobID"].to_numpy()
    assert len(ground_truth_csv) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


def test_load_jobs_filter_day_back_include_all(mock_data):
    """
    Test for filtering by days_back, including CPU only jobs and FAILED/ CANCELLED jobs
    """
    mock_csv, db_path = mock_data
    temp = helper_filter_irrelevant_records(mock_csv, 0)
    res = load_preprocessed_jobs_dataframe_from_duckdb(
        db_path=db_path, days_back=90, include_cpu_only_jobs=True, include_failed_cancelled_jobs=True
    )
    cutoff = datetime.now() - timedelta(days=90)
    ground_truth_csv = temp[temp["StartTime"] >= cutoff]
    expect_job_ids = ground_truth_csv["JobID"].to_numpy()
    assert len(ground_truth_csv) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


def test_load_jobs_custom_query(mock_data, recwarn):
    mock_csv, db_path = mock_data
    query = (
        "SELECT JobID, GPUType, Constraints, StartTime, SubmitTime, NodeList, GPUs, GPUMemUsage, Elapsed FROM Jobs "
        "WHERE Status != 'CANCELLED' AND Status !='FAILED' AND ArrayID is not NULL AND Interactive is not NULL"
    )
    res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=db_path, custom_query=query, include_cpu_only_jobs=True)
    filtered_data = mock_csv[
        (mock_csv["Status"] != "CANCELLED") & (mock_csv["Status"] != "FAILED") & (mock_csv["ArrayID"].notna())
    ].copy()
    assert len(res) == len(filtered_data)
    expect_ids = filtered_data["JobID"].to_list()
    for id in res["JobID"]:
        assert id in expect_ids


def test_load_jobs_custom_query_days_back_1(mock_data, recwarn):
    """
    Test in case custom query does not contain dates_back and dates_back parameter is given

    Expect result will be filtered correctly by dates_back.
    """
    mock_csv, db_path = mock_data
    query = (
        "SELECT JobID, GPUType, Constraints, StartTime, SubmitTime, NodeList, GPUs, GPUMemUsage, Elapsed FROM Jobs "
        "WHERE Status != 'CANCELLED' AND Status !='FAILED' AND ArrayID is not NULL AND Interactive is not NULL"
    )
    res = load_preprocessed_jobs_dataframe_from_duckdb(
        db_path=db_path, custom_query=query, include_cpu_only_jobs=True, days_back=150
    )
    cutoff = datetime.now() - timedelta(days=150)
    filtered_data = mock_csv[
        (mock_csv["Status"] != "CANCELLED")
        & (mock_csv["Status"] != "FAILED")
        & (mock_csv["ArrayID"].notna())
        & (mock_csv["StartTime"] >= cutoff)
    ].copy()
    assert len(res) == len(filtered_data)
    expect_ids = filtered_data["JobID"].to_list()
    for id in res["JobID"]:
        assert id in expect_ids


def test_load_jobs_custom_query_days_back_2(mock_data, recwarn):
    """
    Test in case custom_query already contains dates_back condtion and date_back parameter is given

    Expect the result will be filtered by dates_back condition in the query only and warning is correctly raised
    """
    mock_csv, db_path = mock_data
    cutoff = datetime.now() - timedelta(days=150)
    query = (
        "SELECT JobID, GPUType, Constraints, StartTime, SubmitTime, NodeList, GPUs, GPUMemUsage, Elapsed FROM Jobs "
        "WHERE Status != 'CANCELLED' AND Status !='FAILED' AND ArrayID is not NULL AND Interactive is not NULL "
        f"AND StartTime >= '{cutoff}'"
    )

    res = load_preprocessed_jobs_dataframe_from_duckdb(
        db_path=db_path, custom_query=query, include_cpu_only_jobs=True, days_back=100
    )

    filtered_data = mock_csv[
        (mock_csv["Status"] != "CANCELLED")
        & (mock_csv["Status"] != "FAILED")
        & (mock_csv["ArrayID"].notna())
        & (mock_csv["StartTime"] >= cutoff)
    ].copy()
    expect_ids = filtered_data["JobID"].to_list()
    expect_warning_msg = (
        "Parameter days_back = 100 is passed but custom_query already contained conditions for "
        "filtering by dates_back. dates_back condition in custom_query will be used."
    )

    assert len(recwarn) == 5
    assert str(recwarn[0].message) == expect_warning_msg
    assert len(res) == len(filtered_data)
    for id in res["JobID"]:
        assert id in expect_ids


def test_preprocess_key_errors_raised(mock_data, recwarn):
    """
    Test handling the dataframe loads from database when missing one of the ENFORCE_COLUMNS in constants.py

    Expect to raise KeyError for any of these columns if they are missing in the dataframe
    """
    mock_csv, db_path = mock_data
    for col in ENFORCE_COLUMNS:
        col_names = list(ENFORCE_COLUMNS)
        col_names.remove(col)
        col_str = ", ".join(col_names)
        query = f"SELECT {col_str} FROM Jobs"
        with pytest.raises(
            RuntimeError, match=f"Failed to load jobs DataFrame: 'Column {col} does not exist in dataframe.'"
        ):
            _res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=db_path, custom_query=query)


def test_preprocess_warning_raised(mock_data, recwarn):
    """
    Test handling the dataframe loads from database when missing one of the columns

    These columns are not the one in ENFORCE_COLUMNS so warnings, not errors, are expected to be raised
    """
    mock_csv, db_path = mock_data
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
            f"Column {col} not exist in dataframe, this may result in unexpected outcome when filtering."
        )
        with pytest.warns(UserWarning, match=expect_warning_msg):
            _res = load_preprocessed_jobs_dataframe_from_duckdb(db_path=db_path, custom_query=query)
