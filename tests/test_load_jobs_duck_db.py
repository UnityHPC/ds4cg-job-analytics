import pandas
from src.analysis import load_jobs_dataframe_from_duckdb
from .conftest import helper_filter_irrelevant_records
from src.config.enum_constants import StatusEnum
from datetime import datetime, timedelta


"""
Test for load_jobs_dataframe_from_duckdb.
This test version assume that load_jobs_dataframe_from_duckdb will always ignore cpu only jobs 
    and failed/ cancelled jobs.
"""


# TODO: update these tests when there are new updates on other branches on the load_jobs_dataframe_from_duckdb()
def test_load_jobs_correct_types(mock_data):
    res = load_jobs_dataframe_from_duckdb(db_path=mock_data[1])
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
    res = load_jobs_dataframe_from_duckdb(db_path=db_path)
    expect_num_records = len(ground_truth_csv)
    assert expect_num_records == len(res)


def test_load_jobs_filter_day_back_1(mock_data):
    """
    Test for filtering by days_back
    """
    mock_csv, db_path = mock_data
    temp = helper_filter_irrelevant_records(mock_csv, 0)
    res = load_jobs_dataframe_from_duckdb(db_path=db_path, days_back=90)
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
    res = load_jobs_dataframe_from_duckdb(db_path=db_path, days_back=20)
    cutoff = datetime.now() - timedelta(days=20)
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
