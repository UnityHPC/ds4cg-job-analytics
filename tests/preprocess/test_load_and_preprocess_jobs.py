import pytest
import pandas
from src.utilities import load_and_preprocess_jobs, load_and_preprocess_jobs_custom_query
from ..conftest import preprocess_mock_data
from src.config.enum_constants import OptionalColumnsEnum, RequiredColumnsEnum
from datetime import datetime, timedelta


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_return_correct_types(mock_data_path: str) -> None:
    """
    Basic test on return type of function
    """
    res = load_and_preprocess_jobs(db_path=mock_data_path)
    assert isinstance(res, pandas.DataFrame)


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_no_filter(mock_data_path: str) -> None:
    """
    Test in case there is no filtering, function should return every valid records from database.
    """
    ground_truth = preprocess_mock_data(mock_data_path, min_elapsed_seconds=0)
    res = load_and_preprocess_jobs(db_path=mock_data_path)
    expect_num_records = len(ground_truth)
    assert expect_num_records == len(res)


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
@pytest.mark.parametrize("dates_back", [90, 150])
def test_filter_date_back(mock_data_path: str, dates_back: int) -> None:
    """
    Test for filtering by dates_back.

    Test with multiple different dates_back for higher test coverage.
    """
    temp = preprocess_mock_data(mock_data_path, min_elapsed_seconds=0)
    res = load_and_preprocess_jobs(db_path=mock_data_path, days_back=dates_back)
    cutoff = datetime.now() - timedelta(days=dates_back)
    ground_truth_jobs = temp[(temp["StartTime"] >= cutoff)].copy()
    expect_job_ids = ground_truth_jobs["JobID"].to_numpy()
    assert len(ground_truth_jobs) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_filter_min_elapsed(mock_data_path: str) -> None:
    """
    Test for filtering by days back and minimum elapsed time.
    """
    temp = preprocess_mock_data(mock_data_path, min_elapsed_seconds=13000)
    res = load_and_preprocess_jobs(db_path=mock_data_path, min_elapsed_seconds=13000, days_back=90)
    cutoff = datetime.now() - timedelta(days=90)
    ground_truth_jobs = temp[(temp["StartTime"] >= cutoff)].copy()
    expect_job_ids = ground_truth_jobs["JobID"].to_numpy()
    assert len(ground_truth_jobs) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_filter_date_back_include_all(mock_data_path: str) -> None:
    """
    Test for filtering by days_back, including CPU only jobs and FAILED/ CANCELLED jobs
    """
    temp = preprocess_mock_data(
        mock_data_path,
        min_elapsed_seconds=0,
        include_cpu_only_jobs=True,
        include_custom_qos_jobs=True,
        include_failed_cancelled_jobs=True,
    )
    res = load_and_preprocess_jobs(
        db_path=mock_data_path,
        days_back=90,
        min_elapsed_seconds=0,
        include_cpu_only_jobs=True,
        include_failed_cancelled_jobs=True,
        include_custom_qos_jobs=True,
    )
    cutoff = datetime.now() - timedelta(days=90)
    ground_truth_jobs = temp[temp["StartTime"] >= cutoff]
    expect_job_ids = ground_truth_jobs["JobID"].to_numpy()
    assert len(ground_truth_jobs) == len(res)
    for id in res["JobID"]:
        assert id in expect_job_ids


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
@pytest.mark.parametrize("missing_col", [col.value for col in RequiredColumnsEnum])
def test_missing_required_columns_error_raised(mock_data_path: str, missing_col: str) -> None:
    """
    Test enforcement of errors when the database is missing a required column.

    Expect to raise RuntimeError for any of these columns if they are missing in the dataframe.
    """
    required_col = {e.value for e in RequiredColumnsEnum}
    col_names = required_col.copy()
    col_names.remove(missing_col)
    col_str = ", ".join(col_names)
    query = f"SELECT {col_str} FROM Jobs"
    with pytest.raises(
        RuntimeError, match=f"Failed to load jobs DataFrame. 'Column {missing_col} does not exist in dataframe.'"
    ):
        _res = load_and_preprocess_jobs_custom_query(db_path=mock_data_path, custom_query=query)


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
@pytest.mark.parametrize("missing_col", [col.value for col in OptionalColumnsEnum])
def test_optional_column_warnings(mock_data_path: str, recwarn: pytest.WarningsRecorder, missing_col: str) -> None:
    """
    Test handling the dataframe loads from database when missing one of the columns

    These columns are not in ENFORCE_COLUMNS so only warnings are expected to be raised
    """
    optional_columns = {e.value for e in OptionalColumnsEnum}
    required_columns = {e.value for e in RequiredColumnsEnum}

    required_column_copy = required_columns.copy()
    optional_column_copy = optional_columns.copy()
    optional_column_copy.remove(missing_col)
    final_column_set = required_column_copy.union(optional_column_copy)
    col_str = ", ".join(final_column_set)
    query = f"SELECT {col_str} FROM Jobs"

    expect_warning_msg = (
        f"Column '{missing_col}' is missing from the dataframe. "
        "This may impact filtering operations and downstream processing."
    )
    _res = load_and_preprocess_jobs_custom_query(db_path=mock_data_path, custom_query=query)

    # Check that a warning was raised with the expected message
    assert len(recwarn) > 0
    assert str(recwarn[0].message) == expect_warning_msg


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_custom_query(
    mock_data_frame: pandas.DataFrame, mock_data_path: str, recwarn: pytest.WarningsRecorder
) -> None:
    """
    Test if function fetches expected records when using custom sql query.

    Warnings are ignored since test_optional_column_warnings and test_missing_required_columns_error_raised
        covers warning for optional columns missing.
    """
    query = (
        "SELECT JobID, GPUType, Constraints, StartTime, SubmitTime, "
        "NodeList, GPUs, GPUMemUsage, CPUMemUsage, Elapsed, Partition "
        "FROM Jobs WHERE Status != 'CANCELLED' AND Status !='FAILED' "
        "AND ArrayID is not NULL AND Interactive is not NULL"
    )
    res = load_and_preprocess_jobs_custom_query(db_path=mock_data_path, custom_query=query)
    ground_truth_jobs = mock_data_frame[
        (mock_data_frame["Status"] != "CANCELLED")
        & (mock_data_frame["Status"] != "FAILED")
        & (mock_data_frame["ArrayID"].notna())
        & (mock_data_frame["Interactive"].notna())
    ].copy()
    assert len(res) == len(ground_truth_jobs)
    expect_ids = ground_truth_jobs["JobID"].to_list()
    for id in res["JobID"]:
        assert id in expect_ids


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
@pytest.mark.parametrize("days_back", [90, 150])
def test_custom_query_days_back(
    mock_data_frame: pandas.DataFrame, mock_data_path: str, recwarn: pytest.WarningsRecorder, days_back: int
) -> None:
    """
    Test custom query with dates_back condition.

    Expect the result will be filtered by dates_back condition in the query.

    Warnings are ignored since test_optional_column_warnings and test_missing_required_columns_error_raised
        covers test warning for optional columns missing.
    """
    cutoff = datetime.now() - timedelta(days=days_back)
    query = (
        "SELECT JobID, GPUType, Constraints, StartTime, SubmitTime, "
        "NodeList, GPUs, GPUMemUsage, CPUMemUsage, Elapsed, Partition "
        "FROM Jobs WHERE Status != 'CANCELLED' AND Status !='FAILED' AND ArrayID is not NULL "
        f"AND Interactive is not NULL AND StartTime >= '{cutoff}'"
    )
    res = load_and_preprocess_jobs_custom_query(db_path=mock_data_path, custom_query=query)

    ground_truth_jobs = mock_data_frame[
        (mock_data_frame["Status"] != "CANCELLED")
        & (mock_data_frame["Status"] != "FAILED")
        & (mock_data_frame["ArrayID"].notna())
        & (mock_data_frame["Interactive"].notna())
        & (mock_data_frame["StartTime"] >= cutoff)
    ].copy()
    expect_ids = ground_truth_jobs["JobID"].to_list()

    assert len(res) == len(ground_truth_jobs)
    for id in res["JobID"]:
        assert id in expect_ids


@pytest.mark.parametrize("mock_data_path", [False, True], ids=["false_case", "true_case"], indirect=True)
def test_empty_dataframe_warning(mock_data_path: str, recwarn: pytest.WarningsRecorder) -> None:
    """
    Test handling the dataframe loads from database when the result is empty.

    Expect a UserWarning to be raised with the appropriate message.
    """
    # Query that returns no rows
    query = "SELECT * FROM Jobs WHERE 1=0"
    res = load_and_preprocess_jobs_custom_query(db_path=mock_data_path, custom_query=query)
    assert res.empty
    # Check that the warning is about empty dataframe
    assert len(recwarn) == 1
    assert str(recwarn[0].message) == "Dataframe results from database and filtering is empty."
