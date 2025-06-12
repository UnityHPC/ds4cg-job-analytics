import pytest
import numpy as np
from src import preprocess_data, fill_missing


def test_preprocess_data_filtred_columns_total_data(load_small_data):
    data = preprocess_data(data=load_small_data, min_elapsed=600)
    assert "UUID" not in data.columns 
    assert "JobName" not in data.columns

def test_pre_preocess_data_filtered_GPU_total_data(load_small_data):
    data = preprocess_data(data=load_small_data, min_elapsed=600)
    GPUTypeNull = data["GPUType"].isnull()
    assert not any(GPUTypeNull)
    GPUNull = data["GPUs"].isnull()
    assert not any(GPUNull)

def test_pre_process_data_filtered_status_total_data(load_small_data):
    data = preprocess_data(data=load_small_data, min_elapsed=600)
    statusFailed = data["Status"] == "FAILED"
    statusCancelled = data["Status"] == "CANCELLED"
    assert not any(statusFailed)
    assert not any(statusCancelled)

def test_pre_preprocess_data_include_CPU_job(load_small_data):
    data = preprocess_data(data=load_small_data, min_elapsed=600, include_CPU_only_job=True)
    assert data["GPUType"].value_counts()['CPU'] == 2

def test_pre_process_data_include_FAILED_CANCELLED_job(load_small_data):
    data = preprocess_data(data=load_small_data, min_elapsed=600, include_failed_cancelled_jobs=True)
    assert data["Status"].value_counts()['FAILED'] == 1
    print(data["Status"].value_counts())
    assert 'CANCELLED' not in data["Status"].value_counts() 

def test_pre_process_data_include_all(load_small_data):
    data = preprocess_data(data=load_small_data, min_elapsed=600, include_failed_cancelled_jobs=True, include_CPU_only_job=True)
    assert len(data) == 11
    assert data["GPUType"].value_counts()['CPU'] == 6
    assert data["GPUs"].value_counts()[0] == 6
    assert data["Status"].value_counts()["FAILED"] == 3
    assert data["Status"].value_counts()["CANCELLED"] == 2
    assert data["Status"].value_counts()["COMPLETED"] == 6

def test_pre_process_data_filtered_elapsed_total_data(load_small_data):
    data = preprocess_data(data=load_small_data, min_elapsed=300)
    elapsedLessThanMin = data["Elapsed"] < 300
    assert not any(elapsedLessThanMin)

def test_pre_process_data_filtered_account_total_data(load_small_data):
    data = preprocess_data(data=load_small_data, min_elapsed=600)
    accountRoot = data["Account"] == "root"
    assert not any(accountRoot)

def test_pre_process_data_filtered_partition_total_data(load_small_data):
    data = preprocess_data(data=load_small_data, min_elapsed=600)
    partitionBuilding = data["Partition"] == "building"
    assert not any(partitionBuilding)

def test_pre_process_data_filtered_qos_total_data(load_small_data):
    data = preprocess_data(data=load_small_data, min_elapsed=600)
    qosUpdates = data["QOS"] == "updates"
    assert not any(qosUpdates)


def test_pre_process_data_fill_missing_small_arrayID(small_sample_data):
    data = small_sample_data
    fill_missing(data)
    assert data["ArrayID"].isnull().sum() == 0
    assert data["ArrayID"].tolist() == [-1, 1, 2, -1]

def test_pre_process_data_fill_missing_small_interactive(small_sample_data):
    data = small_sample_data
    fill_missing(data)
    assert data["Interactive"].isnull().sum() == 0
    assert data["Interactive"].tolist() == ["", "Matlab", "", "Matlab"]


# def test_pre_process_data_fill_missing_small_constraints(small_sample_data):
#     data = small_sample_data
#     fill_missing(data)
#     assert data["Constraints"].isnull().sum() == 0
#     assert data["Constraints"].tolist() == ["", ['some constraints'], "", ['some constraints']]


def test_pre_process_data_fill_missing_small_GPUType(small_sample_data):
    data = small_sample_data
    fill_missing(data)
    assert data["GPUType"].isnull().sum() == 0
    assert data["GPUType"].tolist() == ["CPU", "v100", "CPU", "v100"]

def test_pre_process_data_fill_missing_small_GPUs(small_sample_data):
    data = small_sample_data
    fill_missing(data)
    assert data["GPUs"].isnull().sum() == 0
    assert data["GPUs"].tolist() == [0, 1, 0, 4]