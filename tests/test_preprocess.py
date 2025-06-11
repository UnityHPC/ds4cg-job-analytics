import pytest
import numpy as np
from src import preprocess_data, fill_missing

def test_preprocess_data_loading_total_data(load_small_data):
    data = load_small_data
    assert len(data) == 34200

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

# def test_pre_process_data_fill_missing_small_arrayID(small_sample_data):
#     data = small_sample_data
#     fill_missing(data)
#     assert data["ArrayID"].isnull().sum() == 0
#     assert data["ArrayID"].values.all(np.array([-1, 1, 2, -1]))