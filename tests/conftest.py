#!This file is for configuring pytest, including preloading data
import pytest
import pandas as pd
import numpy as np

#loading small local data before every test modules run
@pytest.fixture
def load_small_data():
    data : pd.DataFrame = pd.read_csv("tests/mockData/mock1.csv")
    return data


@pytest.fixture
def small_sample_data():
    data = {
        'JobName' : ['job1', 'job2', 'job3', 'job4'],
        'UUID' : ['123456789', '123456789', '123456789', '123456789'],
        'ArrayID' : [np.nan, 1, 2, np.nan],
        'Interactive' : [np.nan, 'Matlab', np.nan, 'Matlab'],
        'Constraints' : [np.nan, ['some constraints'], np.nan, ['some constraints']],
        'GPUType' : [np.nan, 'v100', np.nan, 'v100'],
        'GPUs' : [np.nan, 1, np.nan, 4],
    }
    df = pd.DataFrame(data)
    return df