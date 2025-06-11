#!This file is for configuring pytest, including preloading data
import pytest
import pandas as pd
import numpy as np

#loading small local data before every test modules run
@pytest.fixture
def load_small_data():
    data : pd.DataFrame = pd.read_csv("data/slurm_data_small.csv")
    return data

@pytest.fixture
def small_sample_data():
    data = {
        'ArrayID' : [np.nan, 1, 2, np.nan],
        'Interactive' : [np.nan, 'Matlab', np.nan, 'Matlab'],
        'Constraints' : [np.nan, '[]', np.nan, '[]'],
    }
    df = pd.DataFrame(data)
    return df