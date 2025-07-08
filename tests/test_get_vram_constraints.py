from src.preprocess import _get_vram_constraint
import pytest
import numpy as np
import pandas as pd


@pytest.fixture()
def mock_constraints():
    mock_data = {
        "Constraints": [
            [
                "'amd1900x'",
                "'gpu:v100'",
                "'vram23'",
                "'gpu:2080_ti'",
                "'gpu:2080'",
            ],  # [ 23, 11, 8, 16] -> max should be 23
            [
                "'amd1900x'",
                "'gpu:v100'",
                "'vram23'",
                "'gpu:2080_ti'",
                "'gpu:2080'",
            ],  # [ 23, 11, 8, 32] -> max should be 32
            np.nan,
            ["'vram23'", "'gpu:2080_ti'", "gpu:1080_ti", "'ib'"],
            ["'intel'", "'avx512'", "'vram23'", "'gpu:m40'", "'gpu:l40s'"],
            ["'gpu:2080_ti'", "gpu:1080_ti", "'ib'", "'gpu:m40'"],
            ["'amd7402'", "'amd7502'", "'amd7543'", "'gpu:titan_x'", "'gpu:gh200'"],
        ],
        "GPUs": [1, 1, 1, 3, 2, 1, 4],
        "GPUMemUsage": [729686000, 17179869200, 0, 729686000, 729686000, 800000000, 800000000],
    }
    return pd.DataFrame(mock_data)


def test_get_vram_constraints(mock_constraints):
    """
    Test the _get_vram_constraint function with mock constraints data.

    This test checks if the function correctly calculates the requested VRAM based on the constraints,
        number of GPUs, GPU memory usage, and check if it handles null values correctly.
    """
    data: pd.DataFrame = mock_constraints
    res = data.apply(lambda row: _get_vram_constraint(row["Constraints"], row["GPUs"], row["GPUMemUsage"]), axis=1)
    # expected values of vram_constraints, each entry is max requested_vram * GPUs_num (default of GPUs_num is 1)
    expect_vram_constraints = [
        23,
        32,
        np.nan,
        23 * 3,
        48 * 2,
        23,
        95 * 4,
    ]
    assert res.equals(pd.Series(expect_vram_constraints))
