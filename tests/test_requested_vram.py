from src.preprocess import get_requested_vram
import pytest
import numpy as np
import pandas as pd


@pytest.fixture()
def mock_constraints():
    mock_data = {
        "Constraints": [
            np.array(
                [
                    "'amd1900x'",
                    "'gpu:v100'",
                    "'vram23'",
                    "'gpu:2080_ti'",
                    "'gpu:2080'",
                ]
            ),  # [ 23, 11, 8, 16] -> max should be 23
            np.array(
                [
                    "'amd1900x'",
                    "'gpu:v100'",
                    "'vram23'",
                    "'gpu:2080_ti'",
                    "'gpu:2080'",
                ]
            ),  # [ 23, 11, 8, 32] -> max should be 32
            np.nan,
            np.array(["'vram23'", "'gpu:2080_ti'", "gpu:1080_ti", "'ib'"]),
            np.array(["'intel'", "'avx512'", "'vram23'", "'gpu:m40'", "'gpu:l40s'"]),
            np.array(["'gpu:2080_ti'", "gpu:1080_ti", "'ib'", "'gpu:m40'"]),
            np.array(["'amd7402'", "'amd7502'", "'amd7543'", "'gpu:titan_x'", "'gpu:gh200'"]),
        ],
        "GPUs": np.array([1, 1, 1, 3, 2, 1, 4]),
        "GPUMemUsage": np.array([729686000, 17179869200, 0, 729686000, 729686000, 800000000, 800000000]),
    }

    return pd.DataFrame(mock_data)


def test_get_vram_constraints(mock_constraints):
    """
    Test the get_requested_vram function with mock constraints data.
    This test checks if the function correctly calculates the requested VRAM based on the constraints,
        number of GPUs, GPU memory usage, and check if it handles null values correctly.
    """
    data: pd.DataFrame = mock_constraints
    res = data.apply(lambda row: get_requested_vram(row["Constraints"], row["GPUs"], row["GPUMemUsage"]), axis=1)
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
