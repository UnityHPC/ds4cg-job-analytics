import pandas as pd
from src.preprocess.preprocess import _get_approx_allocated_vram


# Test for single node VRAM allocation
def test_approx_allocated_vram_single_node():
    gpu_usage = [20, 10, 50, 200, 5, 80]  # in GB
    gpu_usage_bytes = [usage * (2**30) for usage in gpu_usage]  # convert to bytes
    mock_data = pd.DataFrame({
        "JobID": [1, 2, 3, 4, 5, 6],
        "GPUType": [["a100"], ["v100"], ["v100"], ["a100"], ["2080_ti"], ["a40"]],
        "NodeList": [
            ["ece-gpu001"],  # A100 with 40GB
            ["gpu001"],  # V100 with 16GB
            ["power9-gpu008"],  # V100 with 32GB
            ["gpu042"],  # A100 with 80GB
            ["node001"],  # 2080 Ti with 11GB
            ["node002"],  # A40 with 48GB
        ],
        "GPUs": [1, 1, 2, 4, 1, 2],
        "GPUMemUsage": gpu_usage_bytes,
    })

    expected_allocated_vram = [40, 16, 64, 320, 11, 96]

    mock_data["AllocatedVRAM"] = mock_data.apply(
        lambda row: _get_approx_allocated_vram(
            row["JobID"], row["GPUType"], row["NodeList"], row["GPUs"], row["GPUMemUsage"]
        ),
        axis=1,
    )

    assert mock_data["AllocatedVRAM"].tolist() == expected_allocated_vram


# Test for jobs with mixed nodes and VRAM usage below minimum
def test_approx_allocated_vram_mixed_nodes_below_minimum():
    gpu_usage = [100]
    gpu_usage_bytes = [usage * (2**30) for usage in gpu_usage]  # convert to bytes
    mock_data = pd.DataFrame({
        "JobID": [1],
        "GPUType": [["a100"]],
        "NodeList": [["ece-gpu001", "gpu014"]],
        "GPUs": [3],
        "GPUMemUsage": gpu_usage_bytes,
    })

    expected_allocated_vram = [120]  # Minimum VRAM (40 GB per GPU) * 3 GPUs

    mock_data["AllocatedVRAM"] = mock_data.apply(
        lambda row: _get_approx_allocated_vram(
            row["JobID"], row["GPUType"], row["NodeList"], row["GPUs"], row["GPUMemUsage"]
        ),
        axis=1,
    )

    assert mock_data["AllocatedVRAM"].tolist() == expected_allocated_vram


# Test for jobs with mixed nodes and VRAM usage exceeding minimum
def test_approx_allocated_vram_mixed_nodes_exceeding_minimum():
    gpu_usage = [150]  # in GB
    gpu_usage_bytes = [usage * (2**30) for usage in gpu_usage]  # convert to bytes
    mock_data = pd.DataFrame({
        "JobID": [1],
        "GPUType": [["a100"]],
        "NodeList": [["ece-gpu001", "gpu014"]],
        "GPUs": [3],
        "GPUMemUsage": gpu_usage_bytes,
    })

    expected_allocated_vram = [240]  # Higher VRAM (80 GB per GPU) * 3 GPUs

    mock_data["AllocatedVRAM"] = mock_data.apply(
        lambda row: _get_approx_allocated_vram(
            row["JobID"], row["GPUType"], row["NodeList"], row["GPUs"], row["GPUMemUsage"]
        ),
        axis=1,
    )

    assert mock_data["AllocatedVRAM"].tolist() == expected_allocated_vram


def test_multivalent_vram_allocation_a100_only():
    gpu_usage = [100 * (2**30)]  # 100 GiB usage
    mock_data = pd.DataFrame({
        "JobID": [1],
        "GPUType": [{"a100": 3}],  # total 3 A100s
        "NodeList": [["ece-gpu001", "ece-gpu002"]],  # 2 A100 nodes
        "GPUs": [3],
        "GPUMemUsage": gpu_usage,
    })

    # Estimate: 2 nodes * 40GB = 80GB from multivalent lookup
    # 100GB usage → we need at least 20GB more → another 40GB A100
    expected_allocated_vram = [120]

    mock_data["AllocatedVRAM"] = mock_data.apply(
        lambda row: _get_approx_allocated_vram(
            row["JobID"], row["GPUType"], row["NodeList"], row["GPUs"], row["GPUMemUsage"]
        ),
        axis=1,
    )

    assert mock_data["AllocatedVRAM"].tolist() == expected_allocated_vram
