import pandas as pd
from preprocess.preprocess import get_estimated_allocated_vram


# Test for single node VRAM allocation
def test_allocated_vram_single_node():
    mock_data = pd.DataFrame({
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
    })

    expected_allocated_vram = [40, 16, 64, 320, 11, 96]

    mock_data["AllocatedVRAM"] = mock_data.apply(
        lambda row: get_estimated_allocated_vram(row["GPUType"], row["NodeList"], row["GPUs"]), axis=1
    )

    assert mock_data["AllocatedVRAM"].tolist() == expected_allocated_vram
    print("Test passed: Allocated VRAM for single node calculated correctly.")


# Test for multiple nodes VRAM allocation
def test_allocated_vram_multiple_nodes():
    mock_data = pd.DataFrame({
        "GPUType": [["a100", "v100"], ["v100", "2080_ti"], ["a40", "a100"]],
        "NodeList": [
            ["ece-gpu001", "gpu001"],  # A100 with 40GB, V100 with 16GB
            ["power9-gpu008", "node001"],  # V100 with 32GB, 2080 Ti with 11GB
            ["node002", "gpu042"],  # A40 with 48GB, A100 with 80GB
        ],
        "GPUs": [2, 2, 2],
    })

    expected_allocated_vram = [56, 43, 128]

    mock_data["AllocatedVRAM"] = mock_data.apply(
        lambda row: get_estimated_allocated_vram(row["GPUType"], row["NodeList"], row["GPUs"]), axis=1
    )

    assert mock_data["AllocatedVRAM"].tolist() == expected_allocated_vram
    print("Test passed: Allocated VRAM for multiple nodes calculated correctly.")


# Run the tests
test_allocated_vram_single_node()
test_allocated_vram_multiple_nodes()
