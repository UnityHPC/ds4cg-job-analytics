import pytest
from src.config.remote_config import NodeInfoFetcher
from src.config.enum_constants import NodeInfoKeyEnum
from src.warnings import NodeNotFoundWarning
from pathlib import Path
import json
import tempfile
import shutil


class DummyNodeInfoFetcher(NodeInfoFetcher):
    """Dummy fetcher for testing NodeInfoFetcher with static mock data."""

    @property
    def local_path(self):
        """Return path to static mock node info file."""
        return Path("tests/mock_data/mock_remote_configs/node_info.json")


@pytest.fixture
def dummy_node_info_fetcher():
    return DummyNodeInfoFetcher()


def test_get_node_memory_valid(dummy_node_info_fetcher):
    node_info = dummy_node_info_fetcher.get_info()
    found = False
    for batch in node_info:
        nodes = batch.get(NodeInfoKeyEnum.NODES.value)
        ram = batch.get(NodeInfoKeyEnum.RAM.value)
        if nodes and ram is not None:
            node_name = nodes[0]
            node_memory = NodeInfoFetcher.get_node_memory(node_name, node_info)
            assert node_memory == ram, (
                f"Memory for node '{node_name}' with a value of {node_memory} does not match expected value of {ram}."
            )
            found = True
            break
    assert found, f"No valid node with '{NodeInfoKeyEnum.RAM.value}' memory found in mock data."


def test_get_node_memory_node_not_found(dummy_node_info_fetcher):
    node_info = dummy_node_info_fetcher.get_info()
    all_nodes = set()
    for batch in node_info:
        all_nodes.update(batch.get(NodeInfoKeyEnum.NODES.value))
    missing_node = "definitely_not_a_real_node"
    while missing_node in all_nodes:
        missing_node += "_x"
    with pytest.warns(NodeNotFoundWarning, match=f"Node '{missing_node}' not found"):
        NodeInfoFetcher.get_node_memory(missing_node, node_info)


def test_get_node_memory_missing_ram(dummy_node_info_fetcher):
    node_info = dummy_node_info_fetcher.get_info()
    for batch in node_info:
        nodes = batch.get(NodeInfoKeyEnum.NODES.value)
        if nodes and NodeInfoKeyEnum.RAM.value not in batch:
            node_name = nodes[0]
            with pytest.raises(
                ValueError, match="Each node info dictionary must contain the keys listed in NodeInfoKeyEnum."
            ):
                NodeInfoFetcher.get_node_memory(node_name, node_info)
            break


@pytest.mark.parametrize("missing_key", [e.value for e in NodeInfoKeyEnum])
def test_get_node_memory_missing_any_key(dummy_node_info_fetcher, missing_key, monkeypatch):
    node_info = dummy_node_info_fetcher.get_info()
    temp_dir = tempfile.mkdtemp()
    temp_path = f"{temp_dir}/node_info_missing_key.json"
    for i, batch in enumerate(node_info):
        nodes = batch.get(NodeInfoKeyEnum.NODES.value)
        if nodes and missing_key in batch:
            node_info[i].pop(missing_key)
            node_name = nodes[0]
            with open(temp_path, "w") as tf:
                json.dump(node_info, tf)
            monkeypatch.setattr(type(dummy_node_info_fetcher), "local_path", property(lambda self: Path(temp_path)))
            with pytest.raises(
                ValueError, match="Each node info dictionary must contain the keys listed in NodeInfoKeyEnum."
            ):
                NodeInfoFetcher.get_node_memory(node_name, node_info, offline=True)
            temp_path_obj = Path(temp_path)
            if temp_path_obj.exists():
                temp_path_obj.unlink()
            break
    shutil.rmtree(temp_dir)
