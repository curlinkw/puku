import pytest
from langchain_core.load import loads, dumps

from puku_core.graphs.knowledge_trees.nodes import BaseNode


class IndexNode(BaseNode):
    idx: int


@pytest.mark.filterwarnings("ignore::Warning")
def test_serialization():
    node = IndexNode(idx=1)
    assert loads(dumps(node), valid_namespaces=["test_serialization"]).idx == 1
