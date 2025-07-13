from typing import Any
from langchain_core.load.serializable import Serializable

from puku_core.graphs.knowledge_trees.nodes import BaseNode


class BaseKnowledgeReader(Serializable):
    def read(self, node: BaseNode, *args, **kwargs) -> Any:
        raise NotImplementedError
