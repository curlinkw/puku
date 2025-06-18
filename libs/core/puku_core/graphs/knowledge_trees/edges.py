from __future__ import annotations

from typing import TYPE_CHECKING
from langchain_core.load.serializable import Serializable

if TYPE_CHECKING:
    from puku_core.graphs.knowledge_trees.nodes import BaseNode


class BaseEdge(Serializable):
    child: BaseNode
