from __future__ import annotations

from typing import TYPE_CHECKING
from langchain_core.load.serializable import Serializable

from puku_core.load.hashable import UUIDHashable

if TYPE_CHECKING:
    from puku_core.graphs.knowledge_trees.nodes import BaseNode


class BaseEdge(UUIDHashable, Serializable):
    child: BaseNode

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True
