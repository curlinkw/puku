from __future__ import annotations

from typing import Generic, TypeVar
from langchain_core.load.serializable import Serializable

from puku_core.load.hashable import UUIDHashable
from puku_core.graphs.knowledge_trees.nodes.base import NodeType


class BaseEdge(UUIDHashable, Serializable, Generic[NodeType]):
    """Base edge generic for storing edge-based structural information about the tree.

        It is not yet clear what structural information may be needed in the edges. \
            But it is clear that it will redefine the BaseNode with strict TypeEdge typing (without generic)
    """

    child: NodeType

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


EdgeType = TypeVar("EdgeType", bound=BaseEdge)
