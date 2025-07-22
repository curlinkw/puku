from __future__ import annotations

from typing import Generic, TYPE_CHECKING, TypeVar
from pydantic import Field
from langchain_core.load.serializable import Serializable
from puku_core.graphs.knowledge_trees.nodes.base import NodeType

if TYPE_CHECKING:
    from puku_core.graphs.knowledge_trees.edges.base import BaseEdge


class BaseEdgeMetadata(Serializable):
    pass


EdgeMetadataType = TypeVar("EdgeMetadataType", bound=BaseEdgeMetadata, covariant=True)


class EdgeWithMetadata(Serializable, Generic[NodeType, EdgeMetadataType]):
    """Wrapper class combining edge and metadata"""

    edge: BaseEdge[NodeType]
    metadata: EdgeMetadataType

    def __hash__(self) -> int:
        return hash(self.edge)


class MarkdownEdgeMetadata(BaseEdgeMetadata):
    pass
