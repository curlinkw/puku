from typing import Dict, Generic, TypeVar
from pydantic import BaseModel

from puku_core.graphs.knowledge_trees.nodes.base import NodeType
from puku_core.graphs.knowledge_trees.nodes.markdown import MarkdownNode
from puku_core.graphs.knowledge_trees.edges.base import EdgeType, BaseEdge
from puku_core.documents.markdown import MarkdownDocument


DataType = TypeVar("DataType", bound=BaseModel)


class UpdateRequest(BaseModel, Generic[NodeType, DataType]):
    """Knowledge tree node update request."""

    node: NodeType
    """Root node in the knowledge tree"""

    amendment: DataType
    """Information to be stored"""


MarkdownUpdateRequest = UpdateRequest[MarkdownNode, MarkdownDocument]


class NodeAmendmentPropagation(BaseModel, Generic[EdgeType, DataType]):
    """Propagation of amendment among children"""

    node: DataType
    """Node amendment"""

    children: Dict[EdgeType, DataType]
    """Children amendment mapping"""


MarkdownNodeAmendmentPropagation = NodeAmendmentPropagation[
    BaseEdge[MarkdownNode], MarkdownDocument
]
