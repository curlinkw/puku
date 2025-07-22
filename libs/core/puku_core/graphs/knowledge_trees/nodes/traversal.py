from typing import Generic, Optional
from langchain_core.load.serializable import Serializable

from puku_core.graphs.knowledge_trees.nodes.base import NodeType
from puku_core.graphs.knowledge_trees.edges import EdgeWithMetadata
from puku_core.graphs.knowledge_trees.edges.metadata import EdgeMetadataType


class TraversalNode(Serializable, Generic[NodeType, EdgeMetadataType]):
    """A Node with metadata returned during various traversals of the tree"""

    node: NodeType
    """Node itself"""
    parent: Optional[NodeType] = None
    """None if it is the root, otherwise parent node"""
    edge: Optional[EdgeWithMetadata[NodeType, EdgeMetadataType]] = None
    """None if it is the root, otherwise parent->self edge"""
