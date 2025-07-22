"""Similar name as for **true** species"""

from typing import List, Self, Generic
from pydantic import Field

from puku_core.graphs.knowledge_trees.edges.metadata import (
    EdgeWithMetadata,
    EdgeMetadataType,
)
from puku_core.graphs.knowledge_trees.nodes.traversal import TraversalNode
from puku_core.graphs.knowledge_trees.nodes.base import BaseNode


class TrueTreeNode(BaseNode, Generic[EdgeMetadataType]):
    children: List[EdgeWithMetadata[Self, EdgeMetadataType]] = Field(
        default_factory=list
    )

    def descendants(self) -> List[TraversalNode[Self, EdgeMetadataType]]:
        """Return descendants in breadth-first search (BFS) traversal order."""
        traversal: List[TraversalNode] = [TraversalNode(node=self)]
        queue: List[Self] = [self]

        while queue:
            current_node = queue.pop(0)
            for edge_with_metadata in current_node.children:
                traversal.append(
                    TraversalNode(
                        node=edge_with_metadata.edge.child,
                        parent=current_node,
                        edge=edge_with_metadata,
                    )
                )
                queue.append(edge_with_metadata.edge.child)

        return traversal

    def __hash__(self) -> int:
        return super().__hash__()
