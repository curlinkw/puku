from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
from pydantic import Field
from langchain_core.load.serializable import Serializable

if TYPE_CHECKING:
    from puku_core.graphs.knowledge_trees.edges import BaseEdge
    from puku_core.documents.markdown import MarkdownDocument


class BaseNode(Serializable):
    children: List[BaseEdge] = Field(default_factory=list)

    def descendants(self) -> List[TraversalNode]:
        """Return descendants in breadth-first search (BFS) traversal order."""
        traversal: List[TraversalNode] = [TraversalNode(node=self)]
        queue: List[BaseNode] = [self]

        while queue:
            current_node = queue.pop(0)
            for edge in current_node.children:
                traversal.append(
                    TraversalNode(node=edge.child, parent=current_node, edge=edge)
                )
                queue.append(edge.child)

        return traversal

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


class TraversalNode(Serializable):
    """A Node with metadata returned during various traversals of the tree"""

    node: BaseNode
    """Node itself"""
    parent: Optional[BaseNode] = None
    """None if it is the root, otherwise parent node"""
    edge: Optional[BaseEdge] = None
    """None if it is the root, otherwise parent->self edge"""


class MarkdownNode(BaseNode):
    """Node that stores markdown data"""

    description: str
    """What to store inside"""

    data: MarkdownDocument
    """Stored data"""
