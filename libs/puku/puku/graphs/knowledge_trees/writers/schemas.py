from typing import Dict
from pydantic import BaseModel

from puku_core.documents.markdown import parse
from puku_core.graphs.knowledge_trees.nodes.markdown import (
    MarkdownNode,
    MarkdownDocument,
)
from puku_core.graphs.knowledge_trees.edges.base import BaseEdge
from puku_core.graphs.knowledge_trees.types import MarkdownNodeAmendmentPropagation


class AmendmentSplit(BaseModel):
    """Amendment split among children and node"""

    node: str
    """Node amendment"""

    children: Dict[int, str]
    """Children amendment mapping, edge ID is used as key"""

    def to_propagation(self, node: MarkdownNode) -> MarkdownNodeAmendmentPropagation:
        raw_children: Dict[BaseEdge[MarkdownNode], MarkdownDocument] = {}
        for edge in node.children_without_metadata:
            _hash = hash(edge)
            if _hash in self.children:
                raw_children[edge] = parse(self.children[_hash])

        return MarkdownNodeAmendmentPropagation(
            node=parse(self.node), children=raw_children
        )
