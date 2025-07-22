from puku_core.graphs.knowledge_trees.edges.base import BaseEdge
from puku_core.graphs.knowledge_trees.nodes.markdown import MarkdownNode
from puku_core.graphs.knowledge_trees.edges.metadata import (
    MarkdownEdgeMetadata,
    EdgeWithMetadata,
)


class MarkdownEdge(EdgeWithMetadata[MarkdownNode, MarkdownEdgeMetadata]):
    def __init__(self, child: MarkdownNode):
        super().__init__(edge=BaseEdge(child=child), metadata=MarkdownEdgeMetadata())
