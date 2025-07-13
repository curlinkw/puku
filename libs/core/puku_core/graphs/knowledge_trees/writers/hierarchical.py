from typing import Any
from puku_core.documents.markdown import MarkdownDocument
from puku_core.graphs.knowledge_trees.nodes import MarkdownNode, TraversalNode
from puku_core.graphs.knowledge_trees.writers.base import BaseMarkdownKnowledgeWriter


class HierarchicalKnowledgeWriter(BaseMarkdownKnowledgeWriter):
    """Knowledge writer which writes memory hierarchically from top to bottom"""

    def write(self, node: MarkdownNode, amendment: MarkdownDocument) -> Any:
        traversal: list[TraversalNode] = node.descendants()
