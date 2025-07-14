from typing import Dict
from pydantic import BaseModel

from puku_core.graphs.knowledge_trees.nodes import MarkdownNode
from puku_core.graphs.knowledge_trees.edges import BaseEdge
from puku_core.documents.markdown import MarkdownDocument


class MarkdownNodeUpdateRequest(BaseModel):
    """Knowledge tree node update request."""

    node: MarkdownNode
    """Root node in the knowledge tree"""

    amendment: MarkdownDocument
    """Information to be stored"""


class MarkdownNodeAmendmentPropagation(BaseModel):
    """Propagation of amendment among children"""

    node_amendment: MarkdownDocument
    """Node amendment"""

    children_amendment: Dict[BaseEdge, MarkdownDocument]
    """Children amendment mapping"""
