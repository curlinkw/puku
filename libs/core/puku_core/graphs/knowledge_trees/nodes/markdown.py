from puku_core.documents.markdown import MarkdownDocument
from puku_core.graphs.knowledge_trees.nodes.true import TrueTreeNode
from puku_core.graphs.knowledge_trees.edges.metadata import MarkdownEdgeMetadata


class MarkdownNode(TrueTreeNode[MarkdownEdgeMetadata]):
    """Node that stores markdown data"""

    title: str
    """For use as reference"""

    description: str
    """What to store inside"""

    data: MarkdownDocument
    """Stored data"""

    def __hash__(self) -> int:
        return super().__hash__()
