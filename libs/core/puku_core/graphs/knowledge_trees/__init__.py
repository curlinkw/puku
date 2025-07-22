from puku_core.documents.markdown import MarkdownDocument
from puku_core.graphs.knowledge_trees.edges import EdgeWithMetadata
from puku_core.graphs.knowledge_trees.edges.base import BaseEdge
from puku_core.graphs.knowledge_trees.nodes.base import BaseNode
from puku_core.graphs.knowledge_trees.nodes.markdown import MarkdownNode
from puku_core.graphs.knowledge_trees.nodes.traversal import TraversalNode

BaseEdge.model_rebuild()
BaseNode.model_rebuild()
EdgeWithMetadata.model_rebuild()
TraversalNode.model_rebuild()
MarkdownNode.model_rebuild()
