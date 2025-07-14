from typing import Any
from langchain_core.runnables.config import RunnableConfig

from puku_core.documents.markdown import MarkdownDocument
from puku_core.graphs.knowledge_trees.nodes import MarkdownNode, TraversalNode
from puku_core.graphs.knowledge_trees.types import MarkdownNodeUpdateRequest
from puku_core.graphs.knowledge_trees.writers.base import (
    BaseMarkdownKnowledgeWriter,
    BaseMarkdownSingleNodeKnowledgeWriter,
    BaseMarkdownNodeAmendmentSplitter,
)


class HierarchicalMarkdownKnowledgeWriter(BaseMarkdownKnowledgeWriter):
    node_writer: BaseMarkdownSingleNodeKnowledgeWriter
    amendment_splitter: BaseMarkdownNodeAmendmentSplitter

    def invoke(
        self,
        input: MarkdownNodeUpdateRequest,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> Any:
        traversal: list[TraversalNode] = input.node.descendants()
        amendment: dict[MarkdownNode, MarkdownDocument] = {input.node: input.amendment}
