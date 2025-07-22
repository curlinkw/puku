from typing import Any
from langchain_core.runnables.config import RunnableConfig

from puku_core.documents.markdown import MarkdownDocument
from puku_core.graphs.knowledge_trees.nodes.markdown import MarkdownNode
from puku_core.graphs.knowledge_trees.nodes.traversal import TraversalNode
from puku_core.graphs.knowledge_trees.types import (
    MarkdownUpdateRequest,
    MarkdownNodeAmendmentPropagation,
)
from puku_core.graphs.knowledge_trees.writers.base import (
    BaseKnowledgeWriter,
    BaseAmendmentSplitter,
)


class HierarchicalMarkdownKnowledgeWriter(BaseKnowledgeWriter):
    node_writer: BaseKnowledgeWriter
    amendment_splitter: BaseAmendmentSplitter

    def invoke(
        self,
        input: MarkdownUpdateRequest,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> Any:
        traversal: list[TraversalNode] = input.node.descendants()
        amendment: dict[MarkdownNode, MarkdownDocument] = {input.node: input.amendment}

        for tnode in traversal:
            node: MarkdownNode = tnode.node

            amendment_propagation: MarkdownNodeAmendmentPropagation = (
                self.amendment_splitter.invoke(
                    input=MarkdownUpdateRequest(node=node, amendment=amendment[node]),
                    config=config,
                    kwargs=kwargs,
                )
            )

            self.node_writer.invoke(
                input=MarkdownUpdateRequest(
                    node=node, amendment=amendment_propagation.node
                ),
                config=config,
                kwargs=kwargs,
            )

            for (
                edge,
                child_amendment,
            ) in amendment_propagation.children.items():
                child: MarkdownNode = edge.child
                amendment[child] = child_amendment
