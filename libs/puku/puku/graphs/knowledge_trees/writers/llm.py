from langchain_core.language_models.chat_models import BaseChatModel

from langchain_core.runnables.config import RunnableConfig
from puku_core.graphs.knowledge_trees.types import (
    MarkdownNodeAmendmentPropagation,
    MarkdownNodeUpdateRequest,
)
from puku_core.graphs.knowledge_trees.writers.base import (
    BaseMarkdownSingleNodeKnowledgeWriter,
    BaseMarkdownNodeAmendmentSplitter,
)


class ChatModelMarkdownNodeAmendmentSplitter(BaseMarkdownNodeAmendmentSplitter):
    chat_model: BaseChatModel

    def invoke(
        self,
        input: MarkdownNodeUpdateRequest,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> MarkdownNodeAmendmentPropagation:
        return super().invoke(input, config, **kwargs)
