from typing import Any
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig

from puku_core.graphs.knowledge_trees.types import (
    MarkdownNodeUpdateRequest,
    MarkdownNodeAmendmentPropagation,
)


class BaseKnowledgeWriter(RunnableSerializable):
    """A component for memorizing knowledge in a knowledge tree"""

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs) -> Any:
        """Memorize knowledge in a knowledge tree

        Args:
            input (Any): Node update request

        """

        raise NotImplementedError


class BaseMarkdownKnowledgeWriter(BaseKnowledgeWriter):
    """Knowledge writer for markdown nodes"""

    def invoke(
        self,
        input: MarkdownNodeUpdateRequest,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> Any:
        """Memorize knowledge in a markdown knowledge tree

        Args:
            input (MarkdownNodeUpdateRequest): Update request
        """

        raise NotImplementedError


class BaseMarkdownSingleNodeKnowledgeWriter(RunnableSerializable):
    """A component for memorizing knowledge in a single markdown knowledge tree node"""

    def invoke(
        self,
        input: MarkdownNodeUpdateRequest,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> Any:
        """Update information in a single node

        Args:
            input (MarkdownNodeUpdateRequest): Update request
        """

        raise NotImplementedError


class BaseMarkdownNodeAmendmentSplitter(RunnableSerializable):
    """A component for splitting node amendment"""

    def invoke(
        self,
        input: MarkdownNodeUpdateRequest,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> MarkdownNodeAmendmentPropagation:
        """Split node amendment so that you can write it to the children later

        Args:
            input (MarkdownNodeUpdateRequest): Amendment to be splitted
        """

        raise NotImplementedError
