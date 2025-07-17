from typing import Any
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig

from puku_core.graphs.knowledge_trees.types import (
    UpdateRequest,
    NodeAmendmentPropagation,
)


class BaseKnowledgeWriter(RunnableSerializable):
    """A component for memorizing knowledge in a knowledge tree"""

    def invoke(
        self, input: UpdateRequest, config: RunnableConfig | None = None, **kwargs
    ) -> Any:
        """Memorize knowledge in a knowledge tree

        Args:
            input (Any): Node update request

        """

        raise NotImplementedError


class BaseAmendmentSplitter(RunnableSerializable):
    """A component for splitting node amendment"""

    def invoke(
        self,
        input: UpdateRequest,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> NodeAmendmentPropagation:
        """Split node amendment so that you can write it to the children later

        Args:
            input (UpdateRequest): Amendment to be splitted
        """

        raise NotImplementedError
