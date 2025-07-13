from typing import Any, Union, Tuple, Dict
from langchain_core.runnables.config import run_in_executor
from langchain_core.load.serializable import Serializable

from puku_core.documents.markdown import MarkdownDocument
from puku_core.graphs.knowledge_trees.nodes import BaseNode, MarkdownNode
from puku_core.graphs.knowledge_trees.edges import BaseEdge


class BaseKnowledgeWriter(Serializable):
    """A component for memorizing knowledge in a knowledge tree"""

    def write(self, node: BaseNode, *args, **kwargs) -> Any:
        """Memorize knowledge in a knowledge tree

        Args:
            node (BaseNode): Root node in the knowledge tree
        """
        raise NotImplementedError

    async def awrite(self, node: BaseNode, *args, **kwargs) -> Any:
        """Asynchronously memorize knowledge in a knowledge tree

        Args:
            node (BaseNode): Root node in the knowledge tree
        """
        return await run_in_executor(None, self.write, node, *args, **kwargs)


class BaseMarkdownKnowledgeWriter(BaseKnowledgeWriter):
    """Knowledge writer for markdown nodes"""

    def write(self, node: MarkdownNode, amendment: MarkdownDocument) -> Any:
        """Memorize knowledge in a markdown knowledge tree

        Args:
            node (MarkdownNode): Root node in the knowledge tree
            amendment (MarkdownDocument): Information to be stored

        Returns:
            Any: Feedback
        """

        raise NotImplementedError

    def update_node(self, node: MarkdownNode, amendment: MarkdownDocument) -> Any:
        """Update information in a single node

        Args:
            node (MarkdownNode): Node to be updated
            amendment (MarkdownDocument): New information

        Returns:
            Any: Feedback
        """

        raise NotImplementedError

    async def aupdate_node(
        self, node: MarkdownNode, amendment: MarkdownDocument
    ) -> Any:
        """Asynchronously Update information in a single node

        Args:
            node (MarkdownNode): Node to be updated
            amendment (MarkdownDocument): New information

        Returns:
            Any: Feedback
        """

        return run_in_executor(None, self.update_node, node, amendment)

    def split_amendment(
        self,
        node: MarkdownNode,
        amendment: MarkdownDocument,
        only_children: bool = False,
    ) -> Union[
        Tuple[MarkdownDocument, Dict[BaseEdge, MarkdownDocument]],
        Dict[BaseEdge, MarkdownDocument],
    ]:
        """Split amendant so that you can write it to the children later.

        Args:
            node (MarkdownNode): Knowledge tree node
            amendment (MarkdownDocument): Amendment to be splitted
            only_children (bool, optional): Should the information be shared only for children, \
                or should the node itself be taken into account. Defaults to False.

        Returns:
            Union: Amendment for node and children (in case only_children = False) \
                or only for children (in case only_children = True)
        """

        raise NotImplementedError

    async def asplit_amendment(
        self,
        node: MarkdownNode,
        amendment: MarkdownDocument,
        only_children: bool = False,
    ) -> Union[
        Tuple[MarkdownDocument, Dict[BaseEdge, MarkdownDocument]],
        Dict[BaseEdge, MarkdownDocument],
    ]:
        """Asynchronously split amendant so that you can write it to the children later.

        Args:
            node (MarkdownNode): Knowledge tree node
            amendment (MarkdownDocument): Amendment to be splitted
            only_children (bool, optional): Should the information be shared only for children, \
                or should the node itself be taken into account. Defaults to False.

        Returns:
            Union: Amendment for node and children (in case only_children = False) \
                or only for children (in case only_children = True)
        """
        return await run_in_executor(
            None, self.split_amendment, node, amendment, only_children
        )
