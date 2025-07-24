from typing import Annotated, Any
from pydantic import BeforeValidator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.exceptions import OutputParserException

from puku_core.graphs.knowledge_trees.nodes.markdown import MarkdownNode
from puku_core.graphs.knowledge_trees.types import (
    MarkdownNodeAmendmentPropagation,
    MarkdownUpdateRequest,
)
from puku_core.graphs.knowledge_trees.writers.base import (
    BaseKnowledgeWriter,
    BaseAmendmentSplitter,
)
from puku_core.documents.markdown import MarkdownDocument, render
from puku.output_parsers.markdown import MarkdownOutputParser
from puku.graphs.knowledge_trees.writers.schemas import AmendmentSplit
from puku.prompts.validators import BasePromptValidationTemplate, has_template


class ChatModelAmendmentSplitter(BaseAmendmentSplitter):
    prompt: Annotated[
        ChatPromptTemplate,
        BeforeValidator(
            has_template(
                template=BasePromptValidationTemplate(
                    input_variables={
                        "node",
                        "edges_description",
                        "amendment",
                    }
                )
            )
        ),
    ]
    chat_model: BaseChatModel

    @classmethod
    def edges_description(cls, node: MarkdownNode) -> str:
        description: str = ""

        for edge in node.children_without_metadata:
            description += f'[{edge.child.title}]: edge "{hash(edge)}"\n'
            description += f"{edge.child.description}\n\n"

        return description

    def invoke(
        self,
        input: MarkdownUpdateRequest,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> MarkdownNodeAmendmentPropagation:
        node: MarkdownNode = input.node
        amendment: MarkdownDocument = input.amendment

        if not node.children:
            return MarkdownNodeAmendmentPropagation(node=amendment, children={})

        structured_chat_model = self.prompt | self.chat_model.with_structured_output(
            schema=AmendmentSplit
        )

        try:
            split: AmendmentSplit = structured_chat_model.invoke(
                input={
                    "node": render(node.data),
                    "edges_description": self.edges_description(node=node),
                    "amendment": render(amendment),
                },
                config=config,
                kwargs=kwargs,
            )  # type: ignore
            return split.to_propagation(node=node)
        except OutputParserException:
            # some logic of retrying could be here
            return MarkdownNodeAmendmentPropagation(node=amendment, children={})


class ChatModelNodeKnowledgeWriter(BaseKnowledgeWriter):
    prompt: Annotated[
        ChatPromptTemplate,
        BeforeValidator(
            has_template(
                template=BasePromptValidationTemplate(
                    input_variables={
                        "node",
                        "amendment",
                    }
                )
            )
        ),
    ]
    chat_model: BaseChatModel

    def invoke(
        self,
        input: MarkdownUpdateRequest,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> Any:
        node: MarkdownNode = input.node
        amendment: MarkdownDocument = input.amendment
        node.data = (self.prompt | self.chat_model | MarkdownOutputParser()).invoke(
            {"node": render(node.data), "amendment": render(amendment)}
        )
