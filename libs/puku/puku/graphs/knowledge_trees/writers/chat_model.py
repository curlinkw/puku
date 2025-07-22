from typing import Annotated
from pydantic import BeforeValidator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig

from puku_core.graphs.knowledge_trees.nodes.markdown import MarkdownNode
from puku_core.graphs.knowledge_trees.types import (
    MarkdownNodeAmendmentPropagation,
    MarkdownUpdateRequest,
)
from puku_core.graphs.knowledge_trees.writers.base import (
    BaseKnowledgeWriter,
    BaseAmendmentSplitter,
)
from puku.prompts.validators import BasePromptValidationTemplate, has_template


class ChatModelAmendmentSplitter(BaseAmendmentSplitter):
    prompt: Annotated[
        ChatPromptTemplate,
        BeforeValidator(
            has_template(template=BasePromptValidationTemplate(input_variables=[""]))
        ),
    ]
    chat_model: BaseChatModel

    @classmethod
    def edges_description(cls, node: MarkdownNode) -> str:
        description: str = ""

        for edge_with_metadta in node.children:
            child = edge_with_metadta.edge.child
            description += f'[{child.title}]: edge "{hash(edge_with_metadta)}"\n'
            description += f"{child.description}\n\n"

        return description

    def invoke(
        self,
        input: MarkdownUpdateRequest,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> MarkdownNodeAmendmentPropagation:
        # use LinkRefDef to manage children
        return super().invoke(input, config, **kwargs)
