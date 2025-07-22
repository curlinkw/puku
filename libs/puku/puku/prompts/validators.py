from pydantic import BaseModel
from langchain_core.prompts import BasePromptTemplate


class BasePromptValidationTemplate(BaseModel):
    input_variables: list[str]


class has_template(BaseModel):
    template: BasePromptValidationTemplate

    def __call__(self, template: BasePromptTemplate) -> BasePromptTemplate:
        return template
