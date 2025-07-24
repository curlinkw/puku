from pydantic import BaseModel
from langchain_core.prompts import BasePromptTemplate


class BasePromptValidationTemplate(BaseModel):
    input_variables: set[str]


class has_template(BaseModel):
    template: BasePromptValidationTemplate

    def __call__(self, prompt: BasePromptTemplate) -> BasePromptTemplate:
        input_variables = set(prompt.input_variables)
        if input_variables != self.template.input_variables:
            raise ValueError(
                f"Input variables should be: {self.template.input_variables}, but has {input_variables}"
            )
        return prompt
