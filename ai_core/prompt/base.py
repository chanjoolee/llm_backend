from typing import List, Tuple, Dict, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import BaseMessagePromptTemplate
from pydantic import root_validator, model_validator

from ai_core.base import Component


class PromptComponent(Component):
    """
    system, ai, user 들로 이루어진 메시지 리스트를 가지고 있는 컴포넌트입니다.
    프롬프트가 가진 변수에 대해서 변수명과 값을 가진 딕셔너리를 입력으로 받습니다.
    """

    messages: List[Union[Tuple[str, str], BaseMessagePromptTemplate]]
    input_values: Dict[str, str]

    _prompt_template: ChatPromptTemplate = None

    def model_post_init(self, __context):
        self._prompt_template = ChatPromptTemplate.from_messages(self.messages)

    @model_validator(mode="after")
    def _validate_input_variables(self):
        if len(self.input_values) != len(self._prompt_template.input_variables):
            raise ValueError("input_variables length must match with the variables in messages.")

        if self.input_values.keys() != set(self._prompt_template.input_variables):
            raise ValueError("input_variables must match with the variables in messages.")

    def invoke(self):
        return self._prompt_template.format_prompt(**self.input_values).to_messages()
