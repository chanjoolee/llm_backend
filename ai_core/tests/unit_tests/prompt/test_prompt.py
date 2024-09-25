import pytest
from langchain_core.prompts import ChatPromptTemplate

from ai_core.conversation.base import Conversation
from ai_core.prompt.base import PromptComponent


def test_missing_variable():
    with pytest.raises(ValueError) as exc_info:
        PromptComponent(name="prompt1", description="test", messages=[("system", "{name}")],
                        input_values={"name1": "assistant"})


def test_unknown_variable():
    with pytest.raises(ValueError) as exc_info:
        PromptComponent(name="prompt1", description="test", messages=[("system", "{name}")],
                        input_values={"name": "assistant", "name1": "assistant"})


def test_conversation_with_prompt(conversation: Conversation):
    prompt = [
        ("human", "{app_name}라는 플링크 애플리케이션 이름의 의미가 뭐야?"),
        ("ai", """
        플링크 애플리케이션 이름은 세 가지로 이루어져 있습니다. 첫 문자열부터 - 전까지의 문자열은 그룹명을 나타내며, 
        - 다음부터 _ 까지 문자열은 서비스명이고 _ 다음의 문자열은 타입을 나타냅니다. 
        따라서 {app_name}의 경우 그룹명은 {group}, 서비스명은 {service}, 타입은 {type}입니다.
        """),
    ]

    input1 = {
        "app_name": "cdr-output_pofcs",
        "group": "cdr",
        "service": "output",
        "type": "pofcs"
    }

    input2 = {
        "app_name": "lcap-dedup_std_xdr_l",
        "group": "lcap",
        "service": "dedup",
        "type": "std_xdr_l"
    }

    prompt_component1 = PromptComponent(name="prompt1", description="desc", messages=prompt, input_values=input1)
    prompt_component2 = PromptComponent(name="prompt1", description="desc", messages=prompt, input_values=input2)

    conversation.add_prompt(prompt_component1)
    conversation.add_prompt(prompt_component2)

    merged_prompt = ChatPromptTemplate.from_messages(conversation._merge_prompt_messages())

    chat_prompt = ChatPromptTemplate.from_messages(prompt)
    chat_messages = chat_prompt.invoke(input1).to_messages()
    chat_messages.extend(chat_prompt.invoke(input2).to_messages())

    assert merged_prompt.messages[:len(chat_messages)] == chat_messages
