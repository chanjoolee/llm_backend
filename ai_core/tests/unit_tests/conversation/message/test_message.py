from langchain_core.prompts import ChatPromptTemplate

from ai_core.conversation.message.base import create_default_prompt_messages


def test_optional_history_var():
    messages = create_default_prompt_messages()
    prompt = ChatPromptTemplate.from_messages(messages)

    prompt.invoke({})

