from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.language_models import (
    FakeListChatModel,
    FakeListLLM,
    FakeStreamingListLLM,
)
from langchain_core.prompts import ChatPromptTemplate

from ai_core.conversation.base import Conversation


def test_basic_conversation(basic_conversation: Conversation, mock_completion: dict):
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with get_openai_callback() as cb, patch.object(basic_conversation._chat_model, "client", mock_client):
        daisy_message = basic_conversation.invoke("session1", "test1")
        assert daisy_message.message == "Bar Baz"

    assert completed


@pytest.mark.asyncio
async def test_basic_conversation_streaming(monkeypatch, conversation: Conversation, mock_completion: dict):
    response = "Response"

    def create_chat_model(self):
        self._chat_model = FakeListChatModel(responses=[
            response
        ])

    monkeypatch.setattr(Conversation, "_create_chat_model", create_chat_model)

    conversation._create_chat_model()
    await conversation.create_agent()

    # iterator =

    # loop through the iterator and assert that the response is the same as the response above by character one by one
    chunks = []
    async for chunk in conversation.stream("session1", "test1"):
        chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        assert chunk.message == response[i]


def test_title_with_no_history(basic_conversation: Conversation):
    title = basic_conversation.generate_title("no_history")
    assert title is None
