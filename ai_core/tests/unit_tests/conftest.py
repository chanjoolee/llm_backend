from pathlib import Path

import pytest

from ai_core.conversation.base import Conversation, ConversationFactory


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-7fcZavknQda3SQ",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Bar Baz",
                },
                "finish_reason": "stop",
            }
        ],
    }


@pytest.fixture()
def con_str(tmp_path: Path) -> str:
    file_path = tmp_path / "db.sqlite3"
    con_str = f"sqlite:///{file_path}"
    return con_str


@pytest.fixture
def basic_conversation(con_str) -> Conversation:
    return ConversationFactory.create_basic_conversation(
        llm_api_provider="openai",
        llm_model="gpt-4o",
        llm_api_key="api_key",
        llm_api_url="api_url",
        temperature=0.5,
        max_tokens=100,
        conversation_id="test",
        history_connection_str=con_str,
        history_table_name="session_store",
    )


@pytest.fixture
def conversation(con_str) -> Conversation:
    return ConversationFactory.create_conversation(
        llm_api_provider="openai",
        llm_model="gpt-4o",
        llm_api_key="api_key",
        llm_api_url="api_url",
        temperature=0.5,
        max_tokens=100,
        conversation_id="test",
        history_connection_str=con_str,
        history_table_name="session_store",
    )
