import pytest
import pytest_asyncio
from langchain_core.language_models import BaseChatModel

from ai_core.conversation.base import Conversation, ConversationFactory
from ai_core.llm.base import create_chat_model
from ai_core.llm_api_provider import LlmApiProvider


@pytest.fixture()
def history_conn_str() -> str:
    return 'mysql+pymysql://root:wkdwjdgh7!@localhost:3306/daisy'


@pytest.fixture()
def sync_conn_pool(history_conn_str):
    return ConversationFactory.create_sync_connection_pool(history_conn_str)


@pytest_asyncio.fixture()
async def async_conn_pool(history_conn_str):
    return await ConversationFactory.create_async_connection_pool(history_conn_str)


@pytest.fixture
def smart_bee_api_key() -> str:
    return "ba3954fe-9cbb-4599-966b-20b04b5d3441"


@pytest.fixture
def smart_bee_api_url() -> str:
    return "https://aihub-api.sktelecom.com/aihub/v1/sandbox"


@pytest.fixture
def smart_bee_gpt_4o() -> str:
    return "gpt-4o"


@pytest.fixture
def ai_one_api_key() -> str:
    return "sk-gapk-F14i5UwldXoRfXe6AGxpUSD9G-B0JOUR"


@pytest.fixture
def ai_one_api_url() -> str:
    return "https://api.platform.a15t.com/v1"


@pytest.fixture
def ai_one_gpt_4o() -> str:
    return "openai/gpt-4o-2024-05-13"


@pytest.fixture
def ai_one_claude() -> str:
    return "anthropic/claude-3-5-sonnet-20240620"


@pytest.fixture
def smart_bee_chat_model(smart_bee_api_key, smart_bee_api_url, smart_bee_model, request) -> BaseChatModel:
    return create_chat_model(
        llm_api_provider=LlmApiProvider.SMART_BEE.value,
        llm_model=request.getfixturevalue(smart_bee_model),
        llm_api_key=smart_bee_api_key,
        llm_api_url=smart_bee_api_url,
        temperature=0.2,
        max_tokens=100,
    )


@pytest_asyncio.fixture
async def smart_bee_conversation(smart_bee_api_key, smart_bee_api_url, smart_bee_model,
                                 sync_conn_pool, async_conn_pool, request) -> Conversation:
    return ConversationFactory.create_conversation(
        llm_api_provider=LlmApiProvider.SMART_BEE.value,
        llm_model=request.getfixturevalue(smart_bee_model),
        llm_api_key=smart_bee_api_key,
        llm_api_url=smart_bee_api_url,
        temperature=0.2,
        max_tokens=100,
        sync_conn_pool=sync_conn_pool,
        async_conn_pool=async_conn_pool,
    )


@pytest_asyncio.fixture
async def ai_one_conversation(ai_one_api_key, ai_one_api_url, ai_one_model,
                              sync_conn_pool, async_conn_pool, request) -> Conversation:
    return ConversationFactory.create_conversation(
        llm_api_provider=LlmApiProvider.AI_ONE.value,
        llm_model=request.getfixturevalue(ai_one_model),
        llm_api_key=ai_one_api_key,
        llm_api_url=ai_one_api_url,
        temperature=0.2,
        max_tokens=1024,
        sync_conn_pool=sync_conn_pool,
        async_conn_pool=async_conn_pool,
    )
