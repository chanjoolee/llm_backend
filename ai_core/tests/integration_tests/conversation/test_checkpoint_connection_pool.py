import pytest

from ai_core.conversation.base import ConversationFactory
from ai_core.llm_api_provider import LlmApiProvider


@pytest.mark.asyncio
async def test_checkpoint_connection_pool(smart_bee_api_url, smart_bee_api_key, smart_bee_gpt_4o,
                                          sync_conn_pool, async_conn_pool):
    try:
        conv = ConversationFactory.create_conversation(
            llm_api_provider=LlmApiProvider.SMART_BEE.value,
            llm_model=smart_bee_gpt_4o,
            llm_api_key=smart_bee_api_key,
            llm_api_url=smart_bee_api_url,
            temperature=0.2,
            max_tokens=100,
            sync_conn_pool=sync_conn_pool,
            async_conn_pool=async_conn_pool
        )

        conv.add_tool("add_two_numbers", "egnarts", '../tool/add.py')
        await conv.create_agent()

        conversation_id = "thread1"
        await conv.clear(conversation_id)

        messages = conv.invoke(conversation_id, "3 + 4?")
        async for m in messages:
            print(m)

        print(await conv.generate_title(conversation_id))
    finally:
        sync_conn_pool.close()
        async_conn_pool.close()
        await async_conn_pool.wait_closed()
