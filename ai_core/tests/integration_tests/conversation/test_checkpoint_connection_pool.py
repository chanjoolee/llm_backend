import pytest

from ai_core.checkpoint.mysql_saver import MySQLSaver

from ai_core.conversation.base import ConversationFactory
from ai_core.llm_api_provider import LlmApiProvider


@pytest.mark.asyncio
async def test_checkpoint_connection_pool(history_conn_str, sync_conn_pool, async_conn_pool):
    try:
        MySQLSaver.create_tables(sync_conn_pool)

        conv = ConversationFactory.create_conversation(
            llm_api_provider=LlmApiProvider.SMART_BEE.value,
            llm_model="gpt-4o",
            llm_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441",
            llm_api_url="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
            temperature=0.2,
            max_tokens=100,
            sync_conn_pool=sync_conn_pool,
            async_conn_pool=async_conn_pool
        )

        conv.add_tool("add_two_numbers", "egnarts", '../tool/add.py')
        await conv.create_agent()

        conversation_id = "thread1"
        conv.clear(conversation_id)

        messages = conv.invoke(conversation_id, "3 + 4?")
        for m in messages:
            print(m)

        print(conv.generate_title(conversation_id))
    finally:
        sync_conn_pool.close()
        async_conn_pool.close()
        await async_conn_pool.wait_closed()
