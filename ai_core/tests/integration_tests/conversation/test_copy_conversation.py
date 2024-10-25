import pytest

from ai_core.conversation.base import ConversationFactory
from ai_core.llm_api_provider import LlmApiProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_id, smart_bee_model", [("copy_test_thread", "smart_bee_gpt_4o"),])
async def test_conversation_with_agents(smart_bee_conversation, conversation_id, smart_bee_model,
                                        smart_bee_api_url, smart_bee_api_key, smart_bee_gpt_4o, sync_conn_pool, async_conn_pool):
    smart_bee_conversation.add_tool("add", "egnarts", '../tool/add.py')
    try:
        new_conversation_id = "copy_test_thread_1"

        await smart_bee_conversation.clear(conversation_id)
        await smart_bee_conversation.clear(new_conversation_id)

        await smart_bee_conversation.create_agent(debug=False)
        async for m in smart_bee_conversation.invoke(conversation_id, "3과 4를 더한 결과는?"):
            print(m)

        await smart_bee_conversation.copy_conversation(conversation_id, new_conversation_id)

        new_conv = ConversationFactory.create_conversation(
            llm_api_provider=LlmApiProvider.SMART_BEE.value,
            llm_model=smart_bee_gpt_4o,
            llm_api_key=smart_bee_api_key,
            llm_api_url=smart_bee_api_url,
            temperature=0.2,
            max_tokens=100,
            sync_conn_pool=sync_conn_pool,
            async_conn_pool=async_conn_pool
        )

        new_conv.add_tool("add", "egnarts", '../tool/add.py')
        await new_conv.create_agent(debug=False)

        async for m in new_conv.invoke(new_conversation_id, "3과 4를 더한 결과는?"):
            print(m)
    finally:
        await smart_bee_conversation.close_connection_pools()
