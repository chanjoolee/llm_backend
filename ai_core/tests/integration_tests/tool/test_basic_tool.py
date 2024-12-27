import pytest
from langchain_core.messages import AIMessage, ToolMessage

from ai_core.conversation.base import Conversation
from ai_core.conversation.message.base import DaisyMessageRole


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_id, smart_bee_model", [("test1", "smart_bee_gpt_4o"),])
async def test_tool_call(smart_bee_conversation: Conversation, conversation_id, smart_bee_model):
    smart_bee_conversation.add_tool("add", "egnarts", './add.py')
    try:
        await smart_bee_conversation.create_agent(debug=False)
        await smart_bee_conversation.clear(conversation_id)
        messages = []
        async for m in smart_bee_conversation.invoke(conversation_id, "3과 4를 더한 결과는?"):
            print(m)
            messages.append(m)

        assert (messages[0].role == DaisyMessageRole.AI and messages[0].tool_call is not None
                and isinstance(messages[0].raw_message, AIMessage))
        assert (messages[1].role == DaisyMessageRole.AGENT and messages[1].tool_call is not None and messages[1].message == '7'
                and isinstance(messages[1].raw_message, ToolMessage))
        assert (messages[2].role == DaisyMessageRole.AI and messages[2].tool_call is None
                and isinstance(messages[2].raw_message, AIMessage))
    finally:
        await smart_bee_conversation.close_connection_pools()


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_id, smart_bee_model", [("test1", "smart_bee_gpt_4o"),])
async def test_send_message(smart_bee_conversation: Conversation, conversation_id, smart_bee_model):
    smart_bee_conversation.add_tool("send_message_to_daisy_channel", "egnarts", './send_message_to_daisy_channel.py')
    try:
        await smart_bee_conversation.create_agent(debug=False)
        await smart_bee_conversation.clear(conversation_id)
        messages = []
        async for m in smart_bee_conversation.invoke(conversation_id, "테스트 메시지"):
            print(m)
            messages.append(m)
    finally:
        await smart_bee_conversation.close_connection_pools()


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_id, ai_one_model", [("test1", "ai_one_claude"),])
async def test_tool_call_streaming(ai_one_conversation: Conversation, conversation_id, ai_one_model):
    ai_one_conversation.add_tool("add", "egnarts", './add.py')
    await ai_one_conversation.create_agent(debug=True)
    await ai_one_conversation.clear(conversation_id)
    async for m in ai_one_conversation.stream(conversation_id, "3과 4를 더한 결과는?"):
        print("Returned", m)

