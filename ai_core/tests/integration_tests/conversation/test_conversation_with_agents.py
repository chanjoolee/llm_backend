import pytest

from ai_core.tool.base import load_tool


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_id, smart_bee_model", [("test1", "smart_bee_gpt_4o"),])
async def test_conversation_with_agents(smart_bee_conversation, smart_bee_chat_model, conversation_id, smart_bee_model):
    tool = load_tool("add_two_numbers", "egnarts", '../tool/add.py')
    smart_bee_conversation.add_agent("calculator", "A calculator which can add two numbers",
                                     smart_bee_chat_model, [tool])

    await smart_bee_conversation.create_agent()

    await smart_bee_conversation.clear(conversation_id)
    async for m in smart_bee_conversation.invoke(conversation_id, "3121 + 42323?"):
        print(f"RES: {m}")

    print(await smart_bee_conversation.generate_title(conversation_id))
