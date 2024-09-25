import pytest

from ai_core.conversation.base import Conversation


def test_finding_tool(conversation: Conversation):
    conversation.add_tool("add", "egnarts", "./add.py")

    assert len(conversation.tools) == 1


def test_tool_name(conversation: Conversation):
    conversation.add_tool("add2", "egnarts", "./add.py")

    assert len(conversation.tools) == 1
    assert conversation.tools[0].name == "add2_tl_egnarts"


def test_invalid_multiple_tools(conversation: Conversation):
    with pytest.raises(ValueError) as exc_info:
        conversation.add_tool("add", "egnarts", "./multiple_tools.py")

