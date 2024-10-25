from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, AIMessageChunk, SystemMessage, \
    MessageLikeRepresentation
from langchain_core.prompts import SystemMessagePromptTemplate, MessagesPlaceholder

from ai_core.conversation.message.tokens import TokensUsage


# Langgraph에서 State에 메시지 목록을 담는 변수 명
HISTORY_VARIABLE_NAME = "messages"


class DaisyMessageRole(Enum):
    SYSTEM = "system"
    AI = "ai"
    HUMAN = "human"
    AGENT = "agent"


@dataclass
class ToolCall:
    """
    도구를 호출하기 전과 후에 관한 정보를 담고 있습니다.
    """

    run_id: str
    """
    도구를 호출할 때 사용된 run_id입니다.
    """

    name: str
    """
    도구의 이름입니다.
    """

    # tool_info: Optional[Dict[str, Any]] = None
    # """
    # 도구의 이름, 설명, 인자 정보를 담고 있습니다.
    # """

    inputs: Optional[Dict[str, Any]] = None
    """
    도구를 호출할 때 인자로 들어갈 값들을 담고 있습니다.
    """

    output: Optional[Any] = None
    """
    도구를 호출한 후 도구의 결과 값을 담고 있습니다.
    """

    error: Optional[BaseException] = None
    """
    도구를 호출한 후 도구에서 발생한 에러를 담고 있습니다.
    """


@dataclass
class DaisyMessage:
    id: str
    message: str
    role: DaisyMessageRole
    raw_message: Optional[BaseMessage] = None
    tokens_usage: Optional[TokensUsage] = None
    tool_call: Optional[ToolCall] = None

    def pretty_print_raw_message(self):
        return self.raw_message.pretty_repr(html=False)

    @staticmethod
    def convert_tool_call_to_daisy_message(tool_call: ToolCall) -> DaisyMessage:
        return DaisyMessage(
            id=str(tool_call.run_id) + ("-0" if tool_call.inputs else "-1"),
            message='',
            role=DaisyMessageRole.AGENT,
            tool_call=tool_call
        )


@dataclass
class DaisyMessageChunk(DaisyMessage):
    @staticmethod
    def convert_chunk_to_daisy_chunk(chunk: BaseMessage) -> DaisyMessageChunk:
        return DaisyMessageChunk(message=chunk.content, role=DaisyMessageRole.AI)


def convert_to_daisy_message(message: BaseMessage) -> DaisyMessage:
    tokens_usage = None
    tool_call = None
    if isinstance(message, AIMessage):
        if message.usage_metadata:
            tokens_usage = TokensUsage(
                input_tokens=message.usage_metadata.get("input_tokens", 0),
                output_tokens=message.usage_metadata.get("output_tokens", 0),
                total_tokens=message.usage_metadata.get("total_tokens", 0)
            )

        if message.tool_calls:
            tool_call = ToolCall(run_id=message.tool_calls[0]['id'], name=message.tool_calls[0]['name'],
                                 inputs=message.tool_calls[0]['args'])

        role = DaisyMessageRole.AI

        if isinstance(message, AIMessageChunk):
            content = message.content
            if not isinstance(message.content, str):
                if 'text' in message.content[0]:
                    content = message.content[0]['text']
                else:
                    content = ""

            return DaisyMessageChunk(id=message.id, message=content, role=role, raw_message=message,
                                     tokens_usage=tokens_usage,)
    elif isinstance(message, ToolMessage):
        role = DaisyMessageRole.AGENT
        tool_call = ToolCall(run_id=message.tool_call_id, name=message.name, output=message.content)
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")

    return DaisyMessage(id=message.id, raw_message=message, message=message.content, role=role,
                        tokens_usage=tokens_usage, tool_call=tool_call)


def create_default_prompt_messages(system_message="You are a helpful assistant.") -> List[MessageLikeRepresentation]:
    return [
        SystemMessage(system_message)
    ]

