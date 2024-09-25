from typing import Any, Optional, Dict, List
from uuid import UUID

from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from ai_core.conversation.message.base import DaisyMessage
from ai_core.conversation.message.base import ToolCall


class DaisyCallbackHandler(BaseCallbackHandler):
    """
    에이전트가 작업을 수행하는 동안 발생하는 이벤트를 처리하는 콜백 핸들러입니다.
    """

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(messages)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(prompts)

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(action)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        tool_call = ToolCall(run_id=run_id, inputs=inputs, tool_info=serialized)
        self.on_daisy_tool_start(DaisyMessage.convert_tool_call_to_daisy_message(tool_call))

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        tool_call = ToolCall(run_id=run_id, output=output)
        self.on_daisy_tool_end(DaisyMessage.convert_tool_call_to_daisy_message(tool_call))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        tool_call = ToolCall(run_id=run_id, error=error)
        self.on_daisy_tool_error(DaisyMessage.convert_tool_call_to_daisy_message(tool_call))

    def on_daisy_tool_start(self, daisy_message: DaisyMessage):
        """
        도구가 호출될 때 실행되는 콜백 함수입니다.

        Args:
            daisy_message: 도구 호출 정보와 인자 값을 담고 있는 DaisyMessage 객체
        """
        print(f"Tool started with message: {daisy_message}")

    def on_daisy_tool_end(self, daisy_message: DaisyMessage):
        """
        도구가 호출된 후 실행되는 콜백 함수입니다.

        Args:
            daisy_message: 도구 호출 결과 값을 담고 있는 DaisyMessage 객체
        """
        print(f"Tool ended with message: {daisy_message}")

    def on_daisy_tool_error(self, daisy_message: DaisyMessage):
        """
        도구가 에러를 발생시켰을 때 실행되는 콜백 함수입니다.

        Args:
            daisy_message: 도구 Exception 객체를 담고 있는 DaisyMessage 객체
        """
        print(f"Tool errored with message: {daisy_message}")
