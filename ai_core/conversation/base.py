import copy
import os
from importlib.machinery import SourceFileLoader
from typing import Any, Optional, List, Sequence, Literal, Callable, Union

import aiomysql
from aiomysql import Pool
from dbutils.pooled_db import PooledDB
from langchain_core.messages import HumanMessage
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, field_validator, Field

from ai_core.agent.base import create_agent, Agent, create_supervisor_agent
from ai_core.checkpoint.mysql_saver import MySQLSaver
from ai_core.conversation.message.base import DaisyMessage, \
    create_default_prompt_messages, ToolCall, convert_to_daisy_message
from ai_core.data_source.base import DataSource, create_data_source_tool
from ai_core.llm.base import create_chat_model
from ai_core.llm_api_provider import LlmApiProvider
from ai_core.prompt.base import PromptComponent
from ai_core.tool.base import load_tool
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver

SUPPORTED_LLM_PROVIDERS = {LlmApiProvider.AI_ONE.value, LlmApiProvider.SMART_BEE.value, LlmApiProvider.AZURE.value}


class Conversation(BaseModel):
    """
    모든 대화 클래스의 부모 클래스입니다.
    """
    llm_api_url: str
    llm_api_provider: str
    llm_api_key: str
    llm_model: str
    temperature: float
    max_tokens: int

    _chat_model: BaseChatModel = None
    prompt_components: List[PromptComponent] = Field(default_factory=list)

    _runnable: Runnable = None

    sync_conn_pool: Optional[PooledDB] = None
    async_conn_pool: Optional[Pool] = None

    _checkpointer: BaseCheckpointSaver = None

    tools: List[Any] = Field(default_factory=list)

    agents: List[Agent] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context):
        self._create_chat_model()
        self._create_saver()

    @field_validator("llm_api_provider")
    def llm_provider_must_be_supported(cls, v):
        if v not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {v}")
        return v

    async def create_agent(self, debug=False) -> bool:
        """
        대화를 시작하기 전에 대화 에이전트를 먼저 생성해야 합니다. 만약 추가된 프롬프트나 도구, 데이터 소스 등이 존재한다면 그것들이 포함된 에이전트를 생성합니다.

        Returns:
            bool: 에이전트 생성 성공 여부, 이미 에이전트가 생성되어 있으면 False를 반환합니다.
        """

        if self._runnable is not None:
            return False

        if self.prompt_components:
            messages = self._merge_prompt_messages()
        else:
            messages = create_default_prompt_messages()

        # prompt = ChatPromptTemplate.from_messages(messages)

        # await self._create_connection_pool()

        if self.agents:
            if self.tools:
                raise ValueError("Cannot have both agents and tools")

            # TODO 슈퍼바이저 에이전트의 system message도 데이지에서 설정 가능하도록 할 것인가?
            self._runnable = create_supervisor_agent(self._chat_model, self.agents, self._checkpointer, messages)
        else:
            self._runnable = create_agent(self._chat_model, self._checkpointer, self.tools, before_messages=messages,
                                          debug=debug)

        return True

    def _merge_prompt_messages(self) -> List[MessageLikeRepresentation]:
        revised_messages = []
        for prompt_component in self.prompt_components:
            revised_messages.extend(copy.deepcopy(prompt_component.invoke()))

        return revised_messages

    def add_prompt(self, prompt_component: PromptComponent):
        self.prompt_components.append(prompt_component)

    def add_tool(self, name, username, tool_path):
        """
        도구를 추가합니다.

        Args:
            name (str): 도구 이름
            username (str): 사용자 이름
            tool_path (Any): 도구 객체

        """
        self.tools.append(load_tool(name, username, tool_path))

    def add_datasource(self, name: str, username: str, datasource: DataSource):
        tool = create_data_source_tool(name, username, datasource)
        self.tools.append(tool)

    def add_agent(self, name: str, description: str, chat_model: BaseChatModel, tools: Sequence[Union[BaseTool, Callable]],
                  prompt_messages: Optional[List[MessageLikeRepresentation]] = None, debug=False):
        agent_node = create_agent(chat_model, self._checkpointer, tools, before_messages=prompt_messages, debug=debug)
        self.agents.append(Agent(name=name, description=description, agent_node=agent_node))

    def is_agent_created(self) -> bool:
        """
        에이전트가 생성되어 있는지 확인합니다.

        Returns:
            bool: 체인이 생성되어 있으면 True, 아니면 False
        """
        return self._runnable is not None

    def _create_chat_model(self):
        self._chat_model = create_chat_model(
            llm_api_provider=self.llm_api_provider,
            llm_api_key=self.llm_api_key,
            llm_api_url=self.llm_api_url,
            llm_model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def _create_saver(self):
        if self.async_conn_pool:
            self._checkpointer = AIOMySQLSaver(self.async_conn_pool)
        else:
            self._checkpointer = MemorySaver()

    def _create_chat_config(self, conversation_id):
        return {"configurable": {"thread_id": conversation_id}}

    async def invoke(self, conversation_id, message):
        """
        LLM 모델에 사용자 메시지를 보내고 응답을 반환합니다. 응답은 도구의 사용 여부에 따라 여러 개가 될 수 있습니다.
        """
        if self._runnable is None:
            raise ValueError("Agent is not initialized")

        config = self._create_chat_config(conversation_id)
        inputs = {"messages": [HumanMessage(content=message)]}
        async for m in self._runnable.astream(inputs, config=config, stream_mode="values"):
            last_message = m['messages'][-1]
            if isinstance(last_message, HumanMessage):
                continue

            yield convert_to_daisy_message(m['messages'][-1])

    async def stream(self, conversation_id, message, debug=False):
        """
        특정 대화에 사용자 메시지를 보내고 응답을 스트리밍합니다.
        현재 도구를 포함한 대화에서 스트리밍이 지원되는 케이스는 AI ONE + GPT 뿐입니다. 따라서 그 외에는 invoke를 호출해야 합니다.

        Args:
            conversation_id: 대화 ID
            message: 사용자 메시지

        Returns:
            AsyncIterable: 응답 메시지를 스트리밍하는 Iterable
        """
        if self._runnable is None:
            raise ValueError("Agent is not initialized")

        # if self._async_conn.closed:
        #     await self._async_conn._connect()

        config = self._create_chat_config(conversation_id)

        # streaming_iter = self._runnable.stream({"messages": [HumanMessage(content=message)]}, config=config)
        # return DaisyChunkSyncIterator(streaming_iter)
        async for e in self._runnable.astream_events({"messages": [HumanMessage(content=message)]}, config=config, version='v2'):
            kind = e['event']
            handled_event = False
            if kind == 'on_chat_model_stream':
                chunk = e['data']['chunk']
                if chunk.content:
                    handled_event = True
                    # print('on_chat_model_stream')
                    # print(chunk)
                    yield convert_to_daisy_message(chunk)
            elif kind == 'on_tool_start':
                data = e['data']
                tool_call = ToolCall(run_id=e['run_id'], inputs=data['input'], name=e['name'])
                tool_call_message = f"도구 '{tool_call.name}'을 {tool_call.inputs} 인자로 실행합니다."
                yield DaisyMessage.convert_tool_call_to_daisy_message(tool_call, tool_call_message)
            elif kind == 'on_tool_end':
                data = e['data']
                tool_call = ToolCall(run_id=e['run_id'], output=data['output'], name=e['name'])
                yield DaisyMessage.convert_tool_call_to_daisy_message(tool_call, data['output'].content)

            if not handled_event and debug:
                print(f"Unhandled event: {e}")

    def invoke_tool(self, tool_name, args):
        # find tool by name
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if tool is None:
            raise ValueError(f"Tool not found: {tool_name}")

        return tool.invoke(args)

    async def generate_title(self, conversation_id) -> Optional[str]:
        """
        대화 히스토리를 이용하여 제목을 생성합니다.

        Args:
            conversation_id: 대화 ID

        Returns:
            Optional[str]: 생성된 제목, 대화 히스토리가 없으면 None을 반환합니다.
        """
        config = self._create_chat_config(conversation_id)
        checkpoint = await self._checkpointer.aget(config)
        if not checkpoint:
            return None

        history = checkpoint['channel_values']['messages']
        if not history:
            return None

        messages = history.copy()
        messages.append(("human", "{question}"))
        prompt = ChatPromptTemplate.from_messages(
            messages
        )

        title_generator = prompt | self._chat_model
        message_with_title = title_generator.invoke({"question": "앞의 모든 메시지를 보고 제목을 20글자 이내로 명확하게 생성해주세요."})
        return message_with_title.content

    async def copy_conversation(self, conversation_id, new_conversation_id):
        """
        대화를 신규 대화로 복사합니다.

        Args:
            conversation_id: 복사할 대화 ID
            new_conversation_id: 새로운 대화 ID
        """
        if isinstance(self._checkpointer, AIOMySQLSaver):
            await self._checkpointer.clone_thread(conversation_id, new_conversation_id)

    async def clear(self, conversation_id):
        """
        대화 히스토리를 삭제합니다.

        Args:
            conversation_id: 대화 ID
        """
        # checkpoint = empty_checkpoint()
        # self._checkpointer.put(config=config, checkpoint=checkpoint, metadata={})
        if isinstance(self._checkpointer, AIOMySQLSaver):
            await self._checkpointer.delete_thread(conversation_id)


    async def close_connection_pools(self):
        """
        만약 커넥션 풀을 외부에서 제공하지 않았다면 대화 자체에서 생성한 커넥션 풀을 반드시 닫아야 합니다.
        """
        if self.sync_conn_pool:
            self.sync_conn_pool.close()

        if self.async_conn_pool:
            self.async_conn_pool.close()
            await self.async_conn_pool.wait_closed()


class ConversationFactory:
    """
    대화 객체를 생성하는 팩토리 클래스입니다.
    """

    @staticmethod
    def create_conversation(llm_api_url, llm_api_provider, llm_api_key, llm_model, temperature, max_tokens,
                            sync_conn_pool: Optional[PooledDB] = None, async_conn_pool: Optional[Pool] = None, **kwargs) -> (
            Conversation):
        """
        대화 객체를 만듭니다.

        Args:
            llm_api_url (str): LLM API URL
            llm_api_provider (str): LLM API 타입
            llm_api_key (str): LLm API 키
            llm_model (str): LLM 모델
            temperature (float): 온도
            max_tokens (int): 최대 토큰 수
            sync_conn_pool (PooledDB): 동기 커넥션 풀
            async_conn_pool (Pool): 비동기 커넥션 풀
        """
        conversation = Conversation(llm_api_url=llm_api_url, llm_api_provider=llm_api_provider, llm_api_key=llm_api_key,
                                    llm_model=llm_model, temperature=temperature, max_tokens=max_tokens,
                                    sync_conn_pool=sync_conn_pool, async_conn_pool=async_conn_pool, **kwargs)

        return conversation

    @staticmethod
    async def create_basic_conversation(llm_api_url, llm_api_provider, llm_api_key, llm_model, temperature, max_tokens,
                                        sync_conn_pool: Optional[PooledDB] = None, async_conn_pool: Optional[Pool] = None, **kwargs) -> (
            Conversation):
        """
        기본 대화 객체를 만들고 체인을 생성합니다. 기본 대화 객체는 프롬프트, 도구, 데이터소스 등의 컴포넌트를 이용하지 않습니다.

        Args:
            llm_api_url (str): LLM API URL
            llm_api_provider (str): LLM API 타입
            llm_api_key (str): LLM API 키
            llm_model (str): LLM 모델
            temperature (float): 온도
            max_tokens (int): 최대 토큰 수
            sync_conn_pool (PooledDB): 동기 커넥션 풀
            async_conn_pool (Pool): 비동기 커넥션 풀

        Returns:
            Conversation: 생성된 대화 객체
        """
        conversation = Conversation(llm_api_url=llm_api_url, llm_api_provider=llm_api_provider, llm_api_key=llm_api_key,
                                    llm_model=llm_model, temperature=temperature, max_tokens=max_tokens,
                                    sync_conn_pool=sync_conn_pool, async_conn_pool=async_conn_pool, **kwargs)

        await conversation.create_agent()

        return conversation

    @staticmethod
    def create_sync_connection_pool(history_connection_str: str):
        from urllib.parse import urlparse
        url = urlparse(history_connection_str)

        return MySQLSaver.create_sync_connection_pool(
            host=url.hostname,
            user=url.username,
            password=url.password,
            db=url.path[1:],
            port=url.port,
            autocommit=True
        )

    @staticmethod
    async def create_async_connection_pool(history_connection_str: str):
        from urllib.parse import urlparse
        url = urlparse(history_connection_str)

        return await MySQLSaver.create_async_connection_pool(
            host=url.hostname,
            user=url.username,
            password=url.password,
            db=url.path[1:],
            port=url.port,
            autocommit=True
        )
