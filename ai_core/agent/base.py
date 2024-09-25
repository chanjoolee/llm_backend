from typing import Optional, Literal, Sequence, Union, Callable, NamedTuple, TypedDict, Annotated, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, MessageLikeRepresentation, AIMessage
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph, add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import Field, BaseModel


def _should_continue(state: MessagesState) -> Literal["tools", END]:
    messages_in_state = state['messages']
    last_message = messages_in_state[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


class PromptConfig(NamedTuple):
    prompt: ChatPromptTemplate
    history_variable_name: Optional[str] = None

    def invoke(self, history: list):
        return self.prompt.invoke({self.history_variable_name: history})


def create_agent(chat_model: BaseChatModel, checkpointer: BaseCheckpointSaver,
                 tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
                 before_messages: Optional[List[MessageLikeRepresentation]] = None,
                 debug=False) -> CompiledGraph:
    if tools:
        messages_modifier = None
        if before_messages:
            def messages_modifier(messages: list):
                return before_messages + messages

        return create_react_agent(chat_model, tools, messages_modifier=messages_modifier, checkpointer=checkpointer,
                                  debug=debug)

    return create_agent_wo_tools(chat_model, checkpointer, before_messages)


def create_agent_wo_tools(chat_model: BaseChatModel, checkpointer: BaseCheckpointSaver,
                          before_messages: Optional[List[MessageLikeRepresentation]] = None) -> CompiledGraph:
    runnable = RunnableLambda(lambda state: before_messages + state["messages"] if before_messages else state["messages"]) | chat_model

    def call_model(state: MessagesState):
        # messages_in_state = state['messages']
        # response = runnable.invoke({prompt_config.history_variable_name: messages_in_state})
        response = runnable.invoke(state)
        # We return a list, because this will get added to the existing list
        return {"messages": response}

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)

    workflow.set_entry_point("agent")
    workflow.set_finish_point("agent")

    return workflow.compile(checkpointer=checkpointer)


class Agent(BaseModel):
    """
    슈퍼바이저 에이전트를 제외한 모든 에이전트는 이 클래스의 인스턴스여야 합니다.
    대화에서 에이전트를 추가하지 않고 도구 사용되는 에이전트도 포함됩니다.

    슈퍼바이저 에이전트는 이 에이전트의 name과 description을 사용해서 시스템 메시지를 생성합니다.

    agent_node는 에이전트를 실행하는 데 사용되는 Runnable입니다.

    """
    name: str = Field(description="The name of the agent")
    description: str = Field(description="A description of the agent")
    agent_node: Runnable = Field(description="The agent to run")

    class Config:
        arbitrary_types_allowed = True


class SupervisorAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_agent: str
    summary: str


def create_supervisor_agent(chat_model: BaseChatModel, agents: list[Agent],
                            checkpointer: BaseCheckpointSaver) -> CompiledGraph:
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH. Here's a detailed description of the roles that each worker plays, "
        "which you should use to select your workers. \n\n"
        "{descriptions}"
    )

    agent_names = list(map(lambda agent: agent.name, agents))
    options = ["FINISH"] + agent_names

    parser = SimpleJsonOutputParser()# PydanticOutputParser(pydantic_object=AgentDecision)

    descriptions = "\n\n".join([f"{agent.name}: {agent.description}" for agent in agents])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the dialogue above, who should act next? Or should we finish? "
                "If you think there's nothing wrong with finishing the dialogue, you should select 'FINISH'. "
                "You must select one of the following: {options} and "
                "return it in a JSON message with next_agent key and question to the next_agent, no other text."
                "For example, if you want to select the 'agent' and ask 'What is 3 + 4?', you should return: "
                "{{\"next_agent\": \"agent\", \"question_to_next_agent\": \"What is 3 + 4?\"}}. "
                "And if you want to finish the dialogue, you should make final answer to "
                "the last human question with regard to the above dialogue "
                "as much detail as possible even if there is no dialogue and return: "
                "{{\"next_agent\": \"FINISH\", \"answer\": \"detailed answer to the question\"}}."
            ),
        ]
    ).partial(options=str(options), members=str(options), descriptions=descriptions)

    supervisor = prompt | chat_model | parser

    def call_supervisor(state: SupervisorAgentState):
        res = supervisor.invoke(state)
        messages = []
        if res["next_agent"] != "FINISH":
            messages.append(AIMessage(res["question_to_next_agent"], name="start"))
        else:
            messages.append(AIMessage(res["answer"]))

        return {"messages": messages, "next_agent": res["next_agent"]}

    workflow = StateGraph(SupervisorAgentState)
    workflow.add_node("supervisor", call_supervisor)

    conditional_map = {k.name: k.name for k in agents}
    conditional_map["FINISH"] = END

    workflow.add_conditional_edges("supervisor", lambda x: x['next_agent'], conditional_map)

    for agent in agents:
        workflow.add_node(agent.name, agent.agent_node)
        workflow.add_edge(agent.name, "supervisor")

    workflow.add_edge(START, "supervisor")

    return workflow.compile(checkpointer=checkpointer, debug=True)

