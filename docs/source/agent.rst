에이전트
=========

에이전트의 핵심 아이디어는 **LLM을 이용해서 수행할 일련의 작업을 선택하고 실행**\하는 것입니다.

다음은 Langgraph에서 정의한 에이전트를 구성하기 위해 필요한 요소들입니다.

- 흐름의 제어 (순환, 분기, 일시 정지 등)
- 영속성 (각 단계에서의 상태 저장, Human-in-the-loop)
- 스트리밍 지원 (각 노드에서의 결과를 스트리밍으로 전달 (토큰 스트리밍 포함))

ReAct 에이전트
--------------

ReAct는 "Reasoning and Acting"의 약자입니다.
이는 **LLM의 강점과 도구를 통해 외부와 상호 작용하는 능력을 결합**\한 에이전트 패턴입니다.
에이전트는 과제를 해결하거나 질문에 답하기 위해 사고(Thought), 행동(Action), 관찰(Observation)의 순환을 따릅니다.

:프롬프트:
   다음 질문에 최선을 다해 답하십시오. 사용 가능한 도구는 다음과 같습니다:

   {tools}

   다음 형식을 사용하십시오:

   Question: 답해야 할 입력 질문

   **Thought: 무엇을 해야 할지 항상 생각하십시오**

   **Action: 취해야 할 행동, [{tool_names}] 중 하나여야 합니다**

   **Action Input: 행동에 대한 입력**

   **Observation: 행동의 결과**

   ... (이 Thought/Action/Action Input/Observation N번 반복될 수 있습니다.)

   Thought: 이제 최종 답을 알고 있습니다.

   Final Answer: the final answer to the original input question

   시작!

   Question: {input}

   Thought:{agent_scratchpad}

| 출처 <https://smith.langchain.com/hub/hwchase17/react>

슈퍼바이저 에이전트
------------------

모니터링 알람을 처리하는 시스템 에이전트는 슈퍼바이저입니다. 일반적으로 에이전트는 다음과 같은 상태를 가집니다.

.. image:: /_static/agent_state.png
   :width: 800px


슈퍼바이저 에이전트는 운영 에이전트들 중에서 작업을 처리하기에 적합한 에이전트에게 지시를 내리고 결과를 수집하는 것을 반복해 나갑니다.
슈퍼바이저 에이전트나 운영 에이전트들은 기본적으로 ReAct 방식이며 다른 방식으로도 구현될 수 있습니다.

.. image:: /_static/supervisor.png
   :width: 800px

| 출처 <https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/>
