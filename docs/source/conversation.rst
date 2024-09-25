대화
=====

기본 대화
---------

기본 대화는 아무런 컴포넌트를 이용하지 않고 지정한 LLM 모델과 대화를 나누는 것이다. 이를 위해서 먼저 대화 객체를 생성해야 한다.

기본 대화 생성
~~~~~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
      autonumber
      participant b as B/E
      participant f as ConversationFactory
      participant c as Conversation
      box rgb(232, 245, 233) DBMS
         participant ct as 대화 테이블
      end
      b->>+f: create_basic_conversation(params)
      Note right of b: 매개변수는 하단을 참조
      Note right of f: 생성 시점에는 LLM API와 <br/>연결을 시도하지 않습니다.
      f ->> c: new Conversation(params)
      f ->> c: create_agent()
      Note right of f: 아무런 컴포넌트 없이 에이전트를 생성합니다.
      f-->>-b: 대화 객체
      b->>ct: 사용자 ID, 대화 ID, LLM API 이름, LLM 모델, 생성 시간

\

.. autofunction:: ai_core.conversation.base.ConversationFactory.create_basic_conversation

.. note::
    LLM API 주소 대신에 http client 객체를 받을 수 있는지 확인이 필요

\
\

기본 대화 진행
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. mermaid::

   sequenceDiagram
      autonumber
      participant b as B/E
      participant c as Conversation
      participant l as LLM API
      box rgb(232, 245, 233) DBMS
         participant lt as Langchain 히스토리
         participant mt as 메시지 테이블
      end

      loop
         b->>+c: stream(params)
         c-->>b: AsyncIterator
         Note right of b: 매개변수는 하단을 참조
         b->>mt: 사용자 ID, 대화 ID, 시간, 사용자 메시지
         c->>lt: 대화 ID
         Note right of lt: 메시지를 보낼 때마다 <br/> 매번 히스토리를 읽습니다.
         lt-->>c: 메시지 목록
         c->>+l: 사용자 메시지 (HumanMessage)
         c->>lt: 대화 ID, 사용자 메시지 (HumanMessage)
         l-->>c: AI 메시지 (AIMessageChunk)
         c-->>b: AI 메시지 (DaisyMessageChunk)
         l-->>c: AI 메시지 (AIMessageChunk)
         c-->>b: AI 메시지 (DaisyMessageChunk)
         c->>lt: 대화 ID, AI 메시지 (AIMessage)
         b->>mt: 사용자 ID, 대화 ID, 시간, 토큰 사용량, AI 메시지
      end

\

.. autofunction:: ai_core.conversation.base.Conversation.stream

.. note::
    메시지는 스트리밍 방식으로 전달됩니다. 현재는 이 방식으로 토큰 수를 알 수는 없습니다.

\
\

기본 대화 생성 (재개)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
      autonumber
      participant b as B/E
      participant f as ConversationFactory
      participant c as Conversation
      box rgb(232, 245, 233) DBMS
         participant ct as 대화 테이블
      end
      ct-->>b: 사용자 ID, 대화 ID, LLM API 이름, LLM 모델, 생성 시간
      b->>ct: 대화 ID
      ct-->>b: 대화 히스토리
      b->>+f: create_basic_conversation(params)
      Note right of b: 매개변수는 하단을 참조
      Note right of f: 생성 시점에는 LLM API와 <br/>연결을 시도하지 않습니다.
      f ->> c: new Conversation(params)
      f ->> c: create_agent()
      Note right of f: 아무런 컴포넌트 없이 에이전트를 생성합니다.
      f-->>-b: 대화 객체
      b->>ct: 사용자 ID, 대화 ID, LLM API 이름, LLM 모델, 생성 시간

.. autofunction:: ai_core.conversation.ConversationFactory.create_basic_conversation

\
\

기본 대화 검색
~~~~~~~~~~~~~~~~~~~~~~~~
기본 대화에서는 컴포넌트가 존재하지 않으므로 검색어를 포함하거나 포함되지 않은 케이스에서만 노출이 가능합니다. 검색할 때는 Conversation 객체가 필요 없습니다.

.. mermaid::

   sequenceDiagram
      autonumber
      participant b as B/E
      box rgb(232, 245, 233)
         participant m as DBMS
         participant cdb as 대화 테이블
         participant d as B/E 테이블
      end
      b->>+m: 검색어, (컴포넌트, 공개여부, 사용자 ID)
      m->>d: 사용자 ID
      m->>cdb: 대화 ID 목록
      m-->>-b: 대화 목록 ((검색어가 포함된) 마지막 메시지)

\

.. note::
    기본적으로 검색의 대상은 자신의 권한 안에서 볼 수 있는 대화들입니다.

    - 관리자/개발자: 공개된 대화 + 본인 대화
    - 게스트: 공개된 대화

\
\

기본 대화 삭제
~~~~~~~~~~~~~~~
대화를 삭제하려면 먼저 Conversation 객체를 생성해야 합니다.

.. mermaid::

   sequenceDiagram
      autonumber
      participant b as B/E
      participant c as Conversation
      box rgb(232, 245, 233) DBMS
         participant lt as Langchain 히스토리
         participant cdb as 대화 테이블
         participant mt as 메시지 테이블
      end
      b->>+c: clear(대화 ID)
      c->>lt: delete 대화 ID
      lt-->>c: 성공 여부
      c-->>-b: 성공 여부
      b->>cdb: delete 사용자 ID, 대화 ID
      cdb-->>b: 성공 여부
      b->>mt: delete 사용자 ID, 대화 ID
      mt-->>b: 성공 여부

\
\
.. autofunction:: ai_core.conversation.base.Conversation.clear
\
\

기본 대화 예제 코드
----------------------

.. code-block:: python

    from ai_core.conversation.base import ConversationFactory

    conversation = ConversationFactory.create_basic_conversation(llm_api_provider="openai",
        llm_model="gpt-4",
        llm_api_key="api_key",
        llm_api_url="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
        temperature=0.5,
        max_tokens=1024,
        history_connection_str="mysql+pymysql://id:password!@localhost:3306/daisy",
        history_table_name="session_store",
    )

    response = conversation.invoke("session1", "apache kafka에 대해 설명해줘")

    # 히스토리를 기억하고 답변
    response = conversation.invoke("session1", "토픽에 대해 더 자세히 설명해줘")

    # 스트리밍 방식
    for message in conversation.stream("session1", "apache kafka에 대해 설명해줘"):
        print(message)

    # 대화 삭제
    conversation.clear("session1")

    # 히스토리가 삭제되어 기억하지 못함
    response = conversation.invoke("session1", "토픽에 대해 더 자세히 설명해줘")


컴포넌트가 포함된 대화
-------------------------

프롬프트, 도구, 데이터 소스와 같은 컴포넌트를 포함한 대화다. 이를 위해서는 먼저 대화 객체를 생성해야 한다. 대화 생성 시에는 기본 대화에서 필요한 LLM API
정보뿐만 아니라 컴포넌트 정보와 콜백 핸들러를 함께 전달해야 한다. 대화 진행 도중 컴포넌트가 삭제된 경우에는 해당 컴포넌트를 제외하고 계속 진행한다.

컴포넌트가 포함된 대화 생성
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
      autonumber
      participant b as B/E
      participant f as ConversationFactory
      participant c as Conversation
      box rgb(232, 245, 233) DBMS
         participant ct as 대화 테이블
      end
      b->>+f: create_conversation(params)
      Note right of b: 매개변수는 하단을 참조
      Note right of f: 생성 시점에는 LLM API와 <br/>연결을 시도하지 않습니다.
      f ->> c: new Conversation(params)
      f-->>-b: 대화 객체
      b ->> c: add_{component_type}(component)
      b ->> c: add_{component_type}(component)
      b ->> c: create_agent()
      Note right of b: 추가된 컴포넌트가 포함된 에이전트를 생성합니다.
      b->>ct: 사용자 ID, 대화 ID, LLM API 이름, LLM 모델, 컴포넌트 목록, 생성 시간

\
\

컴포넌트가 포함된 대화 진행
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. mermaid::

   sequenceDiagram
      autonumber
      participant b as B/E
      participant c as Conversation
      participant a as Agent
      participant cp as Component
      participant l as LLM API
      box rgb(232, 245, 233) DBMS
         participant lt as Langchain 히스토리
         participant mt as 메시지 테이블
      end

      loop 메시지
         b->>+c: stream(params)
         c-->>b: AsyncIterator
         c->>a: 사용자 메시지 (HumanMessage)
         b->>mt: 사용자 ID, 대화 ID, 시간, 사용자 메시지
         a->>+l: 도구 목록, 사용자 메시지 (HumanMessage)
         c->>lt: 대화 ID, 사용자 메시지 (HumanMessage)
         loop 에이전트의 액션 수행
             alt is 도구 호출
                 l-->>a: tool_calls (AIMessage)
                 a-->>b: 도구 호출 시작 메시지 (DaisyMessage)
                 b->>mt: 도구 호출 시작 메시지
                 a->>+cp: 도구 호출
                 cp-->>a: 호출 결과
                 a-->>b: 도구 호출 종료 메시지 (DaisyMessage)
                 b->>mt: 도구 호출 종료 메시지
                 a->>+l: 도구 목록, 사용자 메시지 (HumanMessage), 도구 메시지 (ToolMessage)
             else is AI 메시지 청크
                 l-->>a: AI 메시지 (AIMessageChunk)
                 a-->>b: AI 메시지 (DaisyMessageChunk)
                 l-->>a: AI 메시지 (AIMessageChunk)
                 a-->>b: AI 메시지 (DaisyMessageChunk)
                 a->>lt: 대화 ID, AI 메시지 (AIMessage)
             end
         end
         b->>mt: 사용자 ID, 대화 ID, 시간, 토큰 사용량, AI 메시지
      end

\
\

.. warning::

    대화가 진행되는 중에 사용 중인 컴포넌트가 삭제되거나 LLM API나 모델이 관리자에 의해 삭제될 수 있습니다. 컴포넌트가 삭제되면 해당 컴포넌트를 제외한
    나머지만 이용해서 대화 진행이 가능합니다. 하지만 LLM API를 활용할 수 없는 경우에는 대화가 중단됩니다.

\
\

대화 제목 자동 생성
-------------------------

대화 제목은 초기에 '제목 없음' 상태로 보여집니다.
대화 제목을 사용자가 직접 입력할 수도 있으며 자동 생성 버튼을 누른 경우에 LLM을 통해 생성할 수 있습니다.
또한 자동 생성된 제목을 사용자가 수정할 수도 있습니다.대화 제목 자동 생성이나 수정은 기본적으로 대화의 주인만 가능합니다.
따라서 다른 사용자의 공개된 대화를 보는 사람은 제목을 변경할 수 없습니다.
만약 대화를 이어나가서 개인 대화가 생성되는 경우에는 그 대화의 주인이 되므로 제목을 변경할 수 있습니다.
시스템 대화는 일반적으로 에이전트가 처리하게 되고 에이전트의 처리가 완료되면 대화의 제목이 자동생성되어야 합니다.
시스템 대화 제목의 자동 생성이나 수정은 관리자(OWNER 포함)만 가능합니다.

.. mermaid::

   sequenceDiagram
      autonumber
      participant b as B/E
      participant c as Conversation
      participant l as LLM API
      box rgb(232, 245, 233) DBMS
         participant lt as Langchain 히스토리
         participant cdb as 대화 테이블
      end
      b->>+c: generate_title(대화 ID)
      c->>lt: select 대화 ID
      lt-->>c: 메시지 목록
      c->>l: 메시지 목록 + 제목 생성 요청 메시지 (LLMMessage)
      l-->>c: 제목 생성 응답 메시지 (LLMMessage)
      c-->>-b: 대화 제목
      b->>cdb: update 사용자 ID, 대화 ID, 대화 제목
      cdb-->>b: 성공 여부

\
\
.. autofunction:: ai_core.conversation.Conversation.base.generate_title
\
\

대화의 복제
-------------------------

사용자는 시스템 대화나 공개된 다른 사람의 대화로부터 대화를 이어나갈 수 있습니다.
대화를 이어나가면 새로운 개인 대화가 시작되게 됩니다. 따라서 기존 대화 히스토리는 자신의 개인 대화 히스토리로 복제되어야 합니다.


.. mermaid::

   sequenceDiagram
      autonumber
      participant b as B/E
      participant f as ConversationFactory
      participant c as Conversation
      box rgb(232, 245, 233) DBMS
         participant ct as 대화 테이블
      end
      b->>+f: copy_conversation_history(대화 ID, 신규 대화 ID)
      b->>+f: create_conversation(params)
      Note right of b: 신규 대화 ID로 생성합니다.
      f ->> c: new Conversation(params)
      f-->>-b: 대화 객체
      b ->> c: add_{component_type}(component)
      b ->> c: add_{component_type}(component)
      b ->> c: create_agent()
      b->>ct: 사용자 ID, 대화 ID, LLM API 이름, LLM 모델, 컴포넌트 목록, 생성 시간

\
\
.. autofunction:: ai_core.conversation.Conversation.base.copy_conversation
\
\