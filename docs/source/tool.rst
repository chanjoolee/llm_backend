도구
=====

도구를 통해서 LLM은 외부로부터 새로운 정보를 얻거나 작업을 수행할 수 있습니다.

도구는 다음 항목으로 이루어져 있습니다.

- 도구의 이름
- 도구의 설명
- 소스 코드 (프론트에서 직접 넣거나 아니면 Gitlab Repository에서 지정 가능)

**LLM은 직접 도구를 호출하지 않습니다.** 단지 LLM은 어떤 도구를 호출하라고 클라이언트에게 알려 줍니다.
그러면 **클라이언트는 해당 도구를 호출할 지 여부를 판단**\해서 그에 따라 행동하게 됩니다. **RAG**\도 LLM 에겐 하나의 도구입니다.


.. mermaid::

   sequenceDiagram
      autonumber
      participant c as LLM Client
      participant l as LLM
      participant t as Tool
      c ->> l: 질문, 도구 스키마 목록
      loop until 도구 호출 메시지
         l -->> c: 호출할 도구 이름, 인자, 호출 ID
         c ->> t: 인자
         t -->> c: 결과
         c ->> l: 질문, 도구 스키마 목록, 호출 ID, 결과
      end
      l -->> c: 답변


파이썬으로 작성된 간단한 도구 예제입니다.

.. code-block:: python

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b.

        Args:
            a: first int
            b: second int
        """
        return a + b

도구 호출 콜백
----------------------

대화나 에이전트에서 도구가 호출되면서 발생되는 이벤트를 처리할 수 있는 콜백 클래스입니다.

.. autoclass:: ai_core.callback.base.DaisyCallbackHandler
    :members:
    :exclude-members: on_tool_start, on_tool_end, on_tool_error

\

도구 호출 정보 결과를 담고 있는 ToolCall 클래스입니다.

.. autoclass:: ai_core.conversation.messages.base.ToolCall
    :members:

\

도구와 콜백이 포함된 대화 생성
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
      autonumber
      participant b as B/E
      participant f as ConversationFactory
      participant c as Conversation
      box rgb(232, 245, 233) 파일시스템
         participant fs as 도구 파일 저장소
      end
      b->>+f: create_conversation(params)
      f ->> c: new Conversation(params)
      f-->>-b: 대화 객체
      b ->> c: add_callback_handler(callback_handler)
      b ->> c: add_callback_handler(callback_handler)
      b ->> c: add_tool(params)
      Note right of b: 매개변수는 하단을 참조
      c ->> fs: 도구 파일 읽기
      c -->> b: 검증 결과
      b ->> c: add_tool(params)
      c ->> fs: 도구 파일 읽기
      c -->> b: 검증 결과
      b ->> c: create_chain()
      Note right of b: 추가된 컴포넌트가 포함된 체인을 생성합니다.

\
\

.. autofunction:: ai_core.conversation.base.Conversation.add_tool

\
\

