.. Daisy documentation master file, created by
   sphinx-quickstart on Thu May 16 09:54:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Daisy's documentation!
=================================
**Daisy** 는 LLM 기반 에이전트를 이용해 **시스템 운영을 자동화**\  또는 **간소화**\하기 위한 시스템입니다.
또한 **업무와 연관된 정보 조회**\, **시스템의 상태 진단**\하거나 **일상 업무들을 지원**\할 수 있습니다.


.. figure:: /_static/current_operation.png
   :figclass: align-center
   :width: 800px

   모니터링 알람 대응 과정 (AS-IS)



.. figure:: /_static/daisy_operation.png
   :figclass: align-center
   :width: 800px

   모니터링 알람 대응 과정 (TO-BE)



Daisy는 Langchain 기반으로 **프롬프트, 도구, 데이터소스, 에이전트**\와 같은 컴포넌트를 **생성하고 테스트할 수 있는 환경을 제공**\합니다.
생성된 **컴포넌트를 이용해서 대화를 나눌 수 있습니다**\.


.. image:: /_static/daisy_components.png
   :width: 800px


Daisy는 모니터링 시스템으로부터 **경고를 받으면 시스템 대화를 시작**\합니다.
시스템 대화는 **슈퍼바이저 에이전트**\를 가지고 있으며 수행에 필요한 운영 에이전트의 실행을 반복하면서 얻은 결과를 슬랙이나 문자를 통해 알림을 주게 됩니다.
에이전트의 결과를 통해 사용자는 어떤 조치를 취해야 하는지 판단할 수 있습니다.
또한 에이전트의 결과에서 의문점이 있다면 **시스템 대화로부터 개인 대화를 이어나갈 수**\ 있습니다.


.. image:: /_static/system_conv.png
   :width: 800px


.. figure:: /_static/overview.png
   :figclass: align-center
   :width: 800px

   Daisy 전체 구성도


Contents
-----------

.. toctree::

   사용자 관리 <user>
   적용 시나리오 <scenario>
   컴포넌트 <component>
   대화 <conversation>
   시스템 대화 <system_conv>
   프롬프트 <prompt>
   도구 <tool>
   데이터소스 <data-source>
   에이전트 <agent>
   배경지식/RAG <base_knowledge/rag>


