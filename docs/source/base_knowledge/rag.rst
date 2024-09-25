RAG란?
=================
| Retrieval-Augmented Generation 의 약자이며 직역하면 검색 증강 생성입니다.
| 검색 기능이 추가된 LLM이라고 생각하면 쉽습니다.
| 여기에서 말하는 검색은 LLM에게 검색을 시키는 것이 아닙니다.
| LLM에게 질의를 할 때 사용자가 가진 자료를 검색하여 LLM이 모르고 있는 정보를 제공해줌으로써 모델의 학습없이 특정 도메인이나 조직의 내부 지식을 기반으로 답변을 받을 수 있습니다.
|
| RAG를 사용함으로써 다음과 같은 이점을 얻을 수 있습니다.

* 할루시네이션 문제를 줄여줍니다.
* 최신 정보를 바탕으로 답변을 생성할 수 있습니다.
* 사용자가 신뢰도 높은 자료를 제공함으로써 답변의 신뢰도를 높입니다.
* 모델 트레이닝에 비해 저렴합니다.


RAG의 동작방식
--------------------
.. image:: /_static/rag.png
   :width: 800px
출처: <https://qdrant.tech/articles/langchain-integration>


Loading
--------------------
| 다양한 형태의 데이터 소스를 읽어서 Text 데이터로 만듭니다.
| PDF, docx, xlsx, pptx, html, confluence …
| Unstructured 라는 오픈소스 라이브러리를 이용


Chunking
--------------------
| LLM으로부터 최적의 답변을 얻기 위해 자료를 적절한 크기로 자릅니다.
| Chunk의 크기가 너무 크다면 원하는 답변과 관계가 없는 내용이 포함될 가능성이 높아집니다.
| Chunk의 크기가 너무 작다면 문서의 앞뒤 문맥을 고려하지 않은 단편적인 내용을 가지고 답변을 줄 가능성이 높아집니다.
| 따라서 적절한 Chunk의 크기를 설정하는 것은 답변의 품질을 결정하는 중요한 요소입니다.

* 각 Chunk가 독립적인 의미를 갖도록 문장, 구절, 단락 등 문서의 구조를 기준으로 나누는 것이 좋습니다.
* LLM 모델의 최대 입력 토큰 수, 비용을 고려하여 나누어야 합니다.


Splitter 구현체
--------------------

* CharacterTextSplitter: python의 split함수와 동일. seperator, chunk size, chunk overlap을 매개변수로 받아 split 실행합니다.
  chunk overlap은 각 split간 중복되는 text의 사이즈를 결정합니다.
* RecursiveCharacterTextSplitter: [’\n\n’, ‘\n’, ‘ ‘, ‘’] 를 가지고 텍스트를 재귀적으로 분할하여 의미적으로 관련 있는 텍스트들이 같이 있도록 하는 것이 목적.
    * cpp, go, java, scala, html, markdown 등 랭귀지를 파라메터로 전달하면 각 언어의 예약 단어를 기준으로 split


Embedding
--------------------
| 임베딩은 텍스트 데이터를 숫자로 이루어진 벡터로 변환하는 과정을 말합니다.
| 벡터 표현을 사용하면 텍스트 데이터를 벡터 공간 내에서 수학적으로 다룰 수 있게 되며, 이를 통해 유사성을 계산할 수 있습니다.
| 임베딩 모델의 성능이 높을수록 일관되고 관련성이 높은 결과가 나옵니다.
| 벡터의 차원 갯수가 많을수록 성능이 좋지만 데이터 크기가 커지는 단점이 있습니다.
| Hugging Face에서 오픈소스 임베딩 모델을 다운로드 받아 사용할 수도 있지만, 컴퓨팅 자원이 풍부하지 않다면 OpenAI, Google, Anthropic 등 회사에서 제공하는 유료 API를 이용하는 것이 좋습니다.

.. image:: /_static/how_embeddings_work.jpg
   :width: 800px

vector들간의 유사도 계산

* Euclidean distance: 두 벡터 간의 직선 거리를 측정. 문장의 길이나 단어 빈도수에 크게 영향을 받음
* Cosine similarity: 두 벡터 간의 각도를 측정. 문장의 길이나 단어 빈도수와 독립적

.. image:: /_static/similar_embeddings.jpg
   :width: 800px

참고: Embedding projector
<https://projector.tensorflow.org/>


Vector Store
--------------------
| 임베딩을 통해 생성된 고차원의 벡터 데이터를 효율적으로 저장하고 조회할 수 있도록 설계된 데이터베이스.
| 전통적인 RDBMS와 다르게 Cosine similarity, Euclidean distance 를 기반으로 데이터를 조회한다.
| Chroma DB, Qdrant, Pinecone 등이 많이 쓰이며 ElasticSearch, Redis, Mysql 등도 vectore store로 사용할 수 있다.

참고: <https://benchmark.vectorview.ai/vectordbs.html>


RAG가 가진 한계
--------------------

한 번의 Vector Store 검색으로 만족스러운 답변을 얻지 못하는 경우가 많습니다.
그 이유는 다음과 같습니다.

* Vector Store에 저장되어있는 자료가 답변을 만들어내기에 충분한 정보를 제공하지 못함
* 질의가 구체적이지 않음

| RAG가 가진 한계를 극복하기 위한 연구가 활발히 진행되고 있으며 가장 유명한 논문인 Corrective RAG(https://arxiv.org/pdf/2401.15884) 와
| Self RAG(https://arxiv.org/pdf/2310.11511)를 구현한 사례들이 많이 공개되고 있습니다.
| LangChain 측에서도 lang-graph를 이용한 구현을 제시하고 있습니다.


Self RAG
--------------------
Reflection Token 을 사용하여 RAG의 한계를 극복하는 방법입니다.
Retrieve한 자료와 Question의 연관성을 평가하여 자료를 사용하여 LLM에 질의를 할지, Question을 수정할지 판단합니다.
또한, LLM에게서 답변을 받았다면 답변이 유효한지를 평가합니다.

.. image:: /_static/reflection_tokens.png
   :width: 800px

.. image:: /_static/self_rag.png
   :width: 800px

.. image:: /_static/self_rag2.png
   :width: 800px


CRAG(Corrective RAG)
--------------------
Self RAG와 유사하게 Retrieve한 자료와 Question의 연관성을 평가한다.
Self RAG와 다른점은 연관성이 없다고 판단하면 질문을 웹 서치에 적합한 형태로 수정한 후 웹 서치를 한다.
웹 서치의 결과를 프롬프트에 추가하여 질문을 한다.

.. image:: /_static/corrective_rag.jpeg
   :width: 800px

| 참고
| <https://blog.langchain.dev/agentic-rag-with-langgraph/>
| <https://github.com/langchain-ai/langchain/blob/master/cookbook/langgraph_self_rag.ipynb>