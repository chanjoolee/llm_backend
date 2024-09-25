데이터소스
==========
데이터소스는 LLM에 답변을 요청할 때 질문과 함께 전달하는 데이터를 담고 있습니다.
LLM은 모르고 있는 내용에 대해 질문하고 싶은 경우 데이터소스를 통해 질문을 구체화하고 답변을 받을 수 있습니다.
예를 들어, SKTelecom datalake에 문제가 생겼을 경우 해결 방안을 LLM에게 물어볼 수 있습니다. LLM은 SKTelecom
datalake에 대한 내용을 모르므로, 질문과 함께 내용을 전달합니다.
내용은 vector store에 embedding 된 형태로 저장됩니다.


텍스트 데이터소스 생성
-------------------------
데이터소스에 대한 정보를 DB에 입력합니다.

.. mermaid::

    sequenceDiagram
        autonumber
        participant b as B/E
        participant u as DataSourceUtils
        box rgb(232, 245, 233) DBMS
            participant t as 데이터소스 테이블
        end
        participant d as DataSource
        b ->> +u: generate_data_source_id(생성한 사람, 데이터소스 이름, 데이터소스 타입) 호출
        u -->> -b: 데이터소스 ID
        b ->> +t: 데이터소스 ID, 이름, 설명, 공개 여부, 태그, 생성 시간, 수정 시간 등
        b ->> +d: 데이터소스 ID, 이름, 설명, 데이터소스 타입(TEXT, GITLAB, ...)
        d -->> -b: 데이터소스 객체

\
\


텍스트 스플릿
-----------------
임베딩을 하기 전에 문서 검색의 정확성을 위해 텍스트 데이터를 적절한 크기로 나누어야 합니다.
Splitter를 이용해 나누어진 각 chunk들은 벡터로 변환 후 vector store에 저장됩니다.

.. mermaid::

    sequenceDiagram
        autonumber
        participant b as B/E
        participant s as Splitter
        b ->> +s: 스플리터 종류, text data, chunk size, chunk overlap
        s -->> -b: splitted texts

\
\


컬렉션 객체 생성
-----------------
Vector store에 저장할 컬렉션을 생성합니다.
컬렉션은 embedding된 데이터를 저장하는 단위입니다.

.. mermaid::

    sequenceDiagram
        autonumber
        participant b as B/E
        participant u as DataSourceUtils
        participant d as DataSource
        b ->> +u: generate_collection_name(데이터소스 ID, 임베딩 모델) 호출
        u -->> -b: 컬렉션 이름
        b ->> +d: add_collection(컬렉션 이름, 데이터소스 ID, LLM API TYPE, LLM API KEY, LLM API URL, 임베딩 모델, 스플릿된 데이터) 호출
        d -->> -b: 컬렉션 객체

\
\


임베딩 & 임베딩 업데이트
----------------------------
임베딩은 텍스트 데이터를 벡터로 변환하는 과정입니다.
임베딩 업데이트는 벡터 데이터를 Vector store에 저장하는 과정이며, 컬렉션 단위로 저장됩니다.
같은 이름의 컬렉션이 Vector store에 이미 존재한다면 Overwrite 되고, 없다면 새로 생성합니다.

.. mermaid::

    sequenceDiagram
        autonumber
        participant b as B/E
        participant c as Collection
        participant v as VectorStore
        box rgb(232, 245, 233) DBMS
            participant t as 컬렉션 테이블
        end
        b ->> +c: embed_documents_and_save_to_chromadb() 호출
        b ->> +t: 컬렉션 이름, 데이터소스 ID, LLM API TYPE, 임베딩 모델, 업데이트 시작 시간, 임베딩 진행 상태
        c ->> +v: 컬렉션 이름, 임베딩 모델, 벡터 데이터
        v -->> -c: 저장 성공한 데이터
        c -->> -b: 저장 성공한 데이터
        b ->> +t: 컬렉션 이름, 데이터소스 ID, 업데이트 종료 시간, 마지막 업데이트 성공 시간, 임베딩 진행 상태

\
\


데이터 삭제
------------
Vector store에 저장된 vector data를 삭제합니다.
삭제는 Vector store의 컬렉션(collection) 단위로 이루어집니다.

.. mermaid::

    sequenceDiagram
        autonumber
        participant b as B/E
        participant c as Collection
        participant v as VectorStore
        box rgb(232, 245, 233) DBMS
            participant t as 컬렉션 테이블
        end
        b ->> +c: delete_collection() 호출
        c ->> +v: delete_collection(컬렉션 이름) 호출
        v -->> -c: None
        c -->> -b: None
        b ->> +t: DB 삭제

\
\


데이터소스 삭제
-----------------
데이터소스에 대한 정보를 DB에서 삭제합니다.
더불어 Vector store에 저장된 vector data가 있다면 함께 삭제합니다.

.. mermaid::

    sequenceDiagram
        autonumber
        participant b as B/E
        participant d as 데이터소스
        participant c as Collection
        participant v as VectorStore
        box rgb(232, 245, 233) DBMS
            participant td as 데이터소스 테이블
            participant tc as 컬렉션 테이블
        end
        loop
            b ->> +c: delete_collection() 호출
            c ->> +v: delete_collection(컬렉션 이름) 호출
            v -->> -c: None
            c -->> -b: None
            b ->> + tc: DB 삭제
        end
        b ->> +td: DB 삭제

\
\


데이터 검색
-----------------
Vector store에 저장된 vector data를 검색합니다.

.. mermaid::

    sequenceDiagram
        autonumber
        participant b as B/E
        participant d as DataSource
        participant r as Retriever
        participant v as VectorStore
        b ->> +d: as_retriever(Search Tpe) 호출
        d -->> -b: Retriever 객체
        b ->> +r: invoke(쿼리) 호출
        r ->> +v: 컬렉션 이름, 검색 쿼리, 결과 개수, Search Type
        v -->> -r: 검색 결과
        r -->> -b: 검색 결과