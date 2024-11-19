import asyncio
from datetime import datetime
from typing import AsyncIterable

from ai_core.data_source.base import create_data_source
from ai_core.data_source.model.document import Document
from ai_core.data_source.splitter import create_splitter, SplitterType
from ai_core.data_source.utils.utils import create_collection_name, split_texts
from ai_core.data_source.vectorstore.search_type import Similarity
from ai_core.llm_api_provider import LlmApiProvider


async def main():
    opensearch_hosts = 'localhost:9200'
    opensearch_auth = ('admin', 'Skapfhd3122!@') # For testing only. Don't store credentials in code.

    # 1. 데이터 로딩
    raw_text: str = \
        ("Apache Flink is an open-source stream-processing framework developed by the Apache Software Foundation. "
         "The core of Apache Flink is a distributed streaming data-flow engine written in Java and Scala. "
         "Flink executes arbitrary dataflow programs in a data-parallel and pipelined manner. "
         "Flink's pipelined runtime system enables the execution of bulk/batch and stream processing programs. "
         "Furthermore, Flink's runtime supports the execution of iterative algorithms. "
         "Flink is designed to run in all common cluster environments,"
         " perform computations at in-memory speed and at any scale.")

    # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name="apache_flink_text",
        created_by="your_nickname",
        description="test description",
        data_source_type="text",
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth)

    llm_api_provider = LlmApiProvider.SMART_BEE.value
    embedding_model_name = "text-embedding-3-large"
    collection_name = create_collection_name(data_source.id, embedding_model_name)

    # 3. 데이터 소스에 컬렉션 추가
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=llm_api_provider,
        llm_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441",
        llm_api_url="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
        llm_embedding_model_name=embedding_model_name)

    preview_start = datetime.now()
    preview_data = data_source.load_preview_data(raw_text=[raw_text])
    preview_end = datetime.now()
    print(preview_data)
    print("Preview data loaded in ", str(preview_end - preview_start))

    # 4. 데이터를 Opensearch에 저장
    data_source.save_data(raw_text=[raw_text])
    print("Data saving task started")

    # 5. 데이터 임베딩 및 Vectorstore에 추가
    data = data_source.read_data()
    splitter = create_splitter(SplitterType.RecursiveCharacterTextSplitter, chunk_size=1000, chunk_overlap=200)
    splitted_documents: AsyncIterable[Document] = split_texts(data, splitter)

    def embed_callback(future):
        try:
            embeded, total = future.result()
            print("Embedding task completed. Number of chunks embedded / Total: ", str(embeded), " / ", str(total))
            collection.last_update_succeeded_at = datetime.now()
        except asyncio.exceptions.CancelledError:
            print("Embedding task was cancelled")
            print("Update embedding state to cancelled")
        except Exception as e:
            print("Embedding task failed: ", e)
            print("Update embedding state to failed")

    embed_task = asyncio.create_task(collection.embed_documents_and_overwrite_to_vectorstore(documents=splitted_documents))
    embed_task.add_done_callback(embed_callback)
    print("Embedding task started")

    await embed_task

    # 6. 유사도 검색
    query = "What is Apache Flink?"
    query_results = collection.similarity_search(query=query, search_type=Similarity(k=4))
    for result in query_results:
        print(result.page_content)

    # 7. Retriever 객체를 이용한 유사도 검색
    query_results = data_source.as_retriever(search_type=Similarity(k=5)).invoke(input=query)
    for result in query_results:
        print(result.page_content)

    # 8. Agent 혹은 대화에 연결할 Tool 반환
    tool = data_source.as_retriever_tool(
        search_type=Similarity(k=5),
        name="Very Important tool name",
        description="Very Important Description")

    # Unclosed client session warning 제거
    await collection.vectorstore.async_client.close()
    await data_source.async_opensearch_client.close()


asyncio.run(main())
