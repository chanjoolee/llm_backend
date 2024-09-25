import asyncio
from datetime import datetime
from typing import Iterable

from ai_core import CHROMA_DB_DEFAULT_PERSIST_DIR
from ai_core.data_source.base import DataSourceType, create_data_source
from ai_core.data_source.model.document import Document
from ai_core.data_source.utils import create_collection_name
from ai_core.data_source.vectorstore.search_type import Similarity
from ai_core.llm_api_provider import LlmApiProvider


async def main():
    url = "https://issues.apache.org/jira"
    access_token = "ATATT3xFfGF0Qm1bN-kLR948-seHsNqxD4TMlTilRtqaiedY4GOUZSfJtL9VJkPo67hvGjtZ-_HuPWpgJU4eK2Kx1VI5m7zpToNKAtNiUHn_6SSv68bS9fzNMoTu5tz5BqhXNPjUpQXNwzOzA9m_hyFvbqa28Algkc2wjSkayAYSpYIcTTHpedk=B6DC5F1F"
    project_key = "KAFKA"
    start = 14600

    # 1. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name="apache_kafka_jira",
        created_by="your_nickname",
        description="test description",
        data_source_type=DataSourceType.JIRA.value)

    llm_api_provider = LlmApiProvider.SMART_BEE.value
    embedding_model_name = "text-embedding-3-large"
    collection_name = create_collection_name(data_source.id, embedding_model_name)

    # 2. 데이터 소스에 컬렉션 추가
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=llm_api_provider,
        llm_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441",
        llm_api_url="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
        llm_embedding_model_name=embedding_model_name,
        persist_directory=CHROMA_DB_DEFAULT_PERSIST_DIR)

    preview_start = datetime.now()
    preview_data = data_source.load_preview_data(url=url, access_token=access_token, project_key=project_key, start=start)
    preview_end = datetime.now()
    print(preview_data)
    print("Preview data loaded in ", str(preview_end - preview_start))

    # 3. 데이터를 텍스트 파일로 저장
    # 예외처리 필요
    #   1) 프로젝트 키가 없는 경우
    #     requests.exceptions.HTTPError: The value 'KAFKAK' does not exist for the field 'project'

    save_task = asyncio.create_task(
        data_source.save_data(url=url,
                              access_token=access_token, project_key=project_key, start=start, limit=1000))

    def save_callback(future):
        print("Data saved successfully")

    save_task.add_done_callback(save_callback)
    print("Data saving task started")

    await save_task

    # 4. 데이터 임베딩 및 ChromaDB에 추가
    documents: Iterable[Document] = data_source.read_data()

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

    embed_task = asyncio.create_task(collection.embed_documents_and_overwrite_to_chromadb(documents=documents))

    embed_task.add_done_callback(embed_callback)
    print("Embedding task started")

    await embed_task

    # 5. 유사도 검색
    query = "ConcurrentModificationException"
    result = collection.similarity_search(query=query, k=4)

    print("Similarity search result: ", result)

    # 6. Retriever 객체를 이용한 유사도 검색
    query_results = data_source.as_retriever(search_type=Similarity(k=5)).invoke(input=query)
    for result in query_results:
        print(result.page_content)

    # 7. Agent 혹은 대화에 연결할 Tool 반환
    tool = data_source.as_retriever_tool(
        search_type=Similarity(k=5),
        name="Very Important tool name",
        description="Very Important Description")


asyncio.run(main())
