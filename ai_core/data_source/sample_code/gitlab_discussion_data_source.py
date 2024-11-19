import asyncio
from datetime import datetime

from ai_core.data_source.base import DataSourceType, create_data_source
from ai_core.data_source.utils.utils import create_collection_name
from ai_core.data_source.vectorstore.search_type import Similarity
from ai_core.llm_api_provider import LlmApiProvider


async def main():
    url = "https://gitlab.tde.sktelecom.com/"
    namespace = "SWIFT"
    project_name = "streams"
    private_token = "tde2-4frDqjSRER4DBGGDzwwY"
    opensearch_hosts = 'localhost:9200'
    opensearch_auth = ('admin', 'Skapfhd3122!@') # For testing only. Don't store credentials in code.

    # 1. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name="swift_streams_gitlab_discussion",
        created_by="your_nickname",
        description="test description",
        data_source_type=DataSourceType.GITLAB_DISCUSSION.value,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth)

    embedding_model_name = "text-embedding-3-large"
    collection_name = create_collection_name(data_source.id, embedding_model_name)

    # 2. 데이터 소스에 컬렉션 추가
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider.SMART_BEE.value,
        llm_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441",
        llm_api_url="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
        llm_embedding_model_name="text-embedding-3-large")

    # 3. 데이터를 텍스트 파일로 저장
    data_source.save_data(url=url, namespace=namespace, project_name=project_name, private_token=private_token)
    print("Data saved successfully")

    # 4. 데이터 임베딩 및 Vectorstore에 추가
    data = data_source.read_data()

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

    embed_task = asyncio.create_task(collection.embed_documents_and_overwrite_to_vectorstore(texts=data))
    embed_task.add_done_callback(embed_callback)
    print("Embedding task started")

    await embed_task

    # 5. 유사도 검색
    query = "What is Swift Streams?"
    query_results = collection.similarity_search(query=query, search_type=Similarity(k=4))
    for result in query_results:
        print(result.page_content)

    # 6. Retriever 객체를 이용한 유사도 검색
    query_results = data_source.as_retriever(search_type=Similarity(k=5)).invoke(input=query)
    for result in query_results:
        print(result.page_content)

    # Unclosed client session warning 제거
    await collection.vectorstore.async_client.close()


asyncio.run(main())
