import asyncio
from datetime import datetime

from ai_core.data_source.base import create_data_source, DataSourceType
from ai_core.data_source.splitter import create_splitter, SplitterType
from ai_core.data_source.utils.utils import create_collection_name, split_texts
from ai_core.data_source.utils.time_utils import get_iso_8601_current_time, iso_8601_str_to_datetime
from ai_core.data_source.vectorstore.search_type import Similarity
from ai_core.llm_api_provider import LlmApiProvider


async def main():
    opensearch_hosts = 'localhost:9200'
    opensearch_auth = ('admin', 'Skapfhd3122!@') # For testing only. Don't store credentials in code.

    # 1. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name="tde_confluence_data_technology",
        created_by="your_nickname",
        description="test description",
        data_source_type=DataSourceType.CONFLUENCE.value,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth)

    llm_api_provider = LlmApiProvider.SMART_BEE.value
    embedding_model_name = "text-embedding-3-large"
    collection_name = create_collection_name(data_source.id, embedding_model_name)

    last_update_succeeded_at = "2024-11-01T00:00:53.000+0900"

    # 2. 데이터 소스에 컬렉션 추가
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=llm_api_provider,
        llm_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441",
        llm_api_url="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
        llm_embedding_model_name=embedding_model_name,
        last_update_succeeded_at=iso_8601_str_to_datetime(last_update_succeeded_at))


    # 3. 데이터를 Opensearch에 저장
    url = "https://confluence.tde.sktelecom.com/"
    access_token = "MDA0MzQ2MzU3Mjk0OriGVPM4nVKmG/pbFOCuftXYKMMs"
    space_key = "DATAENG"

    preview_start = datetime.now()
    preview_data = data_source.load_preview_data(url=url, access_token=access_token, space_key=space_key)
    preview_end = datetime.now()
    print(preview_data)
    print("Preview data loaded in ", str(preview_end - preview_start))

    data_source.save_data(last_update_succeeded_at=get_iso_8601_current_time(),
                          url=url, access_token=access_token, space_key=space_key)

    print("Data saved successfully")

    # 4. 데이터 임베딩 및 Opensearch에 추가
    documents = await data_source.read_data()

    splitter_type = SplitterType.RecursiveCharacterTextSplitter
    splitter = create_splitter(splitter_type=splitter_type, chunk_size=20000, chunk_overlap=0)
    splitted_documents = split_texts(documents=documents, splitter=splitter)

    def embed_callback(future):
        try:
            embeded, total = future.result()
            print("Embedding task completed. Number of chunks embedded / Total: ", str(embeded), " / ", str(total))
            collection.last_update_succeeded_at = get_iso_8601_current_time()
        except asyncio.exceptions.CancelledError:
            print("Embedding task was cancelled")
            print("Update embedding state to cancelled")
        except Exception as e:
            print("Embedding task failed: ", e)
            print("Update embedding state to failed")

    embed_task = asyncio.create_task(
        collection.embed_documents_and_overwrite_to_vectorstore(documents=splitted_documents,
                                                                last_update_succeeded_at=get_iso_8601_current_time()))
    embed_task.add_done_callback(embed_callback)
    print("Embedding task started")

    await embed_task

    # 5. 유사도 검색
    query = "Azure Network"
    result = collection.similarity_search(query=query, search_type=Similarity(k=4))

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

    # Unclosed client session warning 제거
    await data_source.async_opensearch_client.close()
    await collection.vectorstore.async_client.close()


asyncio.run(main())
