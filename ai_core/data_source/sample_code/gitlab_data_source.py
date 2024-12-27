import asyncio
from datetime import datetime

from ai_core.data_source.base import DataSourceType, create_data_source
from ai_core.data_source.splitter import create_splitter, SplitterType
from ai_core.data_source.utils.time_utils import get_iso_8601_current_time, iso_8601_str_to_datetime
from ai_core.data_source.utils.utils import create_collection_name, split_texts
from ai_core.data_source.vectorstore.search_type import Similarity
from ai_core.llm_api_provider import LlmApiProvider


async def main():

    url = "https://gitlab.tde.sktelecom.com/"
    namespace = "ORYX/DAISY"
    project_name = "daisy_backend"
    branch = "develop"
    private_token = "tde2-4frDqjSRER4DBGGDzwwY"

    opensearch_hosts = 'localhost:9200'
    opensearch_auth = ('admin', 'Skapfhd3122!@') # For testing only. Don't store credentials in code.

    last_update_succeeded_at = "2024-11-01T00:00:53.000+0900"

    # 1. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name="daisy_sourcecode",
        created_by="your_nickname",
        description="test description",
        data_source_type=DataSourceType.GITLAB.value,
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
        llm_embedding_model_name=embedding_model_name,
        last_update_succeeded_at=iso_8601_str_to_datetime(last_update_succeeded_at)
    )


    preview_start = datetime.now()
    preview_data = data_source.load_preview_data(url=url,
                                                 namespace=namespace,
                                                 project_name=project_name,
                                                 branch=branch,
                                                 private_token=private_token)
    preview_end = datetime.now()
    print(preview_data)
    print("Preview data loaded in ", str(preview_end - preview_start))

    last_update_succeeded_at = get_iso_8601_current_time()

    # 3. 데이터를 Opensearch에 저장
    data_source.save_data(
        last_update_succeeded_at=get_iso_8601_current_time(),
        url=url, namespace=namespace, 
        project_name=project_name, 
        branch=branch,
        private_token=private_token)


    print("Data saved successfully")

    # 4. 데이터 임베딩 및 Vectorstore에 추가
    documents = await data_source.read_data()
    splitter = create_splitter(splitter_type=SplitterType.RecursiveCharacterTextSplitter, chunk_size=3000)
    splitted_documents = split_texts(documents, splitter)

    def embed_callback(future):
        print("Embedding task completed")
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

    last_update_succeeded_at = get_iso_8601_current_time()

    embed_task = asyncio.create_task(
        collection.embed_documents_and_overwrite_to_vectorstore(
            documents=splitted_documents,
            last_update_succeeded_at=last_update_succeeded_at
        ))
    embed_task.add_done_callback(embed_callback)
    print("Embedding task started")

    await embed_task

    # 5. 유사도 검색
    print("similarity search started")
    query = "ims input cdr data"
    result = collection.similarity_search(query=query, search_type=Similarity(k=4))

    print("Similarity search result: ", result)

    # 6. Retriever 객체를 이용한 유사도 검색
    query_results = data_source.as_retriever(search_type=Similarity(k=5)).invoke(input=query)

    for result in query_results:
        print(result)

    # 7. Agent 혹은 대화에 연결할 Tool 반환
    tool = data_source.as_retriever_tool(
        search_type=Similarity(k=5),
        name="Very Important tool name",
        description="Very Important Description")

    # 8. 데이터 업데이트
    # since 이후에 커밋된 파일을 가져와서 업데이트 한다.
    # 현재 시각을 last_update_succeeded_at으로 설정해야한다.
    since = last_update_succeeded_at
    last_update_succeeded_at = get_iso_8601_current_time()
    data_source.update_data(
        url=url, 
        namespace=namespace, 
        project_name=project_name, 
        branch=branch,
        private_token=private_token, 
        since=since,
        last_update_succeeded_at=get_iso_8601_current_time())

    # 9. 컬렉션 업데이트
    # collection.last_update_succeeded_at 값이 opensearch 문서 metadata에 저장된다.
    # 그러므로 현재 시각을 last_update_succeeded_at으로 설정해야한다.
    collection.last_update_succeeded_at = iso_8601_str_to_datetime(get_iso_8601_current_time())
    updated_documents = await data_source.read_data(since=since)
    splitter = create_splitter(splitter_type=SplitterType.RecursiveCharacterTextSplitter, chunk_size=3000)
    splitted_update_documents = split_texts(updated_documents, splitter)

    await collection.update(
        documents=splitted_update_documents,
        last_update_succeeded_at=get_iso_8601_current_time(),
        data_source_type=data_source.data_source_type
    )

    # Unclosed client session warning 제거
    await data_source.async_opensearch_client.close()
    await collection.vectorstore.async_client.close()

asyncio.run(main())
