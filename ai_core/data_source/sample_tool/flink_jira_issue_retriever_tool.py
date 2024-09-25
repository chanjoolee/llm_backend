'''
Apache flink 1.13.2 버전을 사용 중인데 최근에 Yarn Master 노드가 장애가 발생해서 다운이 됐어. 그러면서 플링크 앱은 재시작이 되었지.
하지만 문제는 재시작이 되면서 가장 마지막 체크포인트에서 복구가 되지 않고 최초에 시작할 때 지정한 save point에서 시작이 되는 문제가 발생했어.
이와 연관된 flink jira 이슈가 있다면 찾아서 모두 알려줘.
---

'''

from datetime import datetime

from langchain_core.tools import tool

from ai_core.data_source.base import create_data_source, DataSourceType
from ai_core.data_source.vectorstore.search_type import Similarity
from ai_core.llm_api_provider import LlmApiProvider

@tool
def flink_jira_issue_retriever_tool(query: str):
    '''
    This is a tool for retrieving apache flink jira issues from VectorStore

    :param query: search query
    :return: search results
    '''

    # 1. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name="apache_kafka_jira",
        created_by="your_nickname",
        description="test description",
        data_source_type=DataSourceType.JIRA.value)

    # 2. 데이터 소스에 컬렉션 추가
    data_source.add_collection(
        collection_name="apache_flink_jira_collection",
        llm_api_provider=LlmApiProvider.SMART_BEE.value,
        llm_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441",
        llm_api_url="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
        llm_embedding_model_name="text-embedding-3-large",
        persist_directory="/data/flink_jira_chroma_db",
        last_update_succeeded_at=datetime.now())

    tool = data_source.as_retriever_tool(
        name="flink_jira_issue_retriever_tool",
        description="This tool retrieves Apache Flink Jira issues",
        search_type=Similarity(k=10)
    )

    try:
        return tool.invoke(query)
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        return stack_trace


print(flink_jira_issue_retriever_tool("플링크 최근 이슈"))