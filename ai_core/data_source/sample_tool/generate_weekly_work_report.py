import json
from datetime import datetime

from langchain_core.tools import tool

from ai_core.data_source.base import create_data_source, DataSourceType
from ai_core.data_source.embedding import SmartBeeEmbeddingModelFactory
from ai_core.data_source.utils import create_collection_name
from ai_core.llm_api_provider import LlmApiProvider


@tool
def get_weekly_tde_jira_issues_tool(jira_assignee_name: str):
    '''
    This is a tool for get weekly tde jira issues.
    param jira_assignee: Jira assignee name, e.g. "1113593"
    '''

    data_source = create_data_source(
        data_source_name="tde_jira",
        created_by="dt-seungbum",
        description="test description",
        data_source_type=DataSourceType.JIRA.value)

    collection = data_source.add_collection(
        collection_name=create_collection_name(data_source.id, SmartBeeEmbeddingModelFactory.TEXT_EMBEDDING_3_LARGE),
        llm_api_provider=LlmApiProvider.SMART_BEE.value,
        llm_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441",
        llm_api_url="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
        llm_embedding_model_name="text-embedding-3-large",
        persist_directory="/data/daisy/tde_jira_chromadb")

    query = "issue"

    filter_date_epoch = int(datetime.strptime("2024-08-21T00:00:00Z", "%Y-%m-%dT%H:%M:%S%z").timestamp())
    filter = {"$and": [{"assignee": jira_assignee_name}, {"last_updated": {"$gte": filter_date_epoch}}]}
    print(f"Filtering with {json.dumps(filter)}")
    results = collection.chroma.similarity_search(query=query, filter=filter, k=10)

    return "==============ISSUE SEPARATOR==============\n".join([result.page_content for result in results])
