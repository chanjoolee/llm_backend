from traceback import print_exc

from langchain_chroma import Chroma
from langchain_core.tools import tool, create_retriever_tool
from langchain_openai import OpenAIEmbeddings

from ai_core.data_source.vectorstore.search_type import Similarity


@tool
def billing_granularify_retriever_tool(query: str):
    '''
    5G SA 도입관련 과금 세부화 내역 조회 시스템 요건 조회

    :param query: search query
    :return: search results
    '''

    embedding_function = OpenAIEmbeddings(
        openai_api_base="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
        model="text-embedding-3-large",
        openai_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441")

    chroma = Chroma(
        collection_name="ds-your_nickname-billing_granularify_system_requirements-te2c71",
        embedding_function=embedding_function,
        persist_directory="/data/daisy/billing_granularify_system_requirements_chromadb")

    search_type = Similarity(k=5)
    retriever = chroma.as_retriever(search_type=search_type.name, search_kwargs=search_type.search_kwargs())

    tool = create_retriever_tool(
        retriever,
        name="billing_granularify_retriever_tool",
        description="5G SA 도입관련 과금 세부화 내역 조회 시스템 요건",
    )

    try:
        return tool.invoke(query)
    except Exception as e:
        print_exc()
        import traceback
        stack_trace = traceback.format_exc()
        return stack_trace

