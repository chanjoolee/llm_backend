from traceback import print_exc

from langchain_chroma import Chroma
from langchain_core.tools import tool, create_retriever_tool
from langchain_openai import OpenAIEmbeddings

from ai_core.data_source.vectorstore.search_type import Similarity


@tool
def knowledge_base_retriever_tool(query: str):
    '''
    This tool retrieves knowledge base.

    :param query: search query
    :return: search results
    '''

    embedding_function = OpenAIEmbeddings(
        openai_api_base="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
        model="text-embedding-3-large",
        openai_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441")

    chroma = Chroma(
        collection_name="knowledge_base",
        embedding_function=embedding_function,
        persist_directory="/data/daisy/knowledge_base_chromadb")

    search_type = Similarity(k=5)
    retriever = chroma.as_retriever(search_type=search_type.name, search_kwargs=search_type.search_kwargs())

    tool = create_retriever_tool(
        retriever,
        name="knowledge_base_retriever_tool",
        description="This is a tool for retrieving knowledge base."
    )

    try:
        return tool.invoke(query)
    except Exception as e:
        print_exc()
        import traceback
        stack_trace = traceback.format_exc()
        return stack_trace
