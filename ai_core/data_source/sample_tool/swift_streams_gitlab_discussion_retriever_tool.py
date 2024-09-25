from traceback import print_exc

from langchain_chroma import Chroma
from langchain_core.tools import tool, create_retriever_tool
from langchain_openai import OpenAIEmbeddings

from ai_core.data_source.vectorstore.search_type import Similarity
from ai_core.llm_api_provider import LlmApiProvider


@tool
def swift_streams_gitlab_discussion_retriever_tool(query: str):
    '''
    This is a tool for retrieving Swift Streams project's discussions(code reviews) from vector store.

    :param query: search query
    :return: search results
    '''

    embedding_function = OpenAIEmbeddings(
        openai_api_base="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
        model="text-embedding-3-large",
        openai_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441")

    chroma = Chroma(
        collection_name="ds-dtseungbum-swift_streams_develop-text-embedding-3-large",
        embedding_function=embedding_function,
        persist_directory="/data/daisy/swift_streams_gitlab_discussion_chromadb")

    search_type = Similarity(k=5)
    retriever = chroma.as_retriever(search_type=search_type.name, search_kwargs=search_type.search_kwargs())

    tool = create_retriever_tool(
        retriever,
        name="swift_streams_gitlab_discussion_retriever_tool",
        description="This is a tool for retrieving Swift Streams project's discussions(code reviews) from vector store."
                    "Swift Streams is a project that provides a platform for streaming data processing in SK Telecom."
                    "The project is developed by the Data Technology team at SK Telecom.",
    )

    try:
        return tool.invoke(query)
    except Exception as e:
        print_exc()
        import traceback
        stack_trace = traceback.format_exc()
        return stack_trace
