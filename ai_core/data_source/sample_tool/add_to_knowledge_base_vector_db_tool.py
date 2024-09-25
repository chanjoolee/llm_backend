from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings


@tool
def add_to_knowledge_base_vector_db_tool(content: str):
    '''
    This is a tool for adding data to knowledge base vector DB.

    :param content: content to add
    :param collection_name: chromadb collection name
    '''

    COLLECTION_NAME = "knowledge_base"
    RETRY_COUNT = 3

    def init_chroma():
        embedding_function = OpenAIEmbeddings(
            openai_api_base="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
            model="text-embedding-3-large",
            openai_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441")

        return Chroma(
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME,
            persist_directory="/data/daisy/knowledge_base_chromadb_test")

    def add_texts(content: str, retry_count: int = 0):
        try:
            chroma = init_chroma()
            chroma._client.clear_system_cache()
            chroma.add_texts([content])
        except Exception as e:
            if retry_count < RETRY_COUNT:
                add_texts(content, retry_count + 1)
            else:
                raise e

    try:
        add_texts(content)

        return "Data added successfully"

    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()

        return stack_trace
