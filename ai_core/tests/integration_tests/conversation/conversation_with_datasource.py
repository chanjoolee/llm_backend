import re
from datetime import datetime

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader

from ai_core.conversation.base import ConversationFactory
from ai_core.data_source.base import DataSourceType, DataSource
from ai_core.data_source.utils import create_data_source_id, create_collection_name
from ai_core.data_source.embedding import AzureEmbeddingModelFactory
from ai_core.data_source.splitter import create_splitter, SplitterType
from ai_core.time_utils import str_to_datetime
from ai_core.llm_api_provider import LlmApiProvider
from ai_core.prompt.base import PromptComponent


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_split_apache_flink_official_documentation() -> list[str]:
    url = "https://learn.microsoft.com/en-us/fabric/get-started"
    loader = RecursiveUrlLoader(
        url=url, max_depth=20, extractor=bs4_extractor,
        base_url="https://learn.microsoft.com/en-us/fabric"
    )
    docs = loader.load()

    print(f"Loaded {len(docs)} documents from {url}.")

    splitter = create_splitter(
        splitter_type=SplitterType.RecursiveCharacterTextSplitter,
        chunk_size=100,
        chunk_overlap=0
    )

    split_data = []
    for doc in docs:
        split_data.extend(splitter.split_text(doc.page_content))

    return split_data


def embed_and_save_to_chromadb(collection):

    print(f"Starting embedding and saving to ChromaDB for {collection.name}.")

    # Calculate elapsed time and print it
    start_time = datetime.now()

    # Caution: This line is expensive and takes a long time to run
    # collection.embed_documents_and_save_to_chromadb()
    end_time = datetime.now()

    elapsed_time = end_time - start_time
    print(f"Elapsed time for embedding and saving to ChromaDB: {elapsed_time}")


# def test_conversation_with_datasource(skt_conversation: Conversation, skt_api_key: str, skt_api_url: str):
skt_api_key = "ba3954fe-9cbb-4599-966b-20b04b5d3441"
skt_api_url = "https://aihub-api.sktelecom.com/aihub/v1/sandbox"
con_str = 'mysql+pymysql://root:wkdwjdgh7!@localhost:3306/daisy'

data_source_name = "ms_fabric_official_doc"
data_source_type = DataSourceType.TEXT
datasource_data = {
    "id": create_data_source_id("egnarts", data_source_name, data_source_type),
    "name": data_source_name,
    "description": """
    Data source containing the official documentation of Microsoft Fabric. Fabric 
    """,
    "data_source_type": DataSourceType.TEXT
}

data_source = DataSource(**datasource_data)

# split_data = load_split_apache_flink_official_documentation()

data_source.add_collection(
    collection_name=create_collection_name(data_source.id, AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_LARGE),
    llm_api_provider=LlmApiProvider.SMART_BEE,
    llm_api_key=skt_api_key,
    llm_embedding_model=AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_LARGE,
    llm_api_url=skt_api_url,
    last_update_succeeded_at=str_to_datetime("2024-06-26 11:09:54"))

# embed_and_save_to_chromadb(data_source.get_latest_collection())

skt_conversation = ConversationFactory.create_conversation(
    llm_api_provider="openai",
    llm_model="gpt-4o",
    llm_api_key=skt_api_key,
    llm_api_url=skt_api_url,
    temperature=0.2,
    max_tokens=100,
    conversation_id="test",
    history_connection_str=con_str,
    history_table_name="test_session_store",
)

prompt = [
    ("system", "데이터 소스를 최대한 활용해서 답변을 주는 AI입니다. 데이터 소스를 검색할 때는 최대한 길게 질의를 만들어서 질의합니다."),
]

p = PromptComponent(name="p", description="d", messages=prompt, input_values={})

skt_conversation.add_prompt(p)
# skt_conversation.add_datasource(data_source_name, "egnarts", data_source)

skt_conversation.create_agent()

session_id = "ds_test_session"
skt_conversation.clear(session_id)

while True:
    user = input("User (q/Q to quit): ")
    if user in {"q", "Q"}:
        print("AI: Byebye")
        break

    skt_conversation.invoke(session_id, user)
    # print(skt_conversation.invoke(session_id, "플링크?"))
