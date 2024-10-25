import os
import re
from datetime import datetime

import pytest
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader

from ai_core.conversation.base import Conversation
from ai_core.data_source.base import DataSourceType, create_data_source
from ai_core.data_source.utils import create_collection_name
from ai_core.data_source.embedding import AzureEmbeddingModelFactory
from ai_core.data_source.splitter import create_splitter, SplitterType
from ai_core.time_utils import str_to_datetime
from ai_core.llm_api_provider import LlmApiProvider
from ai_core.prompt.base import PromptComponent


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_id, smart_bee_model", [("test1", "smart_bee_gpt_4o"),])
async def test_conversation_with_datasource(smart_bee_conversation: Conversation, conversation_id, smart_bee_model,
                                            smart_bee_api_key, smart_bee_api_url):
    def bs4_extractor(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()

    def load_split_fabric_official_documentation() -> list[str]:
        url = "https://learn.microsoft.com/en-us/fabric/get-started"
        loader = RecursiveUrlLoader(
            url=url, max_depth=1, extractor=bs4_extractor,
            # base_url="https://learn.microsoft.com/en-us/fabric"
            base_url="https://learn.microsoft.com/en-us/fabric/get-started"
        )
        docs = loader.load()

        print(f"Loaded {len(docs)} documents from {url}.")

        splitter = create_splitter(
            splitter_type=SplitterType.RecursiveCharacterTextSplitter,
            chunk_size=1000,
            chunk_overlap=0
        )

        split_data = []
        for doc in docs:
            split_data.extend(splitter.split_text(doc.page_content))

        return split_data

    def embed_and_save_to_chromadb(collection, texts):
        print(f"Starting embedding and saving to ChromaDB for {collection.name}.")

        # Calculate elapsed time and print it
        start_time = datetime.now()

        # Caution: Below line can be expensive and takes a long time to run
        collection.embed_documents_and_add_to_chromadb(texts)
        end_time = datetime.now()

        elapsed_time = end_time - start_time
        print(f"Elapsed time for embedding and saving to ChromaDB: {elapsed_time}")

    data_source_name = "ms_fabric_official_doc"
    data_source = create_data_source(data_source_name, "egnarts", """
        Data source containing the official documentation of Microsoft Fabric.
        """, DataSourceType.TEXT)

    split_data = load_split_fabric_official_documentation()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    collection = data_source.add_collection(
        collection_name=create_collection_name(data_source.id, AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_LARGE),
        llm_api_provider=LlmApiProvider.SMART_BEE,
        llm_api_key=smart_bee_api_key,
        llm_embedding_model=AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_LARGE,
        llm_api_url=smart_bee_api_url,
        last_update_succeeded_at=str_to_datetime("2024-06-26 11:09:54"),
        persist_directory=os.path.join(current_dir, "fabric_db")
    )

    # embed_and_save_to_chromadb(collection, split_data)

    prompt = [
        ("system", "데이터 소스를 최대한 활용해서 답변을 주는 AI입니다. 데이터 소스를 검색할 때는 최대한 길게 질의를 만들어서 질의합니다."),
    ]

    p = PromptComponent(name="p", description="d", messages=prompt, input_values={})

    smart_bee_conversation.add_prompt(p)
    smart_bee_conversation.add_datasource(data_source_name, "egnarts", data_source)

    try:
        await smart_bee_conversation.create_agent()

        session_id = "ds_test_session"
        await smart_bee_conversation.clear(session_id)

        # while True:
        #     user = input("User (q/Q to quit): ")
        #     if user in {"q", "Q"}:
        #         print("AI: Byebye")
        #         break
        #
        #     smart_bee_conversation.invoke(session_id, user)
        # messages = smart_bee_conversation.invoke(session_id, "What is microsoft fabric?")
        # for m in messages:
        #     print(m)
        #
        # assert len(messages) == 3
        # assert messages[0].tool_call is not None
    finally:
        await smart_bee_conversation.close_connection_pools()
