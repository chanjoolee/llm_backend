from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import TextLoader

from ai_core.data_source.base import create_data_source
from ai_core.data_source.utils.utils import create_collection_name
from ai_core.data_source.vectorstore.search_type import Similarity

raw_text: str = \
    ("Apache Flink is an open-source stream-processing framework developed by the Apache Software Foundation. "
     "The core of Apache Flink is a distributed streaming data-flow engine written in Java and Scala. "
     "Flink executes arbitrary dataflow programs in a data-parallel and pipelined manner. "
     "Flink's pipelined runtime system enables the execution of bulk/batch and stream processing programs. "
     "Furthermore, Flink's runtime supports the execution of iterative algorithms. "
     "Flink is designed to run in all common cluster environments,"
     " perform computations at in-memory speed and at any scale.")
documents = Document(page_content=raw_text)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs: list[Document] = text_splitter.split_documents([documents])

embeddings = OpenAIEmbeddings(
    openai_api_base="https://aihub-api.sktelecom.com/aihub/v1/sandbox",
    model="text-embedding-3-large",
    openai_api_key="ba3954fe-9cbb-4599-966b-20b04b5d3441")

opensearch_url = "https://localhost:9200"
use_ssl = True
verify_certs = False
http_auth = ("admin", "Skapfhd3122!@")

data_source = create_data_source(
    data_source_name="apache_flink_intro_text",
    created_by="seungbum",
    description="apache flink introduction",
    data_source_type="text")

collection_name = create_collection_name(data_source.id, "text-embedding-3-large")

opensearch = OpenSearchVectorSearch(opensearch_url=opensearch_url,
                                    index_name=collection_name,
                                    embedding_function=embeddings,
                                    use_ssl=use_ssl,
                                    verify_certs=verify_certs,
                                    ssl_show_warn=False,
                                    http_auth=http_auth)

opensearch.add_documents(documents=docs, space_type="cosinesimil")

query = "What did the president say about Ketanji Brown Jackson"

search_type = Similarity(k=1)
opensearch_retriever = opensearch.as_retriever(search_type=search_type.name, search_kwargs=search_type.search_kwargs())

tool = create_retriever_tool(retriever=opensearch_retriever,
                             name="apache_flink_intro_text_retriever",
                             description="This is a data source for the Apache Flink project.")

print(tool.invoke(query))




