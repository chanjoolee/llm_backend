from typing import Iterable

import pytest
from pydantic import ValidationError

from ai_core import CHROMA_DB_TEST_DATA_PATH
from ai_core.data_source.base import DataSourceType, create_data_source
from ai_core.data_source.model.document import Document
from ai_core.data_source.utils import create_collection_name, split_texts, get_first
from ai_core.data_source.embedding import AzureEmbeddingModelFactory
from ai_core.data_source.splitter import create_splitter, SplitterType
from ai_core.time_utils import str_to_datetime
from ai_core.data_source.vectorstore.search_type import Similarity
from ai_core.llm_api_provider import LlmApiProvider


@pytest.fixture
def data_source():
    data_source = create_data_source(
        "daisy_conversation",
        "nickname_123",
        "test description",
        DataSourceType.TEXT.value)

    embedding_model_name = AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_LARGE.name
    llm_api_provider = LlmApiProvider.TEST.value

    data_source.add_collection(
        collection_name=create_collection_name(data_source.id, embedding_model_name),
        llm_api_provider=llm_api_provider,
        llm_api_key="test-api-key",
        llm_embedding_model_name=embedding_model_name,
        llm_api_url="test-api-url",
        collection_metadata={"a": "b"},
        last_update_succeeded_at=str_to_datetime("2024-01-01 16:22:54"))

    return data_source


def _get_splits(chunk_size: int) -> Iterable[Document]:
    with open(CHROMA_DB_TEST_DATA_PATH, "r") as f:
        data = f.read()

    splitter = create_splitter(SplitterType.RecursiveCharacterTextSplitter, chunk_size=chunk_size, chunk_overlap=0)
    documents = [Document(content=data, metadata={})]

    return split_texts(documents, splitter)


@pytest.fixture()
def single_split_data() -> Iterable[Document]:
    return _get_splits(10000)


@pytest.fixture()
def many_splits_data() -> Iterable[Document]:
    return _get_splits(100)


def test_overwrite_to_chromadb(data_source, collection, single_split_data):
    """
        반복적으로 overwrite를 했을 때 검색 결과가 하나만 나와야 한다.
    """

    # llm_api = "test"인 경우 DeterministicFakeEmbedding 반환
    collection.embed_documents_and_overwrite_to_chromadb(single_split_data)
    collection.embed_documents_and_overwrite_to_chromadb(single_split_data)
    collection.embed_documents_and_overwrite_to_chromadb(single_split_data)

    expected: Document = get_first(single_split_data)
    actual = collection.similarity_search(query="test", k=30)
    assert expected.content == actual[0].page_content
    assert len(actual) == 1

    collection.delete_collection()
    assert collection.get_collection() is None


def test_add_to_chromadb(data_source, collection, single_split_data):
    """
        chromaDB에 3번 add 했을 때, chromaDB Collection에 3개의 문서가 들어있어야 한다.
    """

    # llm_api = "test"인 경우 DeterministicFakeEmbedding 반환
    collection.embed_documents_and_add_to_chromadb(single_split_data)
    collection.embed_documents_and_add_to_chromadb(single_split_data)
    collection.embed_documents_and_add_to_chromadb(single_split_data)
    expected = 3

    # chromaDB에서 가져온 값
    actual = collection.get_collection().count()

    assert expected == actual

    collection.delete_collection()
    assert collection.get_collection() is None


def test_as_retriever(data_source):
    """
        데이터소스의 as_retriever를 호출했을 때, 여러 개의 Collection 중에서 가장 최근에 업데이트 성공한 Collection을 사용하는 Retriever를
        반환해야 한다.
    """

    ds = data_source.model_copy()
    ds.add_collection(
              collection_name="data_source_id-text_embedding_3_small",
              llm_api_provider=LlmApiProvider.TEST.value,
              llm_api_key="test-api-key",
              llm_embedding_model_name=AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_SMALL.name,
              llm_api_url="test-api-url",
              last_update_succeeded_at=str_to_datetime("2024-01-01 16:22:54"))
    ds.add_collection(
              collection_name="data_source_id-text_embedding_3_large",
              llm_api_provider=LlmApiProvider.TEST.value,
              llm_api_key="test-api-key",
              llm_embedding_model_name=AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_LARGE.name,
              llm_api_url="test-api-url",
              last_update_succeeded_at=str_to_datetime("2024-06-20 15:05:35"))
    ds.add_collection(
              collection_name="data_source_id-text_embedding_4",
              llm_api_provider=LlmApiProvider.TEST.value,
              llm_api_key="test-api-key",
              llm_embedding_model_name=AzureEmbeddingModelFactory.TEXT_EMBEDDING_ADA_002.name,
              llm_api_url="test-api-url",
              last_update_succeeded_at=str_to_datetime("2024-06-15 15:05:35"))

    # 가장 최근에 업데이트된 collection을 가져옴
    assert ds.as_retriever().metadata["collection_name"] == "data_source_id-text_embedding_3_large"


def test_as_retriever_with_no_last_update_succeeded_at(data_source):
    """
        업데이트 성공한 Collection이 없는 경우 as_retriever를 호출했을 때 ValueError가 발생한다.
    """
    ds = data_source.model_copy()
    ds.collections = {}
    ds.add_collection(
        collection_name="data_source_id-text_embedding_3_small",
        llm_api_provider=LlmApiProvider.TEST.value,
        llm_api_key="test-api-key",
        llm_embedding_model_name=AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_SMALL.name,
        llm_api_url="test-api-url")

    with pytest.raises(ValueError) as e:
        ds.as_retriever()


def test_retriever_query(data_source, collection, single_split_data):
    """
        Retriever를 사용 하여 문서 검색을 했을 때 결과가 잘 나오는 지 테스트 한다.
    """

    expected = single_split_data

    # llm_api = "test"인 경우 DeterministicFakeEmbedding 반환
    collection.embed_documents_and_add_to_chromadb(single_split_data)

    query_results = data_source.as_retriever(Similarity(k=1)).invoke(input="test")
    actual = query_results[0].page_content

    assert get_first(expected).content == actual
    assert len(query_results) == 1

    collection.delete_collection()
    assert collection.get_collection() is None


def test_retriever_tool(data_source, collection, single_split_data):
    """
        Retriever Tool을 사용 하여 문서 검색을 했을 때 결과가 잘 나오는 지 테스트 한다.
    """

    expected = single_split_data

    # llm_api = "test"인 경우 DeterministicFakeEmbedding 반환
    collection.embed_documents_and_add_to_chromadb(single_split_data)

    tool = data_source.as_retriever_tool(name=data_source.name, description=data_source.description)

    actual = tool.invoke(input="test")

    assert get_first(expected) == actual

    collection.delete_collection()
    assert collection.get_collection() is None


def test_metadata(data_source, collection, many_splits_data):
    """
        Collection에 저장된 각 문서들에 split_index라는 메타데이터가 함꼐 저장 되어야 한다.
    """

    # llm_api = "test"인 경우 DeterministicFakeEmbedding 반환
    collection.embed_documents_and_add_to_chromadb(many_splits_data)

    retriever = data_source.as_retriever(Similarity(k=30))
    query_results = retriever.invoke(input="테이블")

    assert query_results
    for r in query_results:
        assert r.metadata.get("split_index")

    collection.delete_collection()
    assert collection.get_collection() is None


def test_collection_metadata(data_source, collection):
    """
        ChromaDB에 Collection이 생성될 때 공통 메타데이터가 저장 되어야 한다.
    """

    collection.delete_collection()
    collection._init_chroma()

    actual_collection = collection.get_collection()

    expected_metadata_keys = ['embedding_model_name', 'hnsw:space']

    for k in expected_metadata_keys:
        assert actual_collection.metadata[k]

    collection.delete_collection()
    assert collection.get_collection() is None


def test_frozen(data_source, collection):
    """
        PyDantic frozen이 설정된 필드 값을 변경하려고 하면 ValidationError가 발생 한다.
    """

    with pytest.raises(ValidationError):
        collection.name = "change"
