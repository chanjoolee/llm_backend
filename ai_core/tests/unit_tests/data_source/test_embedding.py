import pytest
from ai_core.data_source.embedding import AzureEmbeddingModelFactory


@pytest.fixture
def data():
    return ["This is a test sentence."]


@pytest.fixture
def ada_002():
    return AzureEmbeddingModelFactory.TEXT_EMBEDDING_ADA_002


@pytest.fixture
def small():
    return AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_SMALL


@pytest.fixture
def large():
    return AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_LARGE


def test_get_num_tokens_for_openai_embedding_models(data, ada_002, small, large):
    """
        세 모델 모두 cl100k_base 인코딩을 사용하여 토큰화합니다.
    """
    assert ada_002.get_num_tokens(data) == small.get_num_tokens(data) == large.get_num_tokens(data) == 6


def test_get_embedding_cost_for_openai_embedding_models(data, ada_002, small, large):
    assert ada_002.get_embedding_cost(data) == 6.000000000000001e-07
    assert small.get_embedding_cost(data) == 1.2e-07
    assert large.get_embedding_cost(data) == 7.8e-07