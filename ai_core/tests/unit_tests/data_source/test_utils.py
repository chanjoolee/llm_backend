import pytest

from ai_core.data_source.utils.utils import create_data_source_id, create_collection_name
from ai_core.data_source.embedding import EmbeddingModel, AzureEmbeddingModelFactory


def test_create_data_source_id():
    created_by = "nickname_1234567"
    data_source_name = "sk_telecom_datalake_document"
    data_source_id = create_data_source_id(created_by, data_source_name)

    assert data_source_id == f"ds-{created_by}-{data_source_name}"


def test_empty_created_by():
    created_by = ""
    data_source_name = "sk_telecom_datalake"

    with pytest.raises(ValueError) as e:
        create_data_source_id(created_by, data_source_name)


def test_none_data_source_name():
    created_by = "nickname_1234567"
    data_source_name = None

    with pytest.raises(ValueError) as e:
        create_data_source_id(created_by, data_source_name)


def test_create_collection_name():
    data_source_id = "ds-nickname_1234567-sk_telecom_datalake_document"
    embedding_model: EmbeddingModel = AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_LARGE

    collection_name = create_collection_name(data_source_id, embedding_model)

    assert collection_name == "ds-nickname_1234567-sk_telecom_datalake_document-text-embed1712"


def test_empty_data_source_id():
    data_source_id = ""
    embedding_model: EmbeddingModel = AzureEmbeddingModelFactory.TEXT_EMBEDDING_3_LARGE

    with pytest.raises(ValueError) as e:
        create_collection_name(data_source_id, embedding_model)


def test_none_embedding_model():
    data_source_id = "ds-1113593-sk_telecom_datalake_document"
    embedding_model = None

    with pytest.raises(ValueError) as e:
        create_collection_name(data_source_id, embedding_model)
