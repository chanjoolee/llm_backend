from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, Field

import tiktoken
from langchain_core.embeddings import Embeddings, DeterministicFakeEmbedding
from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings

from ai_core.data_source.embedding.embedding_model_provider import EmbeddingModelProvider
from ai_core.llm_api_provider import LlmApiProvider


class EmbeddingModel(BaseModel):
    llm_api_provider: str
    llm_api_uri: str
    llm_api_key: str
    name: str
    embeddings: Embeddings = None
    embedding_price: float
    max_input_tokens: int
    request_interval_sec: float = Field(default=0.0)
    kwargs: dict = Field(default_factory=dict)

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    @abstractmethod
    def get_num_tokens(self, data: list[str]):
        pass

    @abstractmethod
    def encode(self, data: list[str]):
        pass

    def decode(self, tokens: list[int]):
        pass

    @abstractmethod
    def get_embedding_cost(self, data: list[str]):
        pass


class OpenAIEmbeddingModel(EmbeddingModel):
    def get_num_tokens(self, data: list[str]):
        encoding = tiktoken.encoding_for_model(self.name)
        return sum([len(encoding.encode(d)) for d in data])

    def encode(self, data: str):
        encoding = tiktoken.encoding_for_model(self.name)
        return encoding.encode(data)

    def decode(self, tokens: list[int]):
        encoding = tiktoken.encoding_for_model(self.name)
        return encoding.decode(tokens)

    def get_embedding_cost(self, data: list[str]):
        return self.get_num_tokens(data) * self.embedding_price / 1_000_000


class EmbeddingModelFactory(BaseModel):
    llm_api_provider: str
    llm_api_uri: str
    llm_api_key: str
    supported_models: dict[str, list[Any]]
    kwargs: dict = Field(default_factory=dict)

    def _create_openai_embeddings(self, llm_embedding_model_name: str, **kwargs):
        return OpenAIEmbeddings(
            openai_api_base=self.llm_api_uri,
            model=llm_embedding_model_name,
            openai_api_key=self.llm_api_key,
            **kwargs)

    def get_model(self, llm_embedding_model_name: str, **kwargs) -> EmbeddingModel:
       model_meta = self.supported_models.get(llm_embedding_model_name)
       if model_meta:
           model_provider, embedding_price, max_input_tokens, request_interval_sec = model_meta
           if model_provider == EmbeddingModelProvider.OPENAI:
               return OpenAIEmbeddingModel(llm_api_provider=self.llm_api_provider,
                                           llm_api_uri=self.llm_api_uri,
                                           llm_api_key=self.llm_api_key,
                                           name=llm_embedding_model_name,
                                           embedding_price=embedding_price,
                                           max_input_tokens=max_input_tokens,
                                           request_interval_sec=request_interval_sec,
                                           embeddings=self._create_openai_embeddings(llm_embedding_model_name, **kwargs))
           else:
               raise ValueError(f"Unsupported embedding model provider: {model_provider}")
       else:
           raise ValueError(f"Unsupported embedding model: {llm_embedding_model_name} for {self.llm_api_provider}")


class SmartBeeEmbeddingModelFactory(EmbeddingModelFactory):
    """
    Model Information: https://platform.openai.com/docs/models/embeddings
    OpenAI 임베딩 모델의 이용 가격은 100만 토큰당 가격으로 계산됩니다.
    """

    # model_name: model_provider, embedding_price, max_input_tokens, request_interval_sec
    supported_models: dict[str, list[Any]] = {
                        "text-embedding-3-small": [EmbeddingModelProvider.OPENAI, 0.02, 8191, 1],
                        "text-embedding-3-large": [EmbeddingModelProvider.OPENAI, 0.13, 8191, 1],
                        "text-embedding-ada-002": [EmbeddingModelProvider.OPENAI, 0.10, 8191, 1]}


class AzureEmbeddingModelFactory(EmbeddingModelFactory):
    """
        Model Information: https://platform.openai.com/docs/models/embeddings
        OpenAI 임베딩 모델의 이용 가격은 100만 토큰당 가격으로 계산됩니다.
    """

    # model_name: model_provider, embedding_price, max_input_tokens, request_interval_sec
    supported_models: dict[str, list[Any]] = {
                        "text-embedding-3-small": [EmbeddingModelProvider.OPENAI, 0.02, 8191, 0],
                        "text-embedding-3-large": [EmbeddingModelProvider.OPENAI, 0.13, 8191, 0],
                        "text-embedding-ada-002": [EmbeddingModelProvider.OPENAI, 0.10, 8191, 0]}

    def _create_openai_embeddings(self, llm_embedding_model_name: str, **kwargs):
        return AzureOpenAIEmbeddings(
            azure_endpoint=self.llm_api_url,
            model=llm_embedding_model_name,
            openai_api_key=self.llm_api_key,
            **kwargs)


class FakeEmbeddingModelFactory(EmbeddingModelFactory):
    # model_name: model_provider, embedding_price, max_input_tokens, request_interval_sec
    supported_models: dict[str, list[Any]] = {
                        "text-embedding-3-small": [EmbeddingModelProvider.OPENAI, 0.02, 8191, 0],
                        "text-embedding-3-large": [EmbeddingModelProvider.OPENAI, 0.13, 8191, 0],
                        "text-embedding-ada-002": [EmbeddingModelProvider.OPENAI, 0.10, 8191, 0]}

    def _create_openai_embeddings(self, llm_embedding_model_name: str, **kwargs):
        return DeterministicFakeEmbedding()


class AIOneEmbeddingModelFactory(EmbeddingModelFactory):
    """
    Model Information: https://platform.openai.com/docs/models/embeddings
    OpenAI 임베딩 모델의 이용 가격은 100만 토큰당 가격으로 계산됩니다.
    """

    # model_name: model_provider, embedding_price, max_input_tokens, request_interval_sec
    supported_models: dict[str, list[Any]] = {
                        "text-embedding-3-small": [EmbeddingModelProvider.OPENAI, 0.02, 8191, 0],
                        "text-embedding-3-large": [EmbeddingModelProvider.OPENAI, 0.13, 8191, 0],
                        "text-embedding-ada-002": [EmbeddingModelProvider.OPENAI, 0.10, 8191, 0]}


def create_llm_embedding_model(
        llm_api_provider: str,
        llm_api_uri: str,
        llm_api_key: str,
        llm_embedding_model_name: str,
        **kwargs) -> EmbeddingModel:
    if llm_api_provider == LlmApiProvider.SMART_BEE.value:
        model_factory = SmartBeeEmbeddingModelFactory(
            llm_api_provider=llm_api_provider, llm_api_uri=llm_api_uri, llm_api_key=llm_api_key, **kwargs)
    elif llm_api_provider == LlmApiProvider.AZURE.value:
        model_factory = AzureEmbeddingModelFactory(
            llm_api_provider=llm_api_provider, llm_api_uri=llm_api_uri, llm_api_key=llm_api_key, **kwargs)
    elif llm_api_provider == LlmApiProvider.AI_ONE.value:
        model_factory = AIOneEmbeddingModelFactory(
            llm_api_provider=llm_api_provider, llm_api_uri=llm_api_uri, llm_api_key=llm_api_key,
            check_embedding_ctx_length=False, **kwargs)
    elif llm_api_provider == LlmApiProvider.TEST.value:
        model_factory = FakeEmbeddingModelFactory(
            llm_api_provider=llm_api_provider, llm_api_uri=llm_api_uri, llm_api_key=llm_api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported llm_api_provider: {llm_api_provider}")

    return model_factory.get_model(llm_embedding_model_name)
