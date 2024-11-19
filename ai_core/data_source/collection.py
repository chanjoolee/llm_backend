import asyncio
import logging
from datetime import datetime
from typing import Any, Optional, AsyncIterable

from langchain_community.vectorstores import OpenSearchVectorSearch

from ai_core.data_source.embedding import EmbeddingModel, create_llm_embedding_model
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from ai_core.data_source.utils.opensearch_utils import create_opensearch_index_name, aswitch_to_new_index
from ai_core.data_source.utils.utils import split_list_by_length
from ai_core.data_source.vectorstore.search_type import Similarity

logger = logging.getLogger(__name__)


class Collection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    datasource_id: str = Field(str, frozen=True)
    name: str = Field(str, frozen=True)
    llm_api_provider: str
    llm_api_key: str
    llm_api_url: str
    llm_embedding_model_name: str
    vectorstore_hosts: str
    vectorstore_auth: tuple[str, str]
    vectorstore: OpenSearchVectorSearch = None
    llm_embedding_model: EmbeddingModel = None
    last_update_succeeded_at: Optional[datetime] = None

    def __str__(self):
        return (f"Collection: "
                f"name={self.name}, "
                f"datasource_id={self.datasource_id}, "
                f"llm_api_provider={self.llm_api_provider}, "
                f"llm_api_key={self.llm_api_key}, "
                f"llm_api_url={self.llm_api_url}, "
                f"llm_embedding_model_name={self.llm_embedding_model_name}, "
                f"vectorstore_hosts={self.vectorstore_hosts},"
                f"last_update_succeeded_at={self.last_update_succeeded_at}")

    def model_post_init(self, __context: Any) -> None:
        self.llm_embedding_model = (
            create_llm_embedding_model(self.llm_api_provider, self.llm_api_url, self.llm_api_key,
                                       self.llm_embedding_model_name))

        self.vectorstore = OpenSearchVectorSearch(opensearch_url=self.vectorstore_hosts,
                                                  index_name=create_opensearch_index_name(self.name),
                                                  embedding_function=self.llm_embedding_model.embeddings,
                                                  use_ssl=True,
                                                  verify_certs=False,
                                                  ssl_show_warn=False,
                                                  http_auth=self.vectorstore_auth)

    async def cancel_embedding_task(self, embedding_task: asyncio.Task) -> None:
        """
        임베딩 작업을 취소합니다.
        """

        embedding_task.cancel()
        try:
            await embedding_task
        except asyncio.CancelledError:
            logger.info(f"Embedding task of collection {self.name} was cancelled.")

    async def embed_documents_and_add_to_vectorstore(self, documents: AsyncIterable[Document], space_type="cosinesimil") -> (int, int):
        """
        이미 존재하는 collection에 대해 add를 수행한다.

        :param documents: 임베딩 및 vector 저장 대상 데이터와 메타데이터
        :param space_type: 유사도 거리 계산 알고리즘("l2", "l1", "cosinesimil", "linf", "innerproduct")

        :return: 임베딩 후 vectorstore에 저장된 chunk의 개수, 총 chunk 수
        """

        index_alias = self.name

        return await self._batch_add_texts(documents, index_alias, space_type=space_type)

    async def embed_documents_and_overwrite_to_vectorstore(self, documents: AsyncIterable[Document],
                                                           space_type="cosinesimil") -> (int, int):
        """
        이미 존재하는 collection에 대해 overwrite를 수행한다.

        :param documents: 임베딩 및 vector 저장 대상 데이터와 메타데이터
        :param space_type: 유사도 거리 계산 알고리즘("l2", "l1", "cosinesimil", "linf", "innerproduct")

        :return: 임베딩 후 vectorstore에 저장된 chunk의 개수, 총 chunk 수
        """
        client = self.vectorstore.async_client
        index_alias = self.name
        index_name = self.vectorstore.index_name
        prev_index_names = (await client.indices.get(index=f"{index_alias}_*")).keys()

        result: (int, int) = await self._batch_add_texts(documents, index_name, space_type=space_type)

        if result[0] > 0:
            await aswitch_to_new_index(client, index_alias, index_name, prev_index_names)

        return result

    async def _batch_add_texts(self, documents: AsyncIterable[Document], index_name: str, **kwargs) -> (int, int):
        """
        임베딩 호출 횟수를 줄이기 위해 임베딩 모델의 최대 입력 토큰 수를 기준으로 텍스트를 모아서 임베딩을 수행한다.

        :param documents: 임베딩 및 vector 저장 대상 데이터와 메타데이터
        :return: 임베딩 후 vectorstore에 저장된 chunk의 개수, 총 chunk 수
        """

        batches = split_list_by_length(documents, self.llm_embedding_model.max_input_tokens)

        doc_ids = []
        total = 0
        async for batch in batches:
            contents = [doc.content async for doc in batch]
            metadatas = [doc.metadata async for doc in batch]

            total += len(contents)

            logger.info(f"Adding {len(contents)} documents, {self.llm_embedding_model.get_num_tokens(contents)} tokens "
                  f"to vectorstore: {self.name}")

            ids = await self.vectorstore.aadd_texts(texts=contents, metadatas=metadatas, index_name=index_name, **kwargs)

            logger.info(f"Refresh opensearch index: {index_name}")
            await self.vectorstore.async_client.indices.refresh(index=index_name)

            doc_ids.extend(ids)
            if self.llm_embedding_model.request_interval_sec > 0:
                await asyncio.sleep(self.llm_embedding_model.request_interval_sec)

        return len(doc_ids), total

    def similarity_search(self, query: str, search_type: Similarity = Similarity(k=4)) -> list[Document]:
        index_alias = self.name
        return self.vectorstore.similarity_search(query=query, index_name=index_alias, **search_type.search_kwargs)

    def delete_collection(self) -> None:
        logger.info(f"Delete opensearch indices of collection: {self.name}")
        client = self.vectorstore.client
        index_alias = self.name
        index_names = []

        if client.indices.exists_alias(name=index_alias):
            index_names = client.indices.get_alias(name=index_alias).keys()

        for index_name in index_names:
            client.indices.delete(index=index_name)

    async def adelete_collection(self) -> None:
        logger.info(f"Delete opensearch indices of collection: {self.name}")
        client = self.vectorstore.async_client
        index_alias = self.name
        index_names = []

        if await client.indices.exists_alias(name=index_alias):
            index_names = (await client.indices.get_alias(name=index_alias)).keys()

        if index_names:
            for index_name in index_names:
                await client.indices.delete(index=index_name)
