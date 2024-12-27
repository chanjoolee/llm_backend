import asyncio
import logging
from datetime import datetime
from typing import Any, Optional, AsyncIterable

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.vectorstores.opensearch_vector_search import _default_text_mapping

from ai_core.data_source.embedding import EmbeddingModel, create_llm_embedding_model

from pydantic import BaseModel, ConfigDict, Field

from ai_core.data_source.model.document import Document
from ai_core.data_source.utils.opensearch_utils import create_opensearch_index_name, aswitch_to_new_index, \
    adelete_indices_with_alias
from ai_core.data_source.utils.utils import split_list_by_length
from ai_core.data_source.utils.time_utils import iso_8601_str_to_datetime
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

    async def _update(self, documents: AsyncIterable[Document], last_update_succeeded_at: str, space_type="cosinesimil") \
                -> (int, int):

        async def _create_async_iter(document_list: list[Document]) -> AsyncIterable[Document]:
            for doc in document_list:
                yield doc

        async def _delete_documents(documents: AsyncIterable[Document]):
            document_list = []
            doc_ids = set()
            async for doc in documents:
                doc_ids.add(doc.metadata["doc_id"])
                document_list.append(doc)

            query = {"query": {"terms": {"metadata.parent_doc_id.keyword": list(doc_ids) }}}

            logger.info("Execute delete query: ", query)
            await self.vectorstore.async_client.delete_by_query(index=self.name, body=query)

            return document_list

        document_list = await _delete_documents(documents)
        documents = _create_async_iter(document_list)

        return await self.embed_documents_and_add_to_vectorstore(
            documents=documents,
            last_update_succeeded_at=last_update_succeeded_at,
            space_type=space_type)

    async def update(self,
                     documents: AsyncIterable[Document],
                     last_update_succeeded_at: str,
                     data_source_type: str,
                     space_type="cosinesimil") -> (int, int):
        """
        데이터를 업데이트합니다.

        :param documents: 업데이트 대상 데이터와 메타데이터
        :param last_update_succeeded_at: 마지막 성공한 업데이트 시간
        :param data_source_type: 데이터 소스 타입
        :param space_type: 유사도 거리 계산 알고리즘("l2", "l1", "cosinesimil", "linf", "innerproduct")
        :return: 임베딩 후 vectorstore에 저장된 chunk의 개수, 총 chunk 수
        """

        # TODO: GitlabDataSource update 구현
        # TODO: _update 호출했을 때, 문서가 delete, insert 되는 과정에서 문서 다운타임이 생김.
        if data_source_type == "jira":
            return await self._update(documents, last_update_succeeded_at, space_type)
        else:
            return await self.embed_documents_and_overwrite_to_vectorstore(documents, last_update_succeeded_at)


    async def embed_documents_and_add_to_vectorstore(self,
                                                     documents: AsyncIterable[Document],
                                                     last_update_succeeded_at: str,
                                                     space_type="cosinesimil") -> (int, int):
        """
        이미 존재하는 collection에 대해 add를 수행한다.

        :param documents: 임베딩 및 vector 저장 대상 데이터와 메타데이터
        :param index_name: index 이름
        :param last_update_succeeded_at: 마지막 성공한 업데이트 시간
        :param space_type: 유사도 거리 계산 알고리즘("l2", "l1", "cosinesimil", "linf", "innerproduct")

        :return: 임베딩 후 vectorstore에 저장된 chunk의 개수, 총 chunk 수
        """

        index_alias = self.name

        logger.info(f"Start embedding and add documents to vectorstore: {index_alias}")

        result = await self._batch_add_texts(
            documents=documents,
            index_name=index_alias,
            last_update_succeeded_at=last_update_succeeded_at,
            space_type=space_type)

        if result[0] > 0:
            self.last_update_succeeded_at = iso_8601_str_to_datetime(last_update_succeeded_at)

        logger.info(f"Finished embedding and overwrite {result[0]}/{result[1]} documents to vectorstore: {index_alias}")

        return result

    async def embed_documents_and_overwrite_to_vectorstore(self,
                                                           documents: AsyncIterable[Document],
                                                           last_update_succeeded_at: str,
                                                           space_type="cosinesimil") -> (int, int):
        """
        이미 존재하는 collection에 대해 overwrite를 수행한다.

        :param documents: 임베딩 및 vector 저장 대상 데이터와 메타데이터
        :param last_update_succeeded_at: 마지막 성공한 업데이트 시간
        :param space_type: 유사도 거리 계산 알고리즘("l2", "l1", "cosinesimil", "linf", "innerproduct")

        :return: 임베딩 후 vectorstore에 저장된 chunk의 개수, 총 chunk 수
        """
        client = self.vectorstore.async_client
        index_alias = self.name
        index_alias_in_progress = f"{index_alias}_in_progress"
        index_name = self.vectorstore.index_name

        # 이전에 생성한 in-progress index 삭제
        await adelete_indices_with_alias(client, index_alias_in_progress)

        prev_index_names = []
        if await client.indices.exists_alias(name=index_alias):
            prev_index_names = (await client.indices.get_alias(name=index_alias)).keys()

        logger.info(f"Create opensearch index: {index_name}")
        if not await client.indices.exists(index=index_name):
            dim = len(self.llm_embedding_model.embeddings.embed_query("dummy"))
            engine = "nmslib"
            ef_search = 512
            ef_construction = 512
            m = 16
            vector_field = "vector_field"

            mapping = _default_text_mapping(dim, engine, space_type, ef_search, ef_construction, m, vector_field)
            await client.indices.create(index=index_name, body=mapping)
            await client.indices.put_alias(index=index_name, name=index_alias_in_progress)

        logger.info(f"Start embedding and overwrite documents to vectorstore: {index_alias}")
        result: (int, int) = await self._batch_add_texts(documents=documents,
                                                         index_name=index_name,
                                                         last_update_succeeded_at=last_update_succeeded_at,
                                                         space_type=space_type)

        if result[0] > 0:
            await aswitch_to_new_index(client, index_alias, index_alias_in_progress, index_name, prev_index_names)
            self.last_update_succeeded_at = iso_8601_str_to_datetime(last_update_succeeded_at)

        logger.info(f"Finished embedding and overwrite {result[0]}/{result[1]} documents to vectorstore: {index_alias}")

        return result

    async def _batch_add_texts(self, documents: AsyncIterable[Document], index_name: str, last_update_succeeded_at: str,
                               **kwargs) -> (int, int):
        """
        임베딩 호출 횟수를 줄이기 위해 임베딩 모델의 최대 입력 토큰 수를 기준으로 텍스트를 모아서 임베딩을 수행한다.

        :param documents: 임베딩 및 vector 저장 대상 데이터와 메타데이터
        :return: 임베딩 후 vectorstore에 저장된 chunk의 개수, 총 chunk 수
        """

        batches = split_list_by_length(documents, self.llm_embedding_model)

        total = 0
        return_ids = []
        async for batch in batches:
            contents = []
            metadatas = []
            async for doc in batch:
                contents.append(doc.content)
                doc.metadata["last_update_succeeded_at"] = last_update_succeeded_at
                try:
                    doc_id = doc.metadata.pop("doc_id")
                except KeyError:
                    doc_id = None
                doc.metadata["parent_doc_id"] = doc_id

                metadatas.append(doc.metadata)

            total += len(contents)

            logger.info(f"Adding {len(contents)} documents, {self.llm_embedding_model.get_num_tokens(contents)} tokens "
                  f"to vectorstore: {self.name}")

            result_ids = await self.vectorstore.aadd_texts(
                texts=contents, metadatas=metadatas, index_name=index_name, **kwargs)

            return_ids.extend(result_ids)

            logger.info(f"Refresh opensearch index: {index_name}")
            await self.vectorstore.async_client.indices.refresh(index=index_name)

            if self.llm_embedding_model.request_interval_sec > 0:
                await asyncio.sleep(self.llm_embedding_model.request_interval_sec)

        return len(return_ids), total

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
                