import asyncio
import logging
from datetime import datetime
from time import sleep
from typing import Any, Optional, Iterable, List

from chromadb.errors import InvalidCollectionException
from langchain_chroma import Chroma

from ai_core import CHROMA_DB_DEFAULT_PERSIST_DIR
from ai_core.data_source.embedding import EmbeddingModel, create_llm_embedding_model
from langchain_core.documents import Document
from chromadb.api import Collection
from pydantic import BaseModel, ConfigDict, Field

from ai_core.data_source.utils import split_list_by_length


logger = logging.getLogger(__name__)


class Collection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    datasource_id: str = Field(str, frozen=True)
    name: str = Field(str, frozen=True)
    llm_api_provider: str
    llm_api_key: str
    llm_api_url: str
    llm_embedding_model_name: str
    llm_embedding_model: EmbeddingModel = None
    collection_metadata: Optional[dict[str, Any]] = None
    chroma: Chroma = None
    persist_directory: str = Field(default=CHROMA_DB_DEFAULT_PERSIST_DIR, frozen=True)
    last_update_succeeded_at: Optional[datetime] = None

    def model_post_init(self, __context: Any) -> None:
        self.llm_embedding_model = (
            create_llm_embedding_model(self.llm_api_provider, self.llm_api_url, self.llm_api_key,
                                       self.llm_embedding_model_name))

        # Chroma 객체가 생성될 때 ChromaDB에 Collection이 생성된다.
        self._init_chroma()

    async def cancel_embedding_task(self, embedding_task: asyncio.Task) -> None:
        '''
        임베딩 작업을 취소합니다.
        '''

        embedding_task.cancel()
        try:
            await embedding_task
        except asyncio.CancelledError:
            logger.info(f"Embedding task of collection {self.name} was cancelled.")

    async def embed_documents_and_add_to_chromadb(self, documents: Iterable[Document]) -> (int, int):
        '''
        이미 존재하는 collection에 대해 add를 수행한다.

        :param documents: 임베딩 및 vector 저장 대상 데이터와 메타데이터
        :return: 임베딩 후 chromadb에 저장된 chunk의 개수, 총 chunk 수
        '''

        return await self._batch_add_texts(documents)

    async def embed_documents_and_overwrite_to_chromadb(self, documents: Iterable[Document]) -> (int, int):
        '''
        이미 존재하는 collection에 대해 overwrite를 수행한다.

        :param documents: 임베딩 및 vector 저장 대상 데이터와 메타데이터
        :return: 임베딩 후 chromadb에 저장된 chunk의 개수, 총 chunk 수
        '''

        if self.exists_collection():
            self.delete_collection()
            self._init_chroma()

        return await self._batch_add_texts(documents)

    async def _batch_add_texts(self, documents: Iterable[Document]) -> (int, int):
        '''
        임베딩 호출 횟수를 줄이기 위해 임베딩 모델의 최대 입력 토큰 수를 기준으로 텍스트를 모아서 임베딩을 수행한다.

        :param documents: 임베딩 및 vector 저장 대상 데이터와 메타데이터
        :return: 임베딩 후 chromadb에 저장된 chunk의 개수, 총 chunk 수
        '''

        batches = split_list_by_length(documents, self.llm_embedding_model.max_input_tokens)

        doc_ids = []
        total = 0
        for batch in batches:
            contents = [doc.content for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            total += len(contents)

            logger.info(f"Adding {len(contents)} documents to ChromaDB collection: {self.name}")

            doc_ids.extend(self.chroma.add_texts(texts=contents, metadatas=metadatas))
            if self.llm_embedding_model.request_interval_sec > 0:
                sleep(self.llm_embedding_model.request_interval_sec)

        return len(doc_ids), total

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return self.chroma.similarity_search(query=query, k=k)

    def get_collection(self) -> Collection:
        try:
            return self.chroma._client.get_collection(name=self.name)
        except InvalidCollectionException:
            return None
        except ValueError:
            return None

    def exists_collection(self) -> bool:
        return bool(self.get_collection())

    def modify_collection_name(self, new_collection_name: str) -> None:
        logger.info(f"Modifying chromadb collection name: {self.name} -> {new_collection_name}")
        self.chroma._chroma_collection.modify(name=new_collection_name)

    def delete_collection(self) -> None:
        logger.info(f"Deleting chromadb collection: {self.name}")
        self.chroma.delete_collection()

    def _create_default_collection_metadata(self):
        return {
            "embedding_model_name": self.llm_embedding_model.name,
            "hnsw:space": "cosine"
        }

    def _init_chroma(self):
        logger.info(f"Initializing ChromaDB for collection: {self.name}")
        self.chroma = Chroma(
            collection_name=self.name,
            embedding_function=self.llm_embedding_model.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata=self._get_merged_with_default_collection_metadata())

    def _get_merged_with_default_collection_metadata(self) -> dict[str, Any]:
        metadata = self._create_default_collection_metadata()
        if self.collection_metadata:
            metadata.update(self.collection_metadata)
        return metadata
