import sys
import time
from typing import Iterable, AsyncIterable

from opensearchpy import AsyncOpenSearch, OpenSearch
from pydantic import BaseModel, Field

from ai_core.data_source.model.document import Document


def create_opensearch_index_name(prefix: str) -> str:
    return f"{prefix}_{str(int(time.time()))}"


def switch_to_new_index(opensearch_client: OpenSearch, index_alias: str, index_alias_in_progress: str, index_name: str,
                        prev_index_names: list[str]) -> None:
    """
    새로운 index로 alias를 변경하고, 기존 index를 삭제한다.

    :param opensearch_client: OpenSearch client
    :param index_alias: alias 이름
    :param index_alias_in_progress: in progress alias 이름
    :param index_name: 새로운 index 이름
    :param prev_index_names: 이전 index 이름들
    """

    if prev_index_names:
        for prev_index_name in prev_index_names:
            # update alias atomically
            opensearch_client.indices.update_aliases(
                body={"actions": [
                    {"remove": {"index": index_name, "alias": index_alias_in_progress}},
                    {"remove": {"index": prev_index_name, "alias": index_alias}},
                    {"add": {"index": index_name, "alias": index_alias, "is_write_index": True}}]})

            # 기존 index 삭제
            if prev_index_name != index_name and opensearch_client.indices.exists(index=prev_index_name):
                opensearch_client.indices.delete(index=prev_index_name)
    else:
        opensearch_client.indices.update_aliases(
            body={"actions": [
                {"remove": {"index": index_name, "alias": index_alias_in_progress}},
                {"add": {"index": index_name, "alias": index_alias, "is_write_index": True}}]})


async def aswitch_to_new_index(opensearch_client: AsyncOpenSearch, index_alias: str, index_alias_in_progress: str,
                               index_name: str, prev_index_names: list[str]) -> None:
    """
    새로운 index로 alias를 변경하고, 기존 index를 삭제한다.

    :param opensearch_client: OpenSearch client
    :param index_alias: alias 이름
    :param index_alias_in_progress: in progress alias 이름
    :param index_name: 새로운 index 이름
    :param prev_index_names: 이전 index 이름들
    """

    if prev_index_names:
        for prev_index_name in prev_index_names:
            # update alias atomically
            await opensearch_client.indices.update_aliases(
                body={"actions": [
                    {"remove": {"index": index_name, "alias": index_alias_in_progress}},
                    {"remove": {"index": prev_index_name, "alias": index_alias}},
                    {"add": {"index": index_name, "alias": index_alias}}]})

            # 기존 index 삭제
            if prev_index_name != index_name and (await opensearch_client.indices.exists(index=prev_index_name)):
                await opensearch_client.indices.delete(index=prev_index_name)
    else:
        await opensearch_client.indices.update_aliases(
            body={"actions": [
                {"remove": {"index": index_name, "alias": index_alias_in_progress}},
                {"add": {"index": index_name, "alias": index_alias, "is_write_index": True}}]})


class IndexingStatistics(BaseModel):
    doc_count: int = Field(default=0)
    doc_total_bytes: int = Field(default=0)
    doc_max_bytes: int = Field(default=0)
    doc_min_bytes: int = Field(default=sys.maxsize)
    doc_avg_bytes: int = Field(default=0)

    def increment(self, doc_bytes: int):
        self.doc_count += 1
        self.doc_total_bytes += doc_bytes
        self.doc_max_bytes = max(self.doc_max_bytes, doc_bytes)
        self.doc_min_bytes = min(self.doc_min_bytes, doc_bytes)
        if self.doc_count != 0:
            self.doc_avg_bytes = int(self.doc_total_bytes / self.doc_count)

    def decrement(self, doc_bytes: int):
        self.doc_count -= 1
        self.doc_total_bytes -= doc_bytes
        if self.doc_count != 0:
            self.doc_avg_bytes = int(self.doc_total_bytes / self.doc_count)

    def values(self) -> dict:
        if self.doc_count == 0:
            self.doc_min_bytes = 0

        return self.model_dump()


def get_documents(opensearch_client: OpenSearch, index_alias: str, num_docs: int) -> list[dict]:
    """
    index_alias로부터 num_docs 개의 documents를 가져온다.

    :param opensearch_client: OpenSearch client
    :param index_alias: alias 이름
    :param num_docs: 가져올 document 개수
    :return: documents
    """
    return opensearch_client.search(index=index_alias, size=num_docs).get("hits", {}).get("hits", [])


async def aget_documents(async_opensearch_client: AsyncOpenSearch, index_alias: str, num_docs: int) -> list[dict]:
    """
    index_alias로부터 num_docs 개의 documents를 가져온다.

    :param opensearch_client: OpenSearch client
    :param index_alias: alias 이름
    :param num_docs: 가져올 document 개수
    :return: documents
    """

    return (await async_opensearch_client.search(index=index_alias, size=num_docs)).get("hits", {}).get("hits", [])


def search(opensearch_client: OpenSearch, index_name: str, query: dict, size: int = 1000,
           sort: list[dict] = None) -> Iterable[Document]:
    """
    OpenSearch에서 document를 검색한다.
    :param opensearch_client: OpenSearch client
    :param index_name: index 이름
    :param query: 검색 쿼리
    :param sort: 정렬
    :param size: 검색 결과 개수
    :return: documents
    """

    retrieved_size = 1000
    if not sort:
        sort = [{"_id": "asc"}]

    body = {
        "query": query,
        "sort": sort,
        "size": size
    }

    print(body)

    while retrieved_size == 1000:
        response = opensearch_client.search(
            index=index_name,
            body=body
        )

        hits = response['hits']['hits']
        retrieved_size = len(hits)

        if len(hits) > 0:
            last_sort = hits[-1]['sort']
            body["search_after"] = last_sort

        for hit in hits:
            yield Document(**hit["_source"])


async def asearch(async_opensearch_client: AsyncOpenSearch, index_name: str, query: dict,
                  size: int = 1000, sort: list[dict] = None) -> AsyncIterable[Document]:
    """
    OpenSearch에서 document를 검색한다.
    :param async_opensearch_client: OpenSearch async client
    :param index_name: index 이름
    :param query: 검색 쿼리
    :param sort: 정렬
    :param size: 검색 결과 개수
    :return: documents
    """

    retrieved_size = 1000
    if not sort:
        sort = [{"_id": "asc"}]

    body = {
        "query": query,
        "sort": sort,
        "size": size
    }

    while retrieved_size == 1000:
        response = await async_opensearch_client.search(
            index=index_name,
            body=body
        )

        hits = response['hits']['hits']
        retrieved_size = len(hits)

        if len(hits) > 0:
            last_sort = hits[-1]['sort']
            body["search_after"] = last_sort

        for hit in hits:
            yield Document(**hit["_source"])


def delete_indices_with_alias(opensearch_client: OpenSearch, index_alias: str):
    """
    alias와 관련된 모든 index를 삭제한다.

    :param opensearch_client: OpenSearch client
    :param index_alias: alias 이름
    """

    if opensearch_client.indices.exists_alias(name=index_alias):
        indices = opensearch_client.indices.get_alias(name=index_alias).keys()
        for index in indices:
            opensearch_client.indices.delete(index=index)


async def adelete_indices_with_alias(async_opensearch_client: AsyncOpenSearch, index_alias: str):
    """
    alias와 관련된 모든 index를 삭제한다.

    :param async_opensearch_client: OpenSearch async client
    :param index_alias: alias 이름
    """

    if await async_opensearch_client.indices.exists_alias(name=index_alias):
        indices = (await async_opensearch_client.indices.get_alias(name=index_alias)).keys()
        for index in indices:
            await async_opensearch_client.indices.delete(index=index)


async def adelete_by_query(async_opensearch_client: AsyncOpenSearch, index_name: str, query: dict, **kwargs):
    return await async_opensearch_client.delete_by_query(
        index=index_name,
        body=query,
        **kwargs
    )
