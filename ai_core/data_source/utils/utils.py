import re
import os

import psutil
from typing import Iterable, AsyncIterable, AsyncGenerator
import unicodedata

from bs4 import BeautifulSoup
from langchain_text_splitters import TextSplitter

from ai_core.data_source.embedding import EmbeddingModel
from ai_core.data_source.model.document import Document


DATA_SOURCE_ID_PREFIX = "ds"
COLLECTION_NAME_PREFIX = "co"


def create_data_source_id(created_by: str, data_source_name: str) -> str:
    """
    데이터 소스의 아이디를 생성합니다.

    :param created_by: 사용자의 닉네임
    :param data_source_name: 데이터 소스 이름
    :return: 유니크한 데이터 소스의 아이디
    """

    if not created_by:
        raise ValueError("created_by must be provided")
    if not data_source_name:
        raise ValueError("data_source_name must be provided")

    return "-".join([DATA_SOURCE_ID_PREFIX, created_by.lower(), data_source_name.lower()])


def create_collection_name(data_source_id: str, embedding_model_name: str) -> str:
    return "-".join([COLLECTION_NAME_PREFIX, data_source_id, embedding_model_name])


def truncate_content(s: str, max_length: int) -> str:
    return s[:max_length]


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def reduce_new_line(s: str) -> str:
    return s.replace("\n\n", "\n").strip()


async def split_list_by_length(input_list: AsyncIterable[Document], embedding_model: EmbeddingModel) \
        -> AsyncIterable[AsyncIterable[Document]]:
    max_tokens = embedding_model.max_input_tokens
    current_sublist = []
    current_tokens = 0

    async def async_generator(sublist: list[Document]) -> AsyncGenerator[Document, None]:
        for doc in sublist:
            yield doc

    async for item in input_list:
        item_content_tokens = embedding_model.get_num_tokens([item.content])

        if current_tokens + item_content_tokens > max_tokens and current_sublist:
            yield async_generator(current_sublist)

            current_sublist = []
            current_tokens = 0

            current_sublist.append(item)
            current_tokens = item_content_tokens
        else:
            current_sublist.append(item)
            current_tokens += item_content_tokens

    if current_sublist:
        yield async_generator(current_sublist)


def safe_get(data: dict, key: str, default_value=None):
    try:
        return data.get(key, default_value)
    except AttributeError:
        return default_value


async def split_texts(documents: AsyncIterable[Document], splitter: TextSplitter) -> AsyncIterable[Document]:
    async for document in documents:
        split_index = 0
        for split in splitter.split_text(document.content):
            metadata = document.metadata.copy() if document.metadata else {}
            metadata["split_index"] = split_index
            split_index += 1
            yield Document(content=split, metadata=metadata)


def get_first(iterable: Iterable):
    for item in iterable:
        return item


async def aget_first(iterable: AsyncIterable):
    async for item in iterable:
        return item


def remove_control_characters(s):
    s = s.replace("\xa0", "")
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C").strip()


def clean_json_string(json_string):
    if json_string:
        json_string = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_string)
        json_string = re.sub(r'(?<!\\)"', r'\\"', json_string)
        json_string = re.sub(r'[\n\r\t]', '', json_string)

    return json_string


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def list_to_async_generator(lst: list):
    async def async_generator():
        for item in lst:
            yield item

    return async_generator()
