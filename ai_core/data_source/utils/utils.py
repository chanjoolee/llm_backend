import hashlib
import re
from typing import Iterable
import unicodedata

from bs4 import BeautifulSoup
from langchain_text_splitters import TextSplitter

from ai_core.data_source.model.document import Document

COLLECTION_NAME_MAX_LENGTH = 63
DATA_SOURCE_NAME_MAX_LENGTH = 40
CREATED_BY_MAX_LENGTH = 16
DATA_SOURCE_ID_MAX_LENGTH = DATA_SOURCE_NAME_MAX_LENGTH + CREATED_BY_MAX_LENGTH + 1


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
    if len(data_source_name) > DATA_SOURCE_NAME_MAX_LENGTH:
        raise ValueError(f"data_source_name must be less than {DATA_SOURCE_NAME_MAX_LENGTH} characters")
    if len(created_by) > CREATED_BY_MAX_LENGTH:
        raise ValueError(f"created_by must be less than {CREATED_BY_MAX_LENGTH} characters")

    return "-".join(["ds", created_by, data_source_name])


def create_collection_name(data_source_id: str, embedding_model_name: str) -> str:

    # Chromadb Collection Naming Rule
    #   (1) contains 3-63 characters
    #   (2) starts and ends with an alphanumeric character
    #   (3) otherwise contains only alphanumeric characters, underscores or hyphens (-)
    #   (4) contains no two consecutive periods (..)

    if not data_source_id:
        raise ValueError("data_source_id must be provided")
    if not embedding_model_name:
        raise ValueError("embedding_model must be provided")
    if len(data_source_id) > DATA_SOURCE_ID_MAX_LENGTH:
        raise ValueError(f"data_source_id must be less than {DATA_SOURCE_ID_MAX_LENGTH} characters")

    return truncate_name("-".join([data_source_id, embedding_model_name]))


def truncate_name(s: str, max_length: int = COLLECTION_NAME_MAX_LENGTH) -> str:
    if len(s) <= max_length:
        return s

    hash_obj = hashlib.md5(s.encode())
    hash_str = hash_obj.hexdigest()[:4]

    new_max_length = max_length - 4
    truncated = s[:new_max_length]

    return f"{truncated}{hash_str}"


def truncate_content(s: str, max_length: int) -> str:
    return s[:max_length]


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def reduce_new_line(s: str) -> str:
    return s.replace("\n\n", "\n").strip()


def split_list_by_length(input_list: Iterable[Document], max_length: int) -> Iterable[Iterable[Document]]:
    current_sublist = []
    current_length = 0

    for item in input_list:
        item_length = len(item.content)

        if current_length + item_length > max_length:
            if current_sublist:
                yield (x for x in current_sublist)

            current_sublist = [item]
            current_length = item_length
        else:
            current_sublist.append(item)
            current_length += item_length

    if current_sublist:
        yield (x for x in current_sublist)


def safe_get(data: dict, key: str, default_value=None):
    try:
        return data.get(key, default_value)
    except AttributeError:
        return default_value


def split_texts(documents: Iterable[Document], splitter: TextSplitter) -> Iterable[Document]:
    split_index = 0
    for document in documents:
        for split in splitter.split_text(document.content):
            split_index += 1
            metadata = document.metadata if document.metadata else {}
            metadata["split_index"] = split_index
            yield Document(content=split, metadata=metadata)



def get_first(iterable: Iterable):
    for item in iterable:
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
