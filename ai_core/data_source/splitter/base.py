from enum import Enum

from langchain_text_splitters import (
    TextSplitter, RecursiveCharacterTextSplitter, CharacterTextSplitter, Language)

from ai_core.data_source.splitter.wrapper import HTMLHeaderTextSplitterWrapper, HTMLSectionSplitterWrapper, \
    MarkdownHeaderTextSplitterWrapper, RecursiveJsonSplitterWrapper


class SplitterType(Enum):
    RecursiveCharacterTextSplitter = 'RecursiveCharacterTextSplitter'
    CharacterTextSplitter = 'CharacterTextSplitter'
    HTMLHeaderTextSplitter = 'HTMLHeaderTextSplitter'
    HTMLSectionSplitter = 'HTMLSectionSplitter'
    MarkdownHeaderTextSplitter = 'MarkdownHeaderTextSplitter'
    RecursiveJsonSplitter = 'RecursiveJsonSplitter'
    SimillarSentenceSplitter = 'SimillarSentenceSplitter'



def create_splitter(splitter_type: SplitterType, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs) \
        -> TextSplitter:
    """Create splitter based on the type.

    :param SplitterType splitter_type: The type of splitter to get.
    :param int chunk_size: The size of the chunks to split the text into.
    :param int chunk_overlap: The overlap between chunks.
    :param Language language: The language of the text.
    """

    if splitter_type == SplitterType.RecursiveCharacterTextSplitter:
        """
            document가 한 개라면 split_text를 사용하고, 한 개 이상이라면 split_documents를 사용합니다.
            사용법 참고: https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html#langchain_text_splitters.character.RecursiveCharacterTextSplitter
        """

        splitter = RecursiveCharacterTextSplitter
        language = kwargs.get("language", None)
        if language:
            splitter = splitter.from_language(language)

        return splitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif splitter_type == SplitterType.CharacterTextSplitter:
        separator = kwargs.get("separator", "\n\n")
        is_separator_regex = kwargs.get("is_separator_regex", False)

        return CharacterTextSplitter(
            separator=separator,
            is_separator_regex=is_separator_regex,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    elif splitter_type == SplitterType.HTMLHeaderTextSplitter:
        tag = kwargs.get("tag", "h1")
        return HTMLHeaderTextSplitterWrapper(tag=tag)

    elif splitter_type == SplitterType.HTMLSectionSplitter:
        tag = kwargs.get("tag", "h1")
        return HTMLSectionSplitterWrapper(tag=tag)

    elif splitter_type == SplitterType.MarkdownHeaderTextSplitter:
        tag = kwargs.get("tag", "#")
        return MarkdownHeaderTextSplitterWrapper(tag=tag)

    elif splitter_type == SplitterType.RecursiveJsonSplitter:
        max_chunk_size = kwargs.get("max_chunk_size", 2000)
        return RecursiveJsonSplitterWrapper(max_chunk_size=max_chunk_size)

    elif splitter_type == SplitterType.SimillarSentenceSplitter:
        pass
    else:
        raise ValueError('Invalid splitter type: {}'.format(splitter_type))
