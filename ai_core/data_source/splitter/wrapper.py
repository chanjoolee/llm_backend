import json

from langchain_text_splitters import TextSplitter, HTMLHeaderTextSplitter, HTMLSectionSplitter, \
    MarkdownHeaderTextSplitter, RecursiveJsonSplitter

html_header_map = {
    "h1": [("h1", "Header 1")],
    "h2": [("h2", "Header 2")],
    "h3": [("h3", "Header 3")]
}

markdown_header_map = {
    "#": [("#", "Header 1")],
    "##": [("##", "Header 2")],
    "###": [("###", "Header 3")]
}


class HTMLHeaderTextSplitterWrapper(TextSplitter):
    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag
        self.headers_to_split_on = html_header_map.get(tag, [("h1", "Header 1")])
        self.splitter = HTMLHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

    def split_text(self, text: str):
         return [document.page_content for document in self.splitter.split_text(text)]


class HTMLSectionSplitterWrapper(TextSplitter):
    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag
        self.headers_to_split_on = html_header_map.get(tag, [("h1", "Header 1")])
        self.splitter = HTMLSectionSplitter(headers_to_split_on=self.headers_to_split_on)

    def split_text(self, text: str):
        return [document.page_content for document in self.splitter.split_text(text)]


class MarkdownHeaderTextSplitterWrapper(TextSplitter):
    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag
        self.headers_to_split_on = markdown_header_map.get(tag, [("#", "Header 1")])
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

    def split_text(self, text: str):
        return [document.page_content for document in self.splitter.split_text(text)]


class RecursiveJsonSplitterWrapper(TextSplitter):
    def __init__(self, max_chunk_size: int):
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)

    def split_text(self, text: str):
        json_data = json.loads(text)
        return self.splitter.split_text(json_data=json_data)
