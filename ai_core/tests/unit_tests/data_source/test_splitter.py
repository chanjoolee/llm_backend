import math
import unittest

from ai_core.data_source.splitter import create_splitter, SplitterType


def test_recursive_character_text_splitter():
    raw_text = "Hello, world!"

    splitter = create_splitter(
        splitter_type=SplitterType.RecursiveCharacterTextSplitter,
        chunk_size=2,
        chunk_overlap=0
    )

    docs = splitter.split_text(raw_text)
    assert(len(docs) == math.ceil(len(raw_text) / 2))


def test_html_header_text_splitter():
    html_string = """
        <!DOCTYPE html>
        <html>
        <body>
            <div>
                <h1>Foo</h1>
                <p>Some intro text about Foo.</p>
                <div>
                    <h2>Bar main section</h2>
                    <p>Some intro text about Bar.</p>
                    <h3>Bar subsection 1</h3>
                    <p>Some text about the first subtopic of Bar.</p>
                    <h3>Bar subsection 2</h3>
                    <p>Some text about the second subtopic of Bar.</p>
                </div>
                <div>
                    <h2>Baz</h2>
                    <p>Some text about Baz</p>
                </div>
                <br>
                <p>Some concluding text about Foo</p>
            </div>
        </body>
        </html>
        """

    splitter = create_splitter(SplitterType.HTMLHeaderTextSplitter, tag="h1")
    splits = splitter.split_text(html_string)

    assert len(splits) == 2

