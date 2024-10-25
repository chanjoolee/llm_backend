import asyncio
import io
import json
import logging
import random
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from time import sleep
from typing import Dict, Optional, Any, Callable, Iterable
import os
import shutil
import gitlab
import base64
import glob

import langchain_core.documents
from atlassian import Jira, Confluence
from gitlab.v4.objects import Project, ProjectFile
from docx.api import Document as DocxDocument

from ai_core import DATA_SOURCE_SAVE_BASE_DIR, CHROMA_DB_DEFAULT_PERSIST_DIR
from ai_core.base import ComponentType, create_tool_name
from ai_core.data_source.model.document import Document, DocumentEncoder
from ai_core.data_source.utils.utils import create_data_source_id, bs4_extractor, truncate_content, get_first
from ai_core.data_source.collection import Collection
from ai_core.data_source.utils.jira_utils import filter_required_fields, cleanse_fields
from ai_core.data_source.vectorstore.search_type import Similarity
from ai_core.time_utils import datetime_to_str

from langchain_core.retrievers import BaseRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain_community.document_loaders import RecursiveUrlLoader, UnstructuredPDFLoader, UnstructuredFileLoader

from pydantic import BaseModel, Field



logger = logging.getLogger(__name__)



class DataSourceFileLockException(Exception):
    pass


class DataSourceType(Enum):
    TEXT = "text"
    PDF_FILE = "pdf_file"
    CONFLUENCE = "confluence"
    GITLAB = "gitlab"
    GITLAB_DISCUSSION = "gitlab_discussion"
    URL = "url"
    DOC_FILE = "doc_file"
    JIRA = "jira"


class DataSource(BaseModel):
    id: str = Field(str, frozen=True)
    name: str = Field(str, frozen=True)
    description: str
    data_source_type: DataSourceType
    collections: Dict[str, Collection] = Field(default_factory=dict)
    data_dir_path: str = None
    preview_data: str = Field(str)

    SAVE_DIR_BUCKET_SIZE: int = 10000
    PREVIEW_DATA_MAX_LENGTH: int = 10000

    def model_post_init(self, __context: Any) -> None:
        self.data_dir_path = self._get_data_source_save_dir()
        self.read_preview_data()

    def add_collection(self, collection_name: str, llm_api_provider: str, llm_api_key: str, llm_api_url: str,
                       llm_embedding_model_name: str, collection_metadata: Optional[dict[str, str]] = None,
                       last_update_succeeded_at: Optional[datetime] = None,
                       persist_directory: str = CHROMA_DB_DEFAULT_PERSIST_DIR):

        collection = Collection(
            datasource_id=self.id,
            name=collection_name,
            llm_api_provider=llm_api_provider,
            llm_api_key=llm_api_key,
            llm_api_url=llm_api_url,
            llm_embedding_model_name=llm_embedding_model_name,
            collection_metadata=collection_metadata,
            last_update_succeeded_at=last_update_succeeded_at,
            persist_directory=persist_directory
        )

        self.collections[collection.name] = collection

        return collection

    def get_latest_collection(self) -> Collection:
        sorted_collections = (
            sorted(
                filter(lambda c: c.last_update_succeeded_at is not None, self.collections.values()),
                key=lambda collection: collection.last_update_succeeded_at))

        if sorted_collections:
            return sorted_collections[-1]
        else:
            logger.warning(f"No collection found with last_updated_succeeded_at for data source {self.id}")

            return None

    def as_retriever(self, search_type=Similarity(k=4)) -> BaseRetriever:
        latest_collection = self.get_latest_collection()

        '''
            Retriever의 메타데이터 입니다. 
            특정 instance를 식별하기 위한 용도 등으로 사용합니다.
        '''
        metadata = {
            "data_source_id": latest_collection.datasource_id,
            "collection_name": latest_collection.name,
            "llm_api_provider": latest_collection.llm_api_provider,
            "llm_api_url": latest_collection.llm_api_url,
            "llm_embedding_model": latest_collection.llm_embedding_model.name,
            "created_at": datetime_to_str(datetime.now())
        }

        return latest_collection.chroma.as_retriever(
            search_type=search_type.name, metadata=metadata, search_kwargs=search_type.search_kwargs())

    def as_retriever_tool(self, name: str, description: str, search_type=Similarity(k=4)) -> Tool:
        if not name:
            raise ValueError("name must be provided")
        if not description:
            raise ValueError("description must be provided")

        return create_retriever_tool(
            self.as_retriever(search_type=search_type),
            name=name,
            description=description
        )

    @abstractmethod
    def _load_data(self, **kwargs) -> Iterable[Document]:
        pass

    @abstractmethod
    def load_preview_data(self, **kwargs) -> str:
        pass

    async def cancel_save_data_task(self, save_task: asyncio.Task) -> None:
        '''
        데이터 저장 작업을 취소합니다.
        '''

        save_task.cancel()
        try:
            await save_task
        except asyncio.CancelledError:
            logger.info(f"Save task of datasource {self.id} was cancelled.")

    async def save_data(self, **kwargs) -> None:
        '''
        docx, pdf file, confluence 등 다양한 소스로부터 불러온 텍스트 데이터를 파일로 저장합니다.

        :param kwargs: DataSource의 구현체마다 각각 다른 파라메터를 전달 받습니다.
        :return:
        '''

        lock_file_path = self.data_dir_path + "/.lock"
        inprogress_dir = self.data_dir_path + ".inprogress"

        if os.path.exists(lock_file_path):
            logger.warning(f"Request to save data is ignored. {self.data_dir_path} is in use.")
            raise DataSourceFileLockException(f"{self.data_dir_path} is in use.")

        try:
            os.makedirs(self.data_dir_path, exist_ok=True)

            with open(lock_file_path, "w") as f:
                f.write("")

            if inprogress_dir:
                self.delete_inprogress_directory()

            os.makedirs(inprogress_dir)

            documents: Iterable[Document] = self._load_data(**kwargs)

            # 한 디렉토리에 너무 많은 파일이 저장되는 것을 방지하기 위해 bucket 단위로 저장합니다.
            bucket = 0

            for i, document in enumerate(documents):
                if i % self.SAVE_DIR_BUCKET_SIZE == 0:
                    bucket = i // self.SAVE_DIR_BUCKET_SIZE
                    os.makedirs(inprogress_dir + f"/{bucket}", exist_ok=True)

                with open(inprogress_dir + f"/{bucket}/data{i}.txt", "w", encoding="utf-8") as f:
                    f.write(json.dumps(document, cls=DocumentEncoder, indent=2, ensure_ascii=False))

            self.delete_data_directory()
            os.rename(inprogress_dir, self.data_dir_path)

        finally:
            if os.path.exists(inprogress_dir):
                self.delete_inprogress_directory()
            if os.path.exists(lock_file_path):
                os.remove(lock_file_path)

    def read_data(self) -> Iterable[Document]:
        # 텍스트 파일로 저장된 데이터를 불러옵니다.
        file_paths = glob.glob(self.data_dir_path + "/**/data*.txt")

        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()
                json_data = json.loads(data)
                yield Document(content=json_data["content"], metadata=json_data["metadata"])

    def read_preview_data(self) -> str:
        # 이미 저장된 텍스트 파일이 있다면 불러옵니다.
        # preview data는 데이터소스 화면에 보여주기 위한 용도로 사용합니다.

        first_file = self.data_dir_path + "/0/data0.txt"

        if os.path.isfile(first_file):
            with open(first_file, "r", encoding="utf-8") as f:
                content = f.read()
                if len(content) > self.PREVIEW_DATA_MAX_LENGTH:
                    content = content[:self.PREVIEW_DATA_MAX_LENGTH]
                    self.preview_data = content

        return self.preview_data

    def delete_data_directory(self):
        if os.path.exists(self.data_dir_path):
            logger.info(f"Delete data directory: {self.data_dir_path}")
            shutil.rmtree(self.data_dir_path)

    def delete_inprogress_directory(self):
        if os.path.exists(self.data_dir_path + ".inprogress"):
            logger.info(f"Deleting inprogress directory: {self.data_dir_path}.inprogress")
            shutil.rmtree(self.data_dir_path + ".inprogress")

    def _get_data_source_save_dir(self) -> str:
        return DATA_SOURCE_SAVE_BASE_DIR + f"/{self.id}"


class TextDataSource(DataSource):
    def _load_data(self, raw_text: Iterable[str]) -> Iterable[Document]:
        for content in raw_text:
            yield Document(content=content, metadata={})

    def load_preview_data(self, raw_text: Iterable[str]) -> str:
        return truncate_content(get_first(raw_text), self.PREVIEW_DATA_MAX_LENGTH)


class PdfFileDataSource(DataSource):
    def _create_metadata(self, doc_file_path: str):
        file_name = doc_file_path.split("/")[-1]

        return {"file_name": file_name} if file_name else {}

    def _load_data(self, pdf_file_path: str) -> Iterable[Document]:
        loader = UnstructuredPDFLoader(pdf_file_path)
        docs = loader.lazy_load()

        for doc in docs:
            yield Document(content=doc.page_content, metadata=self._create_metadata(pdf_file_path))

    def load_preview_data(self, pdf_file_path: str) -> str:
        from PyPDF2 import PdfReader
        def read_first_page_pdf(file_path, encoding='utf-8'):
            with open(file_path, 'rb') as file:
                reader = PdfReader(io.BytesIO(file.read()))
                first_page = reader.pages[0]
                return first_page.extract_text().encode('utf-8').decode(encoding)

        first_page_content = read_first_page_pdf(pdf_file_path)

        if first_page_content:
            return truncate_content(first_page_content, self.PREVIEW_DATA_MAX_LENGTH)
        else:
            return ""


class ConfluenceDataSource(DataSource):
    def _create_metadata(self, confluence, page):
        metadata = {}

        if page.get("id"):
            metadata["id"] = page["id"]
        if page.get("title"):
            metadata["title"] = page["title"]
        if page.get("_links") and page["_links"].get("webui"):
            metadata["url"] = f"{confluence.url}{page["_links"]["webui"]}"

        return metadata if metadata else {}

    # page는 한 번에 최대 100개씩만 가져와진다.
    def get_pages(self, confluence: Confluence, space_key: str, start: int, batch_size: int):
        return confluence.get_all_pages_from_space(
            space=space_key, start=start, limit=batch_size, status=None, expand="body.view")

    def get_all_pages(self,
                      confluence: Confluence,
                      space_key: str,
                      batch_size: int = 100,
                      request_interval: int = 0.25) -> Iterable[Document]:
        logger.info(f"Start scrape confluence pages from space_key: {space_key}.")
        start = 0
        total_page_count = 0

        while True:
            pages = self.get_pages(confluence, space_key, start, batch_size)
            documents = []

            for page in pages:
                content = page["body"]["view"]["value"]
                metadata = self._create_metadata(confluence, page)
                documents.append(Document(content=content, metadata=metadata))

            page_count = len(documents)
            total_page_count += page_count
            logger.info(f"{page_count} confluence pages are scraped from {start} to {start + page_count - 1}")

            yield from documents

            if page_count < batch_size:
                break

            start += batch_size
            sleep(request_interval)

        logger.info(f"Finish scrape confluence pages from space: {space_key}. Total {total_page_count} pages are scraped.")

    '''
    TDE Confluence는 초당 5회 이내로 API 호출을 제한합니다.

    :param url: Confluence 서버 URL
    :param user_name: Confluence 사용자 이름
    :param access_token: Confluence API 토큰
    :param batch_size: 한 번에 가져올 페이지 수. 최대 100개까지 가능합니다.
    '''
    def _load_data(self, url: str, access_token: str, space_key: str, batch_size: int = 100) -> Iterable[Document]:
        confluence = Confluence(url=url, token=access_token)
        return self.get_all_pages(confluence, space_key, batch_size)

    def load_preview_data(self, url: str, access_token: str, space_key: str, batch_size: int = 1) -> str:
        confluence = Confluence(url=url, token=access_token)
        pages = self.get_pages(confluence, space_key, start=0, batch_size=batch_size)
        first_page = get_first(pages)

        if first_page:
            content = first_page.get("body", "").get("view", "").get("value", "")
            return truncate_content(content, self.PREVIEW_DATA_MAX_LENGTH)
        else:
            return ""


class GitlabDataSource(DataSource):
    def _create_metadata(self, file_content: ProjectFile):
        return {"file_path": file_content.file_path} if file_content.file_path else {}

    def _load_data(self, url: str, namespace: str, project_name: str, branch: str, private_token: str) \
            -> Iterable[Document]:
        def filter_valid_files(tree):
            include_extensions = [".txt", ".md", ".rst", ".html", ".htm", ".xml", ".json", ".csv", ".tsv", ".yaml",
                                  ".yml", ".log", ".py", ".java", ".js", ".ts", ".c", ".cpp", ".h", ".hpp", ".cs",
                                  ".go", ".php", ".rb", ".sh", ".scala", ".sql", ".ipynb", ".conf", ".properties"]

            for item in tree:
                if item['type'] == 'blob' and any([item['path'].endswith(ext) for ext in include_extensions]):
                    yield item

        gl = gitlab.Gitlab(url=url, private_token=private_token)
        project = gl.projects.get(f"{namespace}/{project_name}")
        tree = project.repository_tree(ref=branch, recursive=True, all=True)

        for item in filter_valid_files(tree):
            file_content = project.files.get(file_path=item['path'], ref=branch)
            try:
                decoded_content = base64.b64decode(file_content.content).decode('utf-8')
                metadata = self._create_metadata(file_content)
                yield Document(content=decoded_content, metadata=metadata)
            except UnicodeDecodeError:
                logger.error(f"Can't decode {item['path']}")

    def _get_random_file_from_commits(self, project, branch, max_commits=5):
        try:
            commits = project.commits.list(ref_name=branch, all=True)
            file_paths = set()

            for commit in commits[:max_commits]:
                diff = project.commits.get(commit.id).diff()
                for change in diff:
                    file_paths.add(change['new_path'])

                if len(file_paths) >= 50:  # 충분한 파일을 수집했다면 중단
                    break

            if not file_paths:
                return None

            return random.choice(list(file_paths))

        except gitlab.exceptions.GitlabError as e:
            print(f"Error getting file list: {e}")
            return None

    def load_preview_data(self, url: str, namespace: str, project_name: str, branch: str, private_token: str) -> str:
        gl = gitlab.Gitlab(url=url, private_token=private_token)
        project = gl.projects.get(f"{namespace}/{project_name}")
        preview_file_path = self._get_random_file_from_commits(project, branch)
        file_content = project.files.get(file_path=preview_file_path, ref=branch)
        try:
            decoded_content = base64.b64decode(file_content.content).decode('utf-8')
            return truncate_content(decoded_content, self.PREVIEW_DATA_MAX_LENGTH)
        except UnicodeDecodeError:
            logger.error(f"Can't decode {preview_file_path}")


class UrlDataSource(DataSource):
    def _create_metadata(self, document: langchain_core.documents.Document):
        url = document.metadata.get("source")
        return {"url": url} if url else {}

    def _load_data(self, url: str, max_depth: int, base_url: str, extractor: Callable[[str], str] = bs4_extractor) \
            -> Iterable[Document]:
        '''
        :param url: scraping 하고자 하는 웹사이트의 root url
        :param max_depth: 재귀적으로 링크를 타고 들어갈 최대 깊이
        :param base_url: prevent_outside=True 일 경우 base_url을 기준으로 외부 링크를 제외합니다.
        :param extractor: 페이지에서 텍스트를 추출하는 함수
        :return: generator[str]
        '''

        loader = RecursiveUrlLoader(url=url, max_depth=max_depth, extractor=extractor, base_url=base_url)
        docs = loader.lazy_load()

        for doc in docs:
            yield Document(content=doc.page_content, metadata=self._create_metadata(doc))

    def load_preview_data(self, url: str, base_url: str, extractor: Callable[[str], str] = bs4_extractor) -> str:
        loader = RecursiveUrlLoader(url=url, max_depth=1, extractor=extractor, base_url=base_url)
        docs = loader.lazy_load()

        first_doc = get_first(docs)
        if first_doc:
            return truncate_content(first_doc.page_content, self.PREVIEW_DATA_MAX_LENGTH)
        else:
            return ""


class DocumentFileDataSource(DataSource):
    def _create_metadata(self, doc_file_path: str):
        file_name = doc_file_path.split("/")[-1]

        return {"file_name": file_name} if file_name else {}

    def _load_data(self, doc_file_path: str) -> list[str]:
        loader = UnstructuredFileLoader(file_path=doc_file_path, mode="single")
        docs = loader.lazy_load()

        for doc in docs:
            yield Document(content=doc.page_content, metadata=self._create_metadata(doc_file_path))

    def load_preview_data(self, doc_file_path: str) -> str:
        doc = DocxDocument(doc_file_path)
        first_page_content = []

        for paragraph in doc.paragraphs:
            if paragraph.paragraph_format.page_break_before:
                break
            first_page_content.append(paragraph.text)

        first_page = '\n'.join(first_page_content)

        if first_page:
            return truncate_content(first_page, self.PREVIEW_DATA_MAX_LENGTH)
        else:
            return ""


class JiraDataSource(DataSource):
    def _create_metadata(self, cleaned_fields: dict) -> dict:
        metadata_keys = ["key", "updated", "assignee"]
        metadata = {key: cleaned_fields[key] for key in metadata_keys if key in cleaned_fields}

        return metadata if metadata else {}

    def _load_data(self, url: str, project_key: str, access_token: str, start: int = 0, limit: int = 1000,
                   call_interval: float = 0.05) -> Iterable[Document]:
        jira = Jira(url=url, token=access_token)
        issue_keys = jira.get_project_issuekey_all(project_key, start=start, limit=limit)

        while issue_keys:
            logger.info(f"Retrieved {len(issue_keys)} issue keys of {project_key} project. start from {issue_keys[0]}")

            for issue_key in issue_keys:
                issue = jira.issue(issue_key)

                required_fields = filter_required_fields(issue["fields"])
                cleaned_fields = cleanse_fields(required_fields)
                cleaned_fields["key"] = issue["key"]
                key = {"key": issue["key"]}

                content = json.dumps({**key, **cleaned_fields}, indent=2, ensure_ascii=False)
                metadata = self._create_metadata(cleaned_fields)

                yield Document(content=content, metadata=metadata)

                start += 1
                sleep(call_interval)

            issue_keys = jira.get_project_issuekey_all(project_key, start=start, limit=limit)
        else:
            logger.info(f"No issue keys retrieved. Finish loading jira issues for {project_key} project.")

    def load_preview_data(self, url: str, project_key: str, access_token: str, start: int = 0, limit: int = 1,
                          call_interval: float = 0.05) -> str:
        jira = Jira(url=url, token=access_token)
        issue_keys = jira.get_project_issuekey_all(project_key, start=start, limit=limit)

        if issue_keys:
            issue = jira.issue(issue_keys[0])
            required_fields = filter_required_fields(issue["fields"])
            cleaned_fields = cleanse_fields(required_fields)
            cleaned_fields["key"] = issue["key"]
            key = {"key": issue["key"]}

            return json.dumps({**key, **cleaned_fields}, indent=2, ensure_ascii=False)
        else:
            return ""


class GitlabDiscussionDataSource(DataSource):
    def _get_file_content(self, project: Project, file_path: str, sha: str):
        """특정 파일의 내용 가져오기"""
        # 파일 내용 가져오기 (base64로 인코딩되어 있음)
        try:
            file_obj = project.files.get(file_path=file_path, ref=sha)

            content = file_obj.decode()

            return content.decode('utf-8')
        except Exception as e:
            return None

    def _get_code_context(self, file_content: str, start_line: int, end_line: int):
        """코드 컨텍스트 출력"""
        lines = file_content.split('\n')
        context = lines[max(0, start_line - 1):min(len(lines), end_line)]

        texts = "\n"
        for i, line in enumerate(context, start=start_line):
            texts += f"{i}: {line}\n"

        return texts

    def _load_data(self, url: str, namespace: str, project_name: str, private_token: str):
        gl = gitlab.Gitlab(url=url, private_token=private_token)
        project = gl.projects.get(f"{namespace}/{project_name}")
        merge_requests = list(project.mergerequests.list(all=True))

        discussion_texts = []
        for mr in merge_requests:
            # merge_header = f"\nMerge Request #{mr.iid}: {mr.title}"
            discussions = list(mr.discussions.list(all=True))

            for discussion in discussions:
                note = discussion.attributes['notes'][0]
                if note['system']:
                    continue

                author = note['author']['name']
                if author == "DevSecOps IT보안운영(임시)":
                    continue

                header = f"Discussion ID: {discussion.id}"
                texts = ""
                if 'position' in note:
                    position = note['position']
                    file_path = position['new_path']
                    start_line = position.get('new_line')
                    if start_line is None:
                        continue

                    start_line -= 5 if start_line > 5 else start_line
                    end_line = start_line + 10  # 컨텍스트로 5줄 표시

                    texts += f"\n    파일: {file_path}"
                    file_content = self._get_file_content(project, file_path, mr.sha)
                    if file_content:
                        texts += f"\n    코드 컨텍스트:"
                        texts += self._get_code_context(file_content, start_line, end_line)

                for note in discussion.attributes['notes']:
                    texts += f"\n    작성자: {note['author']['name']}"
                    texts += f"\n    내용: {note['body']}"

                if texts:
                    discussion_texts.append(header + texts)

        return discussion_texts


def create_data_source(data_source_name: str, created_by: str, description: str, data_source_type: str) \
        -> DataSource:
    datasource_data = {
        "id": create_data_source_id(created_by, data_source_name),
        "name": data_source_name,
        "description": description,
        "data_source_type": data_source_type
    }

    if data_source_type == DataSourceType.TEXT.value:
        return TextDataSource(**datasource_data)
    elif data_source_type == DataSourceType.PDF_FILE.value:
        return PdfFileDataSource(**datasource_data)
    elif data_source_type == DataSourceType.CONFLUENCE.value:
        return ConfluenceDataSource(**datasource_data)
    elif data_source_type == DataSourceType.GITLAB.value:
        return GitlabDataSource(**datasource_data)
    elif data_source_type == DataSourceType.URL.value:
        return UrlDataSource(**datasource_data)
    elif data_source_type == DataSourceType.DOC_FILE.value:
        return DocumentFileDataSource(**datasource_data)
    elif data_source_type == DataSourceType.JIRA.value:
        return JiraDataSource(**datasource_data)
    elif data_source_type == DataSourceType.GITLAB_DISCUSSION.value:
        return GitlabDiscussionDataSource(**datasource_data)
    else:
        raise ValueError(f"Invalid data source type: {data_source_type}")


def create_data_source_tool(name: str, username: str, datasource: DataSource) -> Tool:
    tool_name = create_tool_name(ComponentType.DATASOURCE, name, username)
    return datasource.as_retriever_tool(tool_name, datasource.description)