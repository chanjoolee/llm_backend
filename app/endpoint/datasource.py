import asyncio
from datetime import datetime
from enum import Enum
import logging
import os
from typing import Annotated, AsyncIterable, Dict, Iterable, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile ,status,BackgroundTasks,logger as logger_fastapi
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import requests
from sqlalchemy import insert, or_, text
from sqlalchemy.orm import Session , joinedload , selectinload
from ai_core.data_source.base import DataSourceType, create_data_source
from ai_core.data_source.splitter.base import SplitterType, create_splitter ,Language,TextSplitter
from ai_core.data_source.utils.time_utils import datetime_to_iso_8601_str, get_iso_8601_current_time, iso_8601_str_to_datetime
from ai_core.data_source.utils.utils import create_collection_name, create_data_source_id, split_texts
from ai_core.data_source.model.document import Document
from ai_core import AI_CORE_ROOT_DIR ,CHROMA_DB_DEFAULT_PERSIST_DIR , CHROMA_DB_TEST_DATA_PATH , DATA_SOURCE_SAVE_BASE_DIR
from ai_core.llm_api_provider import LlmApiProvider
from app import models, database
from app.utils import utils
from app.endpoint.login import cookie , SessionData , verifier
from app.database import SessionLocal , get_db
import aiofiles
import pydash
from .realtime_updates import broadcast_update
from opensearchpy import OpenSearch
import app.config
from ai_core.data_source.utils import opensearch_utils as op_utils

active_tasks = {}

logger_sql = logging.getLogger('sqlalchemy.engine')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
comment = text("/* is_endpoint_query */ 1=1")

router = APIRouter()
COLLECTION_DIR = os.path.join(os.getenv("DAISY_ROOT_FOLDER") ,'datasource/collection')
CHROMA_DB_DIR = os.path.join(COLLECTION_DIR ,'daisy_chromadb')

opensearch_hosts = os.getenv('opensearch_hosts')
opensearch_auth = (os.getenv('opensearch_username'), os.getenv('opensearch_password'))
base_succeeded_at = "2024-01-01T00:00:53.000+0900"
# Function to calculate total size of files in a directory
def get_total_file_size(directory):
    total_size = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            total_size += os.path.getsize(filepath)
    return total_size

def sanitize_git_url(git_url: str) -> str:
    # Replace special characters with underscores or other safe characters
    return git_url.replace("://", "_").replace("/", "_").replace(".", "_")

def construct_file_save_path(datasource_id, file_name: str) -> str:
    """
    파일경로
    """
    root_path = os.getenv("DAISY_ROOT_FOLDER", "/var/webapps/files")  # Default to /var/webapps if DAISY_ROOT_FOLDER is not set
    # file_save_path = os.path.join(root_path, 'datasource/data', str(datasource_id) + '_' + file_name)
    file_save_path = os.path.join(COLLECTION_DIR,str(datasource_id) + '_' + file_name)
    return file_save_path

# Define a wrapper to handle asyncio.to_thread and callback
async def run_with_callback(sync_func, callback_func, *args, **kwargs):
    result = await asyncio.to_thread(sync_func, *args, **kwargs)
    callback_func(result)

async def async_wrap(generator):
    async for item in generator:
        yield item

@router.post(
    "/",
    response_model=models.Datasource,
    dependencies=[Depends(cookie)],
    description="""<pre>
    Creates a new data source.  model: DatasourceCreate
    <h3>Request Body:</h3>
    
        - name (str): The name of the data source.
        - description (Optional[str]): A description of the data source.
        - visibility (str): The visibility of the data source: private or public.
        - datasource_type (Optional[str]): The type of the data source (e.g., text, pdf_file, confluence, gitlab, gitlab_discussion, url, doc_file, jira).
        - namespace (Optional[str]): Namespace for GitLab projects.
        - project_name (Optional[str]): Project name in GitLab.
        - branch (Optional[str]): Branch name in GitLab.
        - project_key (Optional[str]): Project key for JIRA.
        - space_key (Optional[str]): Space key for Confluence.
        - start (Optional[int]): Start index for JIRA issues.
        - limit (Optional[int]): Limit for JIRA issues.
        - file_path (Optional[str]): File path for document or PDF.
        - raw_text (Optional[str]): Raw text data.
        - url (Optional[str]): URL for below  data source.
            - url
            - confluence
            - jira
            - gitlab
            - gitlab_discussion
        - base_url (Optional[str]): Base URL for below  data source.
            - url
        - max_depth (Optional[int]): Maximum depth for URL crawling.
        - tag_ids (Optional[List[int]]): A list of tag IDs to associate with the datasource.
        - file: Optional file for document or PDF.
    </pre>
    """
)
async def datasource_create(
    name: str = Form(..., description="Name of the data source"),
    description: str = Form(..., description="Description of the data source"),
    visibility: str = Form("private", description="Visibility of the data source: private or public"),
    datasource_type: Optional[DataSourceType] = Form(DataSourceType.TEXT, description="Type of the data source (e.g., text, pdf_file, confluence, gitlab,gitlab_discussion, url, doc_file, jira)"),
    namespace: Optional[str] = Form(None, description="Namespace for GitLab projects"),
    project_name: Optional[str] = Form(None, description="Project name in GitLab"),
    branch: Optional[str] = Form(None, description="Branch name in GitLab"),
    space_key: Optional[str] = Form(None, description="Space key for Confluence"),
    project_key: Optional[str] = Form(None, description="Project key for JIRA"),
    start: Optional[int] = Form(None, description="Start index for JIRA issues"),
    limit: Optional[int] = Form(1000, description="Limit for JIRA issues"),
    token: Optional[str] = Form(None, description="API Token for JIRA"),
    raw_text: Optional[str] = Form(None, description="Raw text data" ),
    url: Optional[str] = Form(None, 
        description="""
            URL for URL data source
            - url
            - confluence
            - jira
            - gitlab
            - gitlab_discussion
        """
    ),
    base_url: Optional[str] = Form(None, description="Base URL for URL data source"),
    max_depth: Optional[int] = Form(None, description="Maximum depth for URL crawling"),
    tag_ids: Optional[str] = Form(None,description="Comma-separated list of tag IDs"),
    # payload: models.DatasourceCreate ,
    file: Annotated[UploadFile ,File(description="File path for document or PDF")] = None,  # File upload
    db: Session = Depends(database.get_db),
    session_data: SessionData = Depends(verifier)
   
):
    logger.info(f"Create Datasource started")
    global active_tasks
    validate_datasource(datasource_type, namespace, project_name, branch, space_key, project_key, start, limit,token, raw_text, url, base_url, max_depth)
    
    # Create a dictionary of all the parameters
    params = {
        "name": name,
        "description": description,
        "visibility": visibility,
        "datasource_type": datasource_type,
        "namespace": namespace,
        "project_name": project_name,
        "branch": branch,
        "space_key": space_key,
        "project_key": project_key,
        "start": start,
        "limit": limit,
        "token": token,
        "raw_text": raw_text,
        "url": url,
        "base_url": base_url,
        "max_depth": max_depth,
        "tag_ids": tag_ids
    }

    # Filter out None values
    update_data = {key: value for key, value in params.items() if value is not None}


    # Generate the datasource_id using the create_data_source_id function
    datasource_id = create_data_source_id(session_data.nickname, update_data['name'])
    

    # Check if a DataSource with the same name already exists
    existing_datasource = (
        db.query(database.DataSource)
        .filter(database.DataSource.datasource_id == datasource_id)
        .filter(comment)  
    ).first()
    if existing_datasource:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"DataSource with this ID of '{datasource_id}' already exists.")

    # DB
    # Create a new DataSource instance
    db_datasource = database.DataSource(
        datasource_id=datasource_id,
        create_user=session_data.user_id
    )
    
    # Apply the filtered data to the SQLAlchemy model instance
    for key, value in update_data.items():
        if key not in ["tag_ids"]:
            if isinstance(value,Enum):
                setattr(db_datasource, key, value.value)
            else: 
                setattr(db_datasource, key, value)
    # Add tags to the tool. something different with other 
    if 'tag_ids' in update_data and len(update_data['tag_ids']) >= 0 :
        tag_ids_list = tag_ids.split(',')
        for tag_id in tag_ids_list:
            db_tag = db.query(database.Tag).filter(comment).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
            if db_tag:
                db_datasource.tags.append(db_tag)


    # text, pdf_file, confluence, gitlab, url, doc_file, jira
    datasource_type1 = DataSourceType(db_datasource.datasource_type)
    if file:
        if datasource_type1 in (DataSourceType.PDF_FILE ,DataSourceType.DOC_FILE):
            # Use the utility function to construct the file save path
            file_save_path = construct_file_save_path(datasource_id=datasource_id,file_name=file.filename)
            dir_save_path = os.path.dirname(file_save_path)
            os.makedirs(dir_save_path, exist_ok=True)
            # Save the file asynchronously using aiofiles
            async with aiofiles.open(file_save_path, 'wb') as out_file:
                content = await file.read()  # Read file content asynchronously
                await out_file.write(content)  # Write the content asynchronously
            db_datasource.file_path = file_save_path
            
    # Add to the database session
    db.add(db_datasource)
    db.flush()
    db.refresh(db_datasource)
    
    
    
    # 데이터를 텍스트 파일로 저장
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )

    
    #  before start task of opensearch , apply to db. if there is error , do not  execute_save_data
    db_datasource.status = database.DatasourceDownloadStatus.downloading
    db.commit()
    # Schedule the execution and apply the callback
    task = asyncio.create_task(datasource_save_data(datasource_type, session_data, db_datasource, data_source) )
    task.add_done_callback(lambda future: datasource_save_data_callback(future,session_data, datasource_id))
    
    active_tasks[datasource_id] = task
    
    return db_datasource

def datasource_save_data_callback(future,session_data, datasource_id):
    logger.info("Datasource Indexing Completed")  
    logger.info("Datasource Callback Started")  
    db_callback = database.SessionLocal()
    try:
        db_datasource = db_callback.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
        if future.exception():
            logging.error(f"Datasource task failed: {future.exception()}")
            db_datasource.status = database.DatasourceDownloadStatus.error
        else:
            # embeded, total = future.result()
            # logger.info("Embedding task completed. Number of chunks embedded / Total: ", str(embeded), " / ", str(total))
            result =  future.result()
            db_datasource.downloaded_at = datetime.now(database.KST)
            # db_datasource.uploaded_at = datetime.now(database.KST)
            db_datasource.status = database.DatasourceDownloadStatus.downloaded
        
        
    except Exception as e:
        db_datasource.status = database.DatasourceDownloadStatus.error
        logger.error(f"Datasource download failed: {e}")
        
    except asyncio.exceptions.CancelledError as e:
        db_datasource.status = database.DatasourceDownloadStatus.cancelled
        # db_embedding.status_message = traceback.format_exc()
        logging.error(f"Datasource download cancelled: {e}")
        
    finally:
        db_datasource.updated_at = datetime.now(database.KST)
        db_datasource.update_user = session_data.user_id
        
        update_message = {
            "type": "datasource_update",
            "data": models.Datasource.from_orm(db_datasource).dict()
        }
        asyncio.create_task(broadcast_update(update_message))
        # active_tasks 해제
        if datasource_id in active_tasks:
            active_tasks.pop(datasource_id,None)
        db_callback.commit()
        db_callback.close()  
        logger.info("Datasource Callback Completed")

async def datasource_save_data(datasource_type, session_data, db_datasource, data_source):
    logger.info("Datasource Create Indexing started")  
    if datasource_type == DataSourceType.TEXT:
        data_size = len(db_datasource.raw_text.encode('utf-8'))
        save_task = await asyncio.to_thread(
            data_source.save_data,
            last_update_succeeded_at=get_iso_8601_current_time(),
            raw_text=[db_datasource.raw_text]
        )
        
    if datasource_type == DataSourceType.PDF_FILE:
        pdf_file_path = db_datasource.file_path
        if os.path.exists(pdf_file_path):
            save_task = await asyncio.to_thread(
                data_source.save_data,
                last_update_succeeded_at=get_iso_8601_current_time(),
                pdf_file_path=db_datasource.file_path
            )
        
    if datasource_type == DataSourceType.DOC_FILE:
        doc_file_path = db_datasource.file_path
        if os.path.exists(doc_file_path):
            save_task = await asyncio.to_thread(
                data_source.save_data,
                last_update_succeeded_at=get_iso_8601_current_time(),
                doc_file_path=db_datasource.file_path
            )
        
    if datasource_type == DataSourceType.CONFLUENCE:
        if utils.is_empty(session_data.token_confluence):
            raise HTTPException(status_code=422, detail=f"Token of Confluence is required for {datasource_type.name} datasource type")
        
        save_task =  await asyncio.to_thread(
            data_source.save_data, 
            last_update_succeeded_at=get_iso_8601_current_time(),
            url=db_datasource.url ,
            access_token=session_data.token_confluence,
            space_key=db_datasource.space_key
        ) 
        
    if datasource_type == DataSourceType.JIRA:
        if utils.is_empty(db_datasource.token):
            raise HTTPException(status_code=422, detail=f"Token of Jira is required for {datasource_type.name} datasource type")
        save_task = await asyncio.to_thread(
            data_source.save_data ,
            last_update_succeeded_at=get_iso_8601_current_time(),
            url=db_datasource.url ,
            access_token=db_datasource.token,
            project_key=db_datasource.project_key,
            start=db_datasource.start, 
            limit=db_datasource.limit
        ) 
        
    if datasource_type == DataSourceType.GITLAB:
        if utils.is_empty(session_data.token_gitlab):
            raise HTTPException(status_code=422, detail=f"Token of Gitlab is required for {datasource_type.name} datasource type")
        save_task = await asyncio.to_thread(
            data_source.save_data,
            last_update_succeeded_at=get_iso_8601_current_time(),
            url=db_datasource.url ,
            namespace=db_datasource.namespace, 
            project_name = db_datasource.project_name,
            branch=db_datasource.branch,
            private_token=session_data.token_gitlab) 
        
    if datasource_type == DataSourceType.GITLAB_DISCUSSION:
        if utils.is_empty(session_data.token_gitlab):
            raise HTTPException(status_code=422, detail=f"Token of Gitlab is required for {datasource_type.name} datasource type")
        save_task = await asyncio.to_thread(
            data_source.save_data,
            last_update_succeeded_at=get_iso_8601_current_time(),
            url=db_datasource.url ,
            namespace=db_datasource.namespace, 
            project_name = db_datasource.project_name,
            private_token=session_data.token_gitlab
        ) 
        
    if datasource_type == DataSourceType.URL:
        save_task = await asyncio.to_thread(
            data_source.save_data,
            last_update_succeeded_at=get_iso_8601_current_time(),
            url=db_datasource.url ,
            base_url=db_datasource.base_url,
            max_depth=db_datasource.max_depth 
        )
    return save_task

def validate_datasource(datasource_type : DataSourceType, namespace, project_name, branch, space_key, project_key, start, limit, token, raw_text, url, base_url, max_depth):
    if datasource_type in [DataSourceType.TEXT]:
        if utils.is_empty(raw_text):
            raise HTTPException(status_code=422, detail=f"raw_text is required for {datasource_type.name} datasource type")
        
    # update_data = payload.dict(exclude_unset=True)
    if datasource_type in [DataSourceType.CONFLUENCE,DataSourceType.GITLAB,DataSourceType.GITLAB_DISCUSSION,DataSourceType.JIRA]:
        if utils.is_empty(url):
            raise HTTPException(status_code=422, detail=f"URL is required for {datasource_type.name} datasource type")
        
    if datasource_type in [DataSourceType.GITLAB]:
        if utils.is_empty(namespace):
            raise HTTPException(status_code=422, detail=f"namespace is required for {datasource_type.name} datasource type")
        if utils.is_empty(project_name):
            raise HTTPException(status_code=422, detail=f"project_name is required for {datasource_type.name} datasource type")
        if utils.is_empty(branch):
            raise HTTPException(status_code=422, detail=f"branch is required for {datasource_type.name} datasource type")
        
    if datasource_type in [DataSourceType.GITLAB_DISCUSSION]:
        if utils.is_empty(namespace):
            raise HTTPException(status_code=422, detail=f"namespace is required for {datasource_type.name} datasource type")
        if utils.is_empty(project_name):
            raise HTTPException(status_code=422, detail=f"project_name is required for {datasource_type.name} datasource type")
        
    if datasource_type in [DataSourceType.CONFLUENCE]:
        if utils.is_empty(space_key):
            raise HTTPException(status_code=422, detail=f"space_key is required for {datasource_type.name} datasource type")
            
    if datasource_type in [DataSourceType.JIRA]:
        if utils.is_empty(project_key):
            raise HTTPException(status_code=422, detail=f"project_key is required for {datasource_type.name} datasource type")
        if utils.is_empty(start):
            raise HTTPException(status_code=422, detail=f"start is required for {datasource_type.name} datasource type")
        
        if utils.is_empty(limit):
            raise HTTPException(status_code=422, detail=f"limit is required for {datasource_type.name} datasource type")
        if utils.is_empty(token):
            raise HTTPException(status_code=422, detail=f"token is required for {datasource_type.name} datasource type")
        
    if datasource_type in [DataSourceType.URL]:
        if utils.is_empty(base_url):
            raise HTTPException(status_code=422, detail=f"base_url is required for {datasource_type.name} datasource type")
        
        if utils.is_empty(max_depth):
            raise HTTPException(status_code=422, detail=f"max_depth is required for {datasource_type.name} datasource type")


@router.put(
    "/{datasource_id}",
    response_model=models.Datasource,
    dependencies=[Depends(cookie)],
    description="""
    Update an existing data source using multipart form data.
    Opensearch 에 download 는 안한다.
    """
)
async def update_datasource(
    datasource_id: str,
    name: Optional[str] = Form(None, description="Name of the data source"),
    description: Optional[str] = Form(None, description="Description of the data source"),
    visibility: Optional[models.Visibility] = Form(None, description="Visibility of the data source: private or public"),
    datasource_type: Optional[DataSourceType] = Form(None, description="Type of the data source (e.g., text, pdf_file, confluence, gitlab, gitlab_discussion, url, doc_file, jira)"),
    namespace: Optional[str] = Form(None, description="Namespace for GitLab projects"),
    project_name: Optional[str] = Form(None, description="Project name in GitLab"),
    branch: Optional[str] = Form(None, description="Branch name in GitLab"),
    space_key: Optional[str] = Form(None, description="Space key for Confluence"),
    project_key: Optional[str] = Form(None, description="Project key for JIRA"),
    start: Optional[int] = Form(None, description="Start index for JIRA issues"),
    limit: Optional[int] = Form(1000, description="Limit for JIRA issues"),
    token: Optional[str] = Form(None, description="API Token for JIRA"),
    raw_text: Optional[str] = Form(None, description="Raw text data" ),
    url: Optional[str] = Form(None, 
        description="""
            URL for URL data source
            - url
            - confluence
            - jira
            - gitlab
            - gitlab_discussion
        """
    ),
    base_url: Optional[str] = Form(None, description="Base URL for URL data source"),
    max_depth: Optional[int] = Form(None, description="Maximum depth for URL crawling"),
    tag_ids: Optional[str] = Form(None, description="Comma-separated list of tag IDs"),
    file: Annotated[UploadFile, File(description="File for document or PDF")] = None,  # File upload
    db: Session = Depends(database.get_db),
    session_data: SessionData = Depends(verifier)
):
    # Fetch the existing datasource from the database
    db_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
    if not db_datasource:
        raise HTTPException(status_code=404, detail="DataSource not found")
    
    # Create a dictionary of all the updated parameters
    params = {
        "name": name,
        "description": description,
        "visibility": visibility,
        "datasource_type": datasource_type,
        "namespace": namespace,
        "project_name": project_name,
        "branch": branch,
        "space_key": space_key,
        "project_key": project_key,
        "start": start,
        "limit": limit,
        "token": token,
        "raw_text": raw_text,
        "url": url,
        "base_url": base_url,
        "max_depth": max_depth,
        "tag_ids": tag_ids
    }

    # Filter out None values
    update_data = {key: value for key, value in params.items() if value is not None}

    # Apply the filtered data to the existing datasource
    for key, value in update_data.items():
        if key not in ["tag_ids"]:
            if isinstance(value,Enum):
                setattr(db_datasource, key, value.value)
            else: 
                setattr(db_datasource, key, value)

        # Add tags to the tool. something different with other 
    if 'tag_ids' in update_data and len(update_data['tag_ids']) >= 0 :
        db_datasource.tags = []
        tag_ids_list = tag_ids.split(',')
        for tag_id in tag_ids_list:
            db_tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
            if db_tag:
                db_datasource.tags.append(db_tag)
                
    datasource_type1 = DataSourceType(db_datasource.datasource_type)
    # update 에서는 validation을 db에 있는 값으로 한다. 왜냐하면, 많은 항목들이 option을로 넘어옴.
    validate_datasource(datasource_type1, db_datasource.namespace, db_datasource.project_name, db_datasource.branch, db_datasource.space_key, db_datasource.project_key, db_datasource.start, db_datasource.limit, db_datasource.token, db_datasource.raw_text, db_datasource.url, db_datasource.base_url, db_datasource.max_depth)
    
    # Handle file upload
    
    if file:
        if datasource_type1 in (DataSourceType.PDF_FILE ,DataSourceType.DOC_FILE):
            # Use the utility function to construct the file save path
            file_save_path = construct_file_save_path(datasource_id=datasource_id, file_name=file.filename)
            dir_save_path = os.path.dirname(file_save_path)
            os.makedirs(dir_save_path, exist_ok=True)
            
            # Save the file asynchronously using aiofiles
            async with aiofiles.open(file_save_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            db_datasource.file_path = file_save_path

    return db_datasource


@router.put(
    "/redownload/{datasource_id}",
    dependencies=[Depends(cookie)],
    description="""
    Re Download data source
    """
)
async def datasource_redownload(
    datasource_id: str,
    db: Session = Depends(database.get_db_async),
    session_data: SessionData = Depends(verifier)
):
    
    logger.info(f"Data Redownload started : {datasource_id}")
    global active_tasks
    
    db_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
    if not db_datasource:
            raise HTTPException(status_code=404, detail="DataSource not found")
    
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )
    
    datasource_type = DataSourceType(db_datasource.datasource_type)
       
    #  before start task of opensearch , apply to db. if there is error , do not  execute_save_data
    before_staus = db_datasource.status
    db_datasource.status = database.DatasourceDownloadStatus.downloading
    db.commit()
    # Schedule the execution and apply the callback
    if before_staus == database.DatasourceDownloadStatus.downloaded:
        task = asyncio.create_task(datasource_update_data(session_data, db_datasource, data_source, datasource_type))
        task.add_done_callback(lambda future: datasource_update_data_callback(future,datasource_id, session_data))
    else: 
        task = asyncio.create_task(datasource_save_data(datasource_type, session_data, db_datasource, data_source) )
        task.add_done_callback(lambda future: datasource_save_data_callback(future,session_data, datasource_id))
    active_tasks[datasource_id] = task
    
    return db_datasource

def datasource_update_data_callback(future, datasource_id, session_data):
    logger.info("Datasource Re Indexing Completed")  
    logger.info("Datasource Callback Started")  
    db_callback = database.SessionLocal()
    try:
        db_datasource = db_callback.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
        if future.exception():
            logging.error(f"Datasource task failed: {future.exception()}")
            db_datasource.status = database.DatasourceDownloadStatus.error
        else:
            result =  future.result()
            db_datasource.downloaded_at = datetime.now(database.KST)
            db_datasource.status = database.DatasourceDownloadStatus.downloaded
        
        
    except Exception as e:
        db_datasource.status = database.DatasourceDownloadStatus.errordk
        logger.error(f"Datasource Re download failed: {e}")
        
    except asyncio.exceptions.CancelledError as e:
        db_datasource.status = database.DatasourceDownloadStatus.cancelled
        # db_embedding.status_message = traceback.format_exc()
        logging.error(f"Datasource Re download cancelled: {e}")
        
    finally:
        db_datasource.updated_at = datetime.now(database.KST)
        db_datasource.update_user = session_data.user_id
        
        update_message = {
            "type": "datasource_update",
            "data": models.Datasource.from_orm(db_datasource).dict()
        }
        asyncio.create_task(broadcast_update(update_message))
        # active_tasks 해제
        if datasource_id in active_tasks:
            active_tasks.pop(datasource_id,None)
        db_callback.commit()
        db_callback.close() 
        logger.info("Datasource Callback Completed")

async def datasource_update_data(session_data, db_datasource, data_source, datasource_type):
    logger.info("Datasource Re Downdoad Indexing started")  
    if db_datasource.downloaded_at: 
        since = datetime_to_iso_8601_str(db_datasource.downloaded_at)
    else:
        # since = datetime_to_iso_8601_str(db_datasource.created_at)
        since = base_succeeded_at
        
    if datasource_type == DataSourceType.TEXT:
        save_task = await asyncio.to_thread(
            data_source.update_data,
            raw_text=[db_datasource.raw_text],
            since=since,
            last_update_succeeded_at=get_iso_8601_current_time()
        )
        
    if datasource_type == DataSourceType.PDF_FILE:
        pdf_file_path = db_datasource.file_path
        if os.path.exists(pdf_file_path):
            save_task = await asyncio.to_thread(
                data_source.update_data,
                pdf_file_path=db_datasource.file_path,
                since=since,
                last_update_succeeded_at=get_iso_8601_current_time()
            )
        
    if datasource_type == DataSourceType.DOC_FILE:
        doc_file_path = db_datasource.file_path
        if os.path.exists(doc_file_path):
            save_task = await asyncio.to_thread(
                data_source.save_data,
                doc_file_path=db_datasource.file_path,
                since=since,
                last_update_succeeded_at=get_iso_8601_current_time()
            )
        
    if datasource_type == DataSourceType.CONFLUENCE:
        if utils.is_empty(session_data.token_confluence):
            raise HTTPException(status_code=422, detail=f"Token of Confluence is required for {datasource_type.name} datasource type")
        save_task =  await asyncio.to_thread(
            data_source.update_data, 
            url=db_datasource.url ,
            access_token=session_data.token_confluence,
            space_key=db_datasource.space_key,
            since=since,
            last_update_succeeded_at=get_iso_8601_current_time()
        ) 
        
    if datasource_type == DataSourceType.JIRA:
        if utils.is_empty(db_datasource.token):
            raise HTTPException(status_code=422, detail=f"Token of Jira is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = await asyncio.to_thread(
            data_source.update_data ,
            url=db_datasource.url ,
            access_token=db_datasource.token,
            project_key=db_datasource.project_key,
            start=db_datasource.start, 
            limit=db_datasource.limit,
            since=since,
            last_update_succeeded_at=get_iso_8601_current_time()
        ) 
        
    if datasource_type == DataSourceType.GITLAB:
        if utils.is_empty(session_data.token_gitlab):
            raise HTTPException(status_code=422, detail=f"Token of Gitlab is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = await asyncio.to_thread(
            data_source.update_data,
            url=db_datasource.url ,
            namespace=db_datasource.namespace, 
            project_name = db_datasource.project_name,
            branch=db_datasource.branch,
            private_token=session_data.token_gitlab,
            since=since,
            last_update_succeeded_at=get_iso_8601_current_time()
        ) 
        
    if datasource_type == DataSourceType.GITLAB_DISCUSSION:
        if utils.is_empty(session_data.token_gitlab):
            raise HTTPException(status_code=422, detail=f"Token of Gitlab is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = await asyncio.to_thread(
            data_source.update_data,
            url=db_datasource.url ,
            namespace=db_datasource.namespace, 
            project_name = db_datasource.project_name,
            private_token=session_data.token_gitlab,
            since=since,
            last_update_succeeded_at=get_iso_8601_current_time()
        ) 
        
    if datasource_type == DataSourceType.URL:
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = await asyncio.to_thread(
            data_source.update_data,
            url=db_datasource.url ,
            base_url=db_datasource.base_url,
            max_depth=db_datasource.max_depth ,
            since=since,
            last_update_succeeded_at=get_iso_8601_current_time()
        )
    return save_task
    

@router.put(
    "/stop_datasource/{datasource_id}",
    response_model=models.Datasource,
    dependencies=[Depends(cookie)],
    description="""
   
    """
)
async def stop_datasource_indexing(
    datasource_id: str,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    db_datasource = db.query(database.DataSource).filter(
        database.DataSource.datasource_id == datasource_id,
        database.db_comment_endpoint
    ).first()
    if not db_datasource:
        raise HTTPException(status_code=404, detail="Datasource is not found")
    
    
     # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )
    
    datasource_task = active_tasks.get(datasource_id)
    if datasource_task: 
        asyncio.create_task(data_source.cancel_save_data_task(datasource_task))
        
    db_datasource.status = database.DatasourceDownloadStatus.cancelled
    return db_datasource   
    

@router.post(
    "/preview_datasource",
    # response_model=models.Datasource,
    dependencies=[Depends(cookie)],
    description="""<pre>
    Preview Datasource
    <h3>Request Body:</h3>
    
        - name (str): The name of the data source.
        - description (Optional[str]): A description of the data source.
        - visibility (str): The visibility of the data source: private or public.
        - datasource_type (Optional[str]): The type of the data source (e.g., text, pdf_file, confluence, gitlab, gitlab_discussion, url, doc_file, jira).
        - namespace (Optional[str]): Namespace for GitLab projects.
        - project_name (Optional[str]): Project name in GitLab.
        - branch (Optional[str]): Branch name in GitLab.
        - project_key (Optional[str]): Project key for JIRA.
        - space_key (Optional[str]): Space key for Confluence.
        - start (Optional[int]): Start index for JIRA issues.
        - limit (Optional[int]): Limit for JIRA issues.
        - file_path (Optional[str]): File path for document or PDF.
        - raw_text (Optional[str]): Raw text data.
        - url (Optional[str]): URL for below  data source.
            - url
            - confluence
            - jira
            - gitlab
            - gitlab_discussion
        - base_url (Optional[str]): Base URL for below  data source.
            - url
        - max_depth (Optional[int]): Maximum depth for URL crawling.
        - tag_ids (Optional[List[int]]): A list of tag IDs to associate with the datasource.
        - file: Optional file for document or PDF.
    </pre>
    """
)
async def preview_datasource(
    name: str = Form(..., description="Name of the data source"),
    description: str = Form(..., description="Description of the data source"),
    visibility: str = Form("private", description="Visibility of the data source: private or public"),
    datasource_type: DataSourceType = Form(DataSourceType.TEXT, description="Type of the data source (e.g., text, pdf_file, confluence, gitlab,gitlab_discussion, url, doc_file, jira)"),
    namespace: Optional[str] = Form(None, description="Namespace for GitLab projects"),
    project_name: Optional[str] = Form(None, description="Project name in GitLab"),
    branch: Optional[str] = Form(None, description="Branch name in GitLab"),
    project_key: Optional[str] = Form(None, description="Project key for JIRA"),
    space_key: Optional[str] = Form(None, description="Space key for Confluence"),
    start: Optional[int] = Form(None, description="Start index for JIRA issues"),
    limit: Optional[int] = Form(1000, description="Limit for JIRA issues"),
    token: Optional[str] = Form(None, description="API Token for JIRA"),
    raw_text: Optional[str] = Form(None, description="Raw text data" ),
    url: Optional[str] = Form(None, 
        description="""
            URL for URL data source
            - url
            - confluence
            - jira
            - gitlab
            - gitlab_discussion
        """
    ),
    base_url: Optional[str] = Form(None, description="Base URL for URL data source"),
    max_depth: Optional[int] = Form(None, description="Maximum depth for URL crawling"),
    tag_ids: Optional[str] = Form(None,description="Comma-separated list of tag IDs"),
    # payload: models.DatasourceCreate ,
    file: Annotated[UploadFile ,File(description="File path for document or PDF")] = None,  # File upload
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    
    validate_datasource(datasource_type, namespace, project_name, branch, space_key, project_key, start, limit,token, raw_text, url, base_url, max_depth)
        
    # Create a dictionary of all the parameters
    params = {
        "name": name,
        # "description": description,
        # "visibility": visibility,
        # "datasource_type": datasource_type,
        # "namespace": namespace,
        # "project_name": project_name,
        # "branch": branch,
        # "project_key": project_key,
        # "space_key": space_key,
        # "start": start,
        # "limit": limit,
        # "raw_text": raw_text,
        # "url": url,
        # "base_url": base_url,
        # "max_depth": max_depth,
        # "tag_ids": tag_ids,
    }

    # Filter out None values
    update_data = {key: value for key, value in params.items() if value is not None}


    # Generate the datasource_id using the create_data_source_id function
    datasource_id = create_data_source_id(session_data.nickname, update_data['name'])

    # Check if a DataSource with the same name already exists
    existing_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
        
    info_datasource = None
    if existing_datasource:
        # 파일때문에 기존의 path를 그대로 사용하기 위해서.
        info_datasource = models.Datasource.from_orm(existing_datasource)
    else:
        # create_user_info = models.User.from_orm(session_data)
        db_user = db.query(database.User).filter(database.db_comment_endpoint).filter(database.User.user_id == session_data.user_id).first()
        create_user_info = models.User.from_orm(db_user)
        info_datasource = models.Datasource(
            datasource_id=datasource_id,
            name=name,  # Set default or required values here
            description=description,  # You can customize this
            created_at=datetime.now(database.KST),
            updated_at=datetime.now(database.KST),
            tags = [],
            embeddings = [],
            create_user_info = create_user_info
        )
    
    info_datasource.datasource_type=datasource_type.value
    if not utils.is_empty(namespace) :
        info_datasource.namespace=namespace
    if not utils.is_empty(project_name):
        info_datasource.project_name=project_name
    if not utils.is_empty(branch):
        info_datasource.branch=branch
    if not utils.is_empty(project_key):
        info_datasource.project_key=project_key
    if not utils.is_empty(start):
        info_datasource.start=start
    if not utils.is_empty(limit):
        info_datasource.limit=limit
    if not utils.is_empty(token):
        info_datasource.token=token
    if not utils.is_empty(space_key):
        info_datasource.space_key=space_key
    if not utils.is_empty(raw_text):
        info_datasource.raw_text=raw_text
    if not utils.is_empty(url): 
        info_datasource.url=url
    if not utils.is_empty(base_url):
        info_datasource.base_url=base_url
    if not utils.is_empty(max_depth): 
        info_datasource.max_depth=max_depth
    

    # text, pdf_file, confluence, gitlab, url, doc_file, jira
    if file:
        if datasource_type in (DataSourceType.PDF_FILE ,DataSourceType.DOC_FILE):
            # Use the utility function to construct the file save path
            file_save_path = construct_file_save_path(datasource_id=datasource_id,file_name=file.filename)
            dir_save_path = os.path.dirname(file_save_path)
            os.makedirs(dir_save_path, exist_ok=True)
            # Save the file asynchronously using aiofiles
            async with aiofiles.open(file_save_path, 'wb') as out_file:
                content = await file.read()  # Read file content asynchronously
                await out_file.write(content)  # Write the content asynchronously
            info_datasource.file_path = file_save_path
    
    preview_data = preview_datasource_sub(datasource_type,info_datasource, session_data )
        
    return preview_data

def preview_datasource_sub(datasource_type,info_datasource, session_data ):
    # 1. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=info_datasource.name,
        created_by=session_data.nickname,
        description=info_datasource.description,
        data_source_type=info_datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )   
    
    if datasource_type == DataSourceType.TEXT:
        preview_data = data_source.load_preview_data(raw_text=[info_datasource.raw_text])
        
    if datasource_type == DataSourceType.PDF_FILE:
        preview_data = data_source.load_preview_data(pdf_file_path=info_datasource.file_path)
        
    if datasource_type == DataSourceType.DOC_FILE:
        preview_data = data_source.load_preview_data(
        doc_file_path=info_datasource.file_path)
    
    if datasource_type == DataSourceType.CONFLUENCE:
        if utils.is_empty(session_data.token_confluence):
            raise HTTPException(status_code=422, detail=f"Token of Confluence is required for {datasource_type.name} datasource type")
        
        preview_data = data_source.load_preview_data(url=info_datasource.url, access_token=session_data.token_confluence, space_key=info_datasource.space_key)
    
    if datasource_type == DataSourceType.JIRA:
        if utils.is_empty(info_datasource.token):
            raise HTTPException(status_code=422, detail=f"Token of Jira is required for {datasource_type.name} datasource type")
        preview_data = data_source.load_preview_data(url=info_datasource.url, access_token=info_datasource.token, project_key=info_datasource.project_key, start=info_datasource.start)
        
    if datasource_type == DataSourceType.GITLAB:
        if utils.is_empty(session_data.token_gitlab):
            raise HTTPException(status_code=422, detail=f"Token of Gitlab is required for {datasource_type.name} datasource type")   
        
        preview_data = data_source.load_preview_data(
            url=info_datasource.url,
            namespace=info_datasource.namespace,
            project_name=info_datasource.project_name,
            branch=info_datasource.branch,
            private_token=session_data.token_gitlab
        )
        
    if datasource_type == DataSourceType.GITLAB_DISCUSSION:
        if utils.is_empty(session_data.token_gitlab):
            raise HTTPException(status_code=422, detail=f"Token of Gitlab is required for {datasource_type.name} datasource type")
        
        # to do
    
    if datasource_type == DataSourceType.URL:
        preview_data = data_source.load_preview_data(url=info_datasource.url, base_url=info_datasource.base_url)
    return preview_data


@router.delete(
    "/{datasource_id}",
    dependencies=[Depends(cookie)],
    description="Delete a data source by its ID.",
)
async def delete_datasource(
    datasource_id: str,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    logger.info(f"Start Delete datasource : {datasource_id}") 
    # Query the database to find the data source by its ID
    db_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()

    # If the data source doesn't exist, raise a 404 error
    if not db_datasource:
        raise HTTPException(status_code=404, detail="Data source not found")


    # 물리적으로 데이타소스를 삭제한다.    
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )
    for db_embedding in db_datasource.embeddings:
        db_llm_api = db_embedding.llm_api
        
        collection_name = create_collection_name(data_source.id, db_embedding.embedding_model)
        collection = data_source.add_collection(
            collection_name=collection_name,
            llm_api_provider=LlmApiProvider(db_llm_api.llm_api_provider),
            llm_api_key=db_llm_api.llm_api_key,
            llm_api_url=db_llm_api.llm_api_url,
            llm_embedding_model_name=db_embedding.embedding_model,
            last_update_succeeded_at=db_embedding.success_at
        )
        logger.info(f"Delete embedding. colletion name: {collection_name}") 
        collection.delete_collection()
    logger.info(f"Delete Datasource  datasouce id: {data_source.id}") 
    data_source.delete_data()
    
    
    # Delete the data source from the database
    db_datasource.tags = []
    db.delete(db_datasource)

    return {"detail": f"Data source {datasource_id} has been deleted successfully."}

@router.post(
    "/embedding",
    response_model=models.Embedding,
    dependencies=[Depends(cookie)],
    description="""
    Creates a new embedding for a given data source and stores it in ChromaDB.

    This endpoint performs the following:
    
    1. **Data Retrieval**: The data is retrieved from the specified data source based on its type (e.g., text, PDF, DOC, URL).
    
    2. **Embedding Process**: The documents from the data source are split into chunks using the specified text splitter and embedded using the selected LLM model. The embeddings are then stored in ChromaDB.

    3. **Status Tracking**: The status of the embedding process is tracked, and the result is returned once the task is completed. The status can be "updating", "updated", or "failed".

    ### Request Body (Models: EmbeddingCreate)
    
    - **datasource_id**: The unique identifier of the data source.
    - **embedding_model**: The name of the LLM embedding model to use for generating embeddings.
    - **splitter** (Optional): The type of text splitter to use for dividing the documents into chunks (e.g., RecursiveCharacterTextSplitter, CharacterTextSplitter).
    - **chunk_size** (Optional): The size of the chunks for text splitting (only required for certain splitter types).
    - **chunk_overlap** (Optional): The overlap size between consecutive chunks (only required for certain splitter types).
    - **separator** (Optional): The separator to use for splitting text (only required for CharacterTextSplitter).
    - **is_separator_regex** (Optional): Boolean indicating whether the separator is a regex (only required for CharacterTextSplitter).
    - **tag** (Optional): Tag to use for certain splitter types (e.g., HTMLHeaderTextSplitter, MarkdownHeaderTextSplitter).
    - **language** (Optional): Language information required for RecursiveCharacterTextSplitter.
    - **max_chunk_size** (Optional): Maximum chunk size for RecursiveJsonSplitter.
    
    ### Steps in the Process:
    
    1. The data source is retrieved using the `datasource_id`. The types of data sources can be:
       - **TEXT**: Plain text data.
       - **PDF_FILE**: PDF documents.
       - **DOC_FILE**: DOC documents.
       - **URL**: Data fetched from a URL.
       - **CONFLUENCE**: Data retrieved from Confluence.
       - **JIRA**: Data retrieved from Jira.
       - **GITLAB**: Data from GitLab projects.
       - **GITLAB_DISCUSSION**: GitLab discussion data.
    
    2. Once the data is retrieved, it is saved and embedded into ChromaDB.
    
    3. The embeddings are generated using the specified LLM model, and the process is tracked with status updates.

    ### Response:
    
    The API returns the created embedding information, including the status, embedding ID, and timestamps for the process (started, completed, and success timestamps).
    
    ### Example Request:
    
    ```json
    {
      "datasource_id": "ds-example",
      "embedding_model": "text-embedding-ada-002",
      "splitter": "CharacterTextSplitter",
      "chunk_size": 1000,
      "chunk_overlap": 50,
      "separator": "\\n",
      "is_separator_regex": false
    }
    ```
    """
)
async def create_embedding(
    payload: models.EmbeddingCreate,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    logger.info(f"Start embedding create. datasource_id : {payload.datasource_id}")
    
    global active_tasks
    
    payload_data = payload.dict(exclude_unset=True)

    # Retrieve the data source
    db_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == payload.datasource_id).first()
    if not db_datasource:
        raise HTTPException(status_code=404, detail="DataSource not found")

    # Check if the data source status is 'downloaded'
    if db_datasource.status != "downloaded":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot create embedding. DataSource status must be 'downloaded', but found '{db_datasource.status}'."
        )
        
    # Retrieve LLM API configuration based on the llm_api_id
    # Here you can use your models/logic to get API config details based on llm_api_id
    # Example API data retrieval
    db_llm_api = db.query(database.LlmApi).filter(database.db_comment_endpoint).filter(database.LlmApi.llm_api_id == payload.llm_api_id).first()
    if not db_llm_api:
        raise HTTPException(status_code=404, detail="LLM API configuration not found")
    
    
    data_source_type = DataSourceType(db_datasource.datasource_type)
    # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )
    
    # 3. 데이터 소스에 컬렉션 추가
    collection_name = create_collection_name(data_source.id, payload.embedding_model)
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider(db_llm_api.llm_api_provider),
        llm_api_key=db_llm_api.llm_api_key,
        llm_api_url=db_llm_api.llm_api_url,
        llm_embedding_model_name=payload.embedding_model,
        last_update_succeeded_at=iso_8601_str_to_datetime(base_succeeded_at)
    )
    
    data_size = 0
    

    # 5. 데이터 임베딩 및 ChromaDB에 추가
    # Create a new embedding entry in the database
    
    db_embedding = database.Embedding(
        # datasource_id=payload.datasource_id,
        embedding_id=collection_name,
        status="updating",
        started_at = datetime.now(database.KST),
        data_size = data_size
    )
    for key, value in payload_data.items():
        if key not in ["xxxxxxxxxxx"]:
            if key in ["splitter"]:
                setattr(db_embedding, key, value.value)
            else : 
                setattr(db_embedding, key, value)
    
    db.add(db_embedding)
    # db.flush()
    # db.refresh(db_embedding)

    

    async def  execute_embedding():
        logger.info(f"Data embedding create task started : {collection_name}")
        data = await data_source.read_data()
    
        # Filter out keys from payload_data
        splitter_payload_data = {key: value for key, value in payload_data.items() if value is not None and key not in [
            # 'splitter', 
            # 'chunk_size', 
            # 'chunk_overlap',
            'language'
        ]}
        
        last_update_succeeded_at = get_iso_8601_current_time()
        if 'splitter' in splitter_payload_data and not utils.is_empty(splitter_payload_data['splitter']):
            spliter_type = SplitterType(payload.splitter)
            # if spliter_type in [SplitterType.RecursiveCharacterTextSplitter] : 
            if 'language' in payload_data and hasattr(Language,payload_data['language']):
                splitter_payload_data['language'] = Language(payload_data['language'])

            splitter = create_splitter(spliter_type, **splitter_payload_data)
            splitted_documents: Iterable[Document] = split_texts(documents=data, splitter=splitter)
            embed_task = asyncio.create_task(
                collection.embed_documents_and_overwrite_to_vectorstore(
                    documents=splitted_documents,
                    last_update_succeeded_at=last_update_succeeded_at
                )
            )
        else:
            embed_task = asyncio.create_task(
                collection.embed_documents_and_overwrite_to_vectorstore(
                    documents=data,
                    last_update_succeeded_at=last_update_succeeded_at
                )
            )
        
        if embed_task :  
            embed_task.add_done_callback(lambda future: embed_callback(future))
            active_tasks[collection_name] = embed_task
            
        # return embed_task
          
    
    def embed_callback(future):
        logger.info(f"Data embedding create completed : {collection_name}") 
        logger.info(f"Data embedding create calback started")
        db_callback = database.SessionLocal()
        try:
            query_embedding = db_callback.query(database.Embedding).filter(database.db_comment_endpoint)
            query_embedding = query_embedding.filter(database.Embedding.datasource_id == payload.datasource_id)
            query_embedding = query_embedding.filter(database.Embedding.embedding_id == collection_name)
            db_embedding = query_embedding.first()
            if future.exception():
                logger.info(f"Embedding task failed: {str(future.exception())}")
                db_embedding.status = database.EmbeddingStatus.failed
                db_embedding.status_message = str(future.exception())
                # db_embedding.status_message = traceback.format_exc()
            else:
                embeded, total = future.result()
                logger.info(f"Embedding task completed. Number of chunks embedded {str(embeded)} / Total {str(total)}: ")
                # logger.info("Embedding task completed. Number of chunks embedded / Total: ", str(embeded), " / ", str(total))
                collection.last_update_succeeded_at = iso_8601_str_to_datetime(get_iso_8601_current_time())
                db_embedding.success_at = datetime.now(database.KST)
                db_embedding.status=database.EmbeddingStatus.updated
                
                # embedding volume
                pattern = f"{db_embedding.embedding_id}*"
                # response_json 은 store.size = '185.1mb'  처럼 나온다.
                response_json = data_source.opensearch_client.cat.indices(index=pattern, format="json")
                # response 에서는 좀 계산을 해 줘야 한다.
                response = data_source.opensearch_client.indices.get(index=pattern)
                index_names = list(response.keys())
                for index_name in index_names:
                    # Get index stats
                    stats = data_source.opensearch_client.indices.stats(index=index_name)
                    
                    total_size_in_bytes = stats['indices'][index_name]['total']['store']['size_in_bytes']
                    total_size_in_mb = total_size_in_bytes / (1024 * 1024)  # Convert to MB
                    db_embedding.data_size = total_size_in_bytes
                    logger.info(f"Total size of index '{index_name}': {total_size_in_bytes} bytes ({total_size_in_mb:.2f} MB)")
            
        except Exception as e:
            db_embedding.status = database.EmbeddingStatus.failed
            db_embedding.status_message = str(e)
            # db_embedding.status_message = traceback.format_exc()
            logging.error(f"Embedding index failed: {e}")
            
        except asyncio.exceptions.CancelledError as e:
            db_embedding.status = database.EmbeddingStatus.cancelled
            db_embedding.status_message = str(e)
            # db_embedding.status_message = traceback.format_exc()
            logging.error(f"Embedding index cancelled: {e}")  
            
        finally:
            db_embedding.completed_at = datetime.now(database.KST)
            db_callback.commit()
            
            update_message = {
                "type": "embedding_update",
                "data": models.Embedding.from_orm(db_embedding).dict()
            }
            asyncio.create_task(broadcast_update(update_message))
            # active_tasks 해제
            if collection_name in active_tasks:
                active_tasks.pop(collection_name,None)
            db_callback.close()  
            logger.info(f"Data embedding create calback completed")
    
    
    #  before start task of opensearch , apply to db. if there is error , do not  execute_save_data    
    db_embedding.status = database.EmbeddingStatus.updating
    db.commit()
    asyncio.create_task(execute_embedding())    
    return db_embedding



@router.put(
    "/{datasource_id}/{embedding_id}",
    response_model=models.Embedding,
    dependencies=[Depends(cookie)],
    description="""
    사용안함
    """
)
async def update_embedding(
    datasource_id: str,
    embedding_id : str,
    payload: models.EmbeddingUpdate,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    payload_data = payload.dict(exclude_unset=True)
    
    db_embedding = db.query(database.Embedding).filter(
        database.Embedding.datasource_id == datasource_id, 
        database.Embedding.embedding_id == embedding_id,
        database.db_comment_endpoint
    ).first()
    if not db_embedding:
        raise HTTPException(status_code=404, detail="Embedding is not found")
   
    for key, value in payload_data.items():
        if key not in ["xxxxxxxxxxx"]:
            if key in ["splitter"]:
                setattr(db_embedding, key, value.value)
            else : 
                setattr(db_embedding, key, value)

    

    # Retrieve the data source
    db_datasource = db_embedding.datasource

    # Retrieve LLM API configuration based on the llm_api_id
    # Here you can use your models/logic to get API config details based on llm_api_id
    db_llm_api = db.query(database.LlmApi).filter(database.db_comment_endpoint).filter(database.LlmApi.llm_api_id == db_embedding.llm_api_id).first()
    if not db_llm_api:
        raise HTTPException(status_code=404, detail="LLM API configuration not found")
    
    
    data_source_type = DataSourceType(db_datasource.datasource_type)
    # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )
    # 3. 데이터 소스에 컬렉션 추가
    collection_name = create_collection_name(data_source.id, db_embedding.embedding_model)
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider(db_llm_api.llm_api_provider),
        llm_api_key=db_llm_api.llm_api_key,
        llm_api_url=db_llm_api.llm_api_url,
        llm_embedding_model_name=payload.embedding_model
    )
    # collection = data_source.collections[collection_name]
    # db_embedding.embedding_id = collection_name
    collection.last_update_succeeded_at = iso_8601_str_to_datetime(get_iso_8601_current_time())
    
    db.flush()
    


    # 4. 데이터를 텍스트 파일로 저장
    data = data_source.read_data()
    # Filter out keys from payload_data
    # 이 부분을 db 부분으로 바꾼다.
    splitter_payload_data = {key: value for key, value in payload_data.items() if value is not None and key not in [
        # 'splitter', 
        # 'chunk_size', 
        # 'chunk_overlap',
        'language'
    ]}

    if 'splitter' in splitter_payload_data and not utils.is_empty(splitter_payload_data['splitter']):
        spliter_type = SplitterType(payload.splitter)
        # if spliter_type in [SplitterType.RecursiveCharacterTextSplitter] : 
        if 'language' in payload_data and hasattr(Language,payload_data['language']):
            splitter_payload_data['language'] = Language(payload_data['language'])

        splitter = create_splitter(spliter_type, **splitter_payload_data)
        splitted_documents: Iterable[Document] = split_texts(data, splitter)
        embed_task = asyncio.create_task(collection.embed_documents_and_overwrite_to_chromadb(documents=splitted_documents))
    else:
        embed_task = asyncio.create_task(collection.embed_documents_and_overwrite_to_chromadb(documents=data))
        
    def embed_callback(future):
        if future.exception():
            logger.info("Embedding task failed: ", future.exception())
            logger.info("Update embedding state to failed")
        else:
            embeded, total = future.result()
            logger.info("Embedding task completed. Number of chunks embedded / Total: ", str(embeded), " / ", str(total))
            collection.last_update_succeeded_at = datetime.now(database.KST)
            db_embedding.success_at = datetime.now(database.KST)
            db_embedding.completed_at = datetime.now(database.KST)
            db_embedding.status="updated"
            
    embed_task.add_done_callback(lambda future: embed_callback(future))

    await embed_task
    # End embedding

    db.flush()
    db.refresh(db_embedding)
    return db_embedding



@router.put(
    "/reembeding/{datasource_id}/{embedding_id}",
    response_model=models.Embedding,
    dependencies=[Depends(cookie)],
    description="""
   
    """
)
async def embedding_reindex(
    datasource_id: str,
    embedding_id : str,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    
    db_embedding = db.query(database.Embedding).filter(
        database.Embedding.datasource_id == datasource_id, 
        database.Embedding.embedding_id == embedding_id,
        database.db_comment_endpoint
    ).first()
    if not db_embedding:
        raise HTTPException(status_code=404, detail="Embedding is not found")
   
    payload_data = models.Embedding.from_orm(db_embedding).dict()
    # Retrieve the data source
    db_datasource = db_embedding.datasource
    
        # Check if the data source status is 'downloaded'
    if db_datasource.status != "downloaded":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reindex embedding. DataSource status must be 'downloaded', but found '{db_datasource.status}'."
        )

    # Retrieve LLM API configuration based on the llm_api_id
    # Here you can use your models/logic to get API config details based on llm_api_id
    db_llm_api = db.query(database.LlmApi).filter(database.db_comment_endpoint).filter(database.LlmApi.llm_api_id == db_embedding.llm_api_id).first()
    if not db_llm_api:
        raise HTTPException(status_code=404, detail="LLM API configuration not found")
    
    
    data_source_type = DataSourceType(db_datasource.datasource_type)
    # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )
    
    # 3. 데이터 소스에 컬렉션 추가
    collection_name = create_collection_name(data_source.id, db_embedding.embedding_model)
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider(db_llm_api.llm_api_provider),
        llm_api_key=db_llm_api.llm_api_key,
        llm_api_url=db_llm_api.llm_api_url,
        llm_embedding_model_name=db_embedding.embedding_model,
        last_update_succeeded_at=iso_8601_str_to_datetime(get_iso_8601_current_time())
    )
    
    
    # # 실제인덱스를 찾는다. 여기서 해당 index는 하나만 나와야 한다.
    # op_client = collection.vectorstore.client
    # wildcard_name = f"{collection_name}*"
    # co_response = op_client.cat.indices(index=wildcard_name, format="json")
    # # 실제 index 로 바꿔준다.
    # collection.vectorstore.index_name = co_response[0]['index']
    
    db.flush()

    async def  execute_embedding():
        logger.info(f"Data embedding reindex task started : {collection_name}")
        data = await data_source.read_data()
        db_embedding_dict = models.Embedding.from_orm(db_embedding).dict()
        splitter_payload_data = {key: value for key, value in db_embedding_dict.items() if value is not None and key not in [
            # 'splitter', 
            # 'chunk_size', 
            # 'chunk_overlap',
            'language'
        ]}
        
        last_update_succeeded_at = get_iso_8601_current_time()
        if 'splitter' in splitter_payload_data and not utils.is_empty(splitter_payload_data['splitter']):
            spliter_type = SplitterType(db_embedding.splitter)
            # if spliter_type in [SplitterType.RecursiveCharacterTextSplitter] : 
            if 'language' in payload_data and hasattr(Language,payload_data['language']):
                splitter_payload_data['language'] = Language(payload_data['language'])

            splitter = create_splitter(spliter_type, **splitter_payload_data)
            splitted_documents: Iterable[Document] = split_texts(documents=data, splitter=splitter)
            embed_task = asyncio.create_task(
                collection.update(
                    documents=splitted_documents,
                    last_update_succeeded_at=last_update_succeeded_at,
                    data_source_type=data_source.data_source_type
                )
            )
        else:
            embed_task = asyncio.create_task(
                collection.update(
                    documents=data,
                    last_update_succeeded_at=last_update_succeeded_at,
                    data_source_type=data_source.data_source_type
                )
            )
        
        if embed_task :        
            embed_task.add_done_callback(lambda future: embed_callback(future))
            active_tasks[collection_name] = embed_task
        # return embed_task
          
    
    def embed_callback(future):
        logger.info(f"Data embedding reindex completed : {collection_name}") 
        logger.info(f"Data embedding reindex calback Started")
        db_callback = database.SessionLocal()
        try:
            query_embedding = db_callback.query(database.Embedding).filter(database.db_comment_endpoint).filter(database.db_comment_endpoint)
            query_embedding = query_embedding.filter(database.Embedding.datasource_id == datasource_id)
            query_embedding = query_embedding.filter(database.Embedding.embedding_id == collection_name)
            db_embedding = query_embedding.first()
            if future.exception():
                logger.info(f"Embedding task failed: {str(future.exception())}")
                db_embedding.status = database.EmbeddingStatus.failed
                db_embedding.status_message = str(future.exception())
            else:
                embeded, total = future.result()
                logger.info(f"Embedding task completed. Number of chunks embedded {str(embeded)} / Total {str(total)}: ")
                # logger.info("Embedding task completed. Number of chunks embedded / Total: ", str(embeded), " / ", str(total))
                collection.last_update_succeeded_at = iso_8601_str_to_datetime(get_iso_8601_current_time())
                db_embedding.success_at = datetime.now(database.KST)
                db_embedding.status=database.EmbeddingStatus.updated
                
                # embedding volume
                pattern = f"{db_embedding.embedding_id}*"
                # response_json 은 store.size = '185.1mb'  처럼 나온다.
                response_json = data_source.opensearch_client.cat.indices(index=pattern, format="json")
                # response 에서는 좀 계산을 해 줘야 한다.
                response = data_source.opensearch_client.indices.get(index=pattern)
                index_names = list(response.keys())
                for index_name in index_names:
                    # Get index stats
                    stats = data_source.opensearch_client.indices.stats(index=index_name)
                    
                    total_size_in_bytes = stats['indices'][index_name]['total']['store']['size_in_bytes']
                    total_size_in_mb = total_size_in_bytes / (1024 * 1024)  # Convert to MB
                    db_embedding.data_size = total_size_in_bytes
                    logger.info(f"Total size of index '{index_name}': {total_size_in_bytes} bytes ({total_size_in_mb:.2f} MB)")
            
        except Exception as e:
            db_embedding.status = database.EmbeddingStatus.failed
            db_embedding.status_message = str(e)
            # db_embedding.status_message = traceback.format_exc()
            logging.error(f"Embedding reindex failed: {e}")
            
        except asyncio.exceptions.CancelledError as e:
            db_embedding.status = database.EmbeddingStatus.cancelled
            db_embedding.status_message = str(e)
            # db_embedding.status_message = traceback.format_exc()
            logging.error(f"Embedding reindex cancelled: {e}")
            
        finally:
            db_embedding.completed_at = datetime.now(database.KST)
            db_callback.commit()
            
            update_message = {
                "type": "embedding_update",
                "data": models.Embedding.from_orm(db_embedding).dict()
            }
            asyncio.create_task(broadcast_update(update_message))
            # active_tasks 해제
            if collection_name in active_tasks:
                active_tasks.pop(collection_name,None)
            db_callback.close()  
            logger.info(f"Data embedding reindex calback completed")
            
    
    
    #  before start task of opensearch , apply to db. if there is error , do not  execute_save_data    
    db_embedding.status = database.EmbeddingStatus.updating
    asyncio.create_task(execute_embedding())
    db.commit()
    return db_embedding


@router.put(
    "/stop_embeding/{datasource_id}/{embedding_id}",
    response_model=models.Embedding,
    dependencies=[Depends(cookie)],
    description="""
   
    """
)
async def stop_embeding_indexing(
    datasource_id: str,
    embedding_id : str,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    db_embedding = db.query(database.Embedding).filter(
        database.Embedding.datasource_id == datasource_id, 
        database.Embedding.embedding_id == embedding_id,
        database.db_comment_endpoint
    ).first()
    if not db_embedding:
        raise HTTPException(status_code=404, detail="Embedding is not found")

    payload_data = models.Embedding.from_orm(db_embedding).dict()
    # Retrieve the data source
    db_datasource = db_embedding.datasource
    
    # Retrieve LLM API configuration based on the llm_api_id
    # Here you can use your models/logic to get API config details based on llm_api_id
    db_llm_api = db.query(database.LlmApi).filter(database.db_comment_endpoint).filter(database.LlmApi.llm_api_id == db_embedding.llm_api_id).first()
    if not db_llm_api:
        raise HTTPException(status_code=404, detail="LLM API configuration not found")
    
    
    data_source_type = DataSourceType(db_datasource.datasource_type)
    # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )
    
    # 3. 데이터 소스에 컬렉션 추가
    collection_name = create_collection_name(data_source.id, db_embedding.embedding_model)
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider(db_llm_api.llm_api_provider),
        llm_api_key=db_llm_api.llm_api_key,
        llm_api_url=db_llm_api.llm_api_url,
        llm_embedding_model_name=db_embedding.embedding_model,
        last_update_succeeded_at=iso_8601_str_to_datetime(get_iso_8601_current_time())
    )
    
    # # 작을 취소하려고. 그러나 작업을 찾을 수 없음
    # op_client = collection.vectorstore.client
    # target_index = collection_name
    # tasks = op_client.tasks.list(detailed=True)
    
    # index_tasks = [
    #     task_id 
    #     for node_id, node_data in tasks['nodes'].items()
    #     for task_id, task_data in node_data['tasks'].items()
    #     if target_index in task_data.get('description', '')
    # ]
    
    # for task_id in index_tasks:
    #     cancel_response = op_client.tasks.cancel(task_id=task_id)
    
    
    
    # find asyncio task
    embedding_task =  active_tasks.get(collection_name)
    if embedding_task: 
        # asyncio.create_task(collection.cancel_embedding_task(embedding_task))
        await collection.cancel_embedding_task(embedding_task)
        
    return db_embedding   
        
    
@router.post(
    "/preview_embedding",
    # response_model=models.Embedding,
    dependencies=[Depends(cookie)]
)
async def preview_embedding(
    payload: models.EmbeddingPreview,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    # Example to show how you can process the data


    payload_data = payload.dict(exclude_unset=True)

    # Retrieve the data source
    db_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == payload.datasource_id).first()
    if not db_datasource:
        raise HTTPException(status_code=404, detail="DataSource not found")
   
    # 1. 데이터 소스 생성
    datasource_type = DataSourceType(db_datasource.datasource_type)
    info_datasource = models.Datasource.from_orm(db_datasource)
    
    preview_data = preview_datasource_sub(
        datasource_type=datasource_type,
        info_datasource=info_datasource,
        session_data=session_data
    )
    
    async def read_data(preview_data) -> AsyncIterable[Document]:
        # 텍스트 파일로 저장된 데이터를 불러옵니다.
        yield Document(content=preview_data, metadata="")
    
    preview_data_i = read_data(preview_data)
    # Filter out keys from payload_data
    splitter_payload_data = {key: value for key, value in payload_data.items() if value is not None and key not in [
        # 'splitter', 
        # 'chunk_size', 
        # 'chunk_overlap',
        'language'
    ]}

    spliter_type = SplitterType(payload.splitter)
    # if spliter_type in [SplitterType.RecursiveCharacterTextSplitter] : 
    if 'language' in payload_data and hasattr(Language,payload_data['language']):
        splitter_payload_data['language'] = Language(payload_data['language'])

    splitter = create_splitter(spliter_type, **splitter_payload_data)
    splitted_documents = split_texts(preview_data_i, splitter)
        
    result: List[dict] = []
    async for document in splitted_documents:
        result.append({
            "content": document.content,
            "metadata": document.metadata
        })

    # Return the list of dictionaries, which is FastAPI-compatible
    return result

@router.post(
    "/similarity_search",
    dependencies=[Depends(cookie)],
    description="""
        쿼리테스트
    """
)
async def similarity_search(
    payload: models.EmbeddingSimilaritySearch,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    # Example to show how you can process the data


    payload_data = payload.dict(exclude_unset=True)

    # Retrieve the data source
    db_embedding = db.query(database.Embedding).filter(database.db_comment_endpoint).filter(database.Embedding.embedding_id == payload.embedding_id).first()
    if not db_embedding:
        raise HTTPException(status_code=404, detail="Embedding is not found")

    
    data_source_type = DataSourceType(db_embedding.datasource.datasource_type)
    # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=db_embedding.datasource.name,
        created_by=db_embedding.datasource.create_user_info.nickname,
        description=db_embedding.datasource.description,
        data_source_type=db_embedding.datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )
    # 3. 데이터 소스에 컬렉션 추가

    collection_name = create_collection_name(db_embedding.datasource.datasource_id, db_embedding.embedding_model)
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider(db_embedding.llm_api.llm_api_provider),
        llm_api_key=db_embedding.llm_api.llm_api_key,
        llm_api_url=db_embedding.llm_api.llm_api_url,
        llm_embedding_model_name=db_embedding.embedding_model)
    
    # # 실제인덱스를 찾는다. 여기서 해당 index는 하나만 나와야 한다.
    # op_client = collection.vectorstore.client
    # wildcard_name = f"{collection_name}*"
    # co_response = op_client.cat.indices(index=wildcard_name, format="json")
    # # 실제 index 로 바꿔준다.
    # collection.vectorstore.index_name = co_response[0]['index']
    
    query_results = collection.similarity_search(query=payload.query)
    return query_results

@router.get(
    "/{datasource_id}",
    response_model=models.Datasource,
    dependencies=[Depends(cookie)],
    summary="Get a specific data source by ID",
    description="Retrieve detailed information about a specific data source using its ID."
)
async def get_datasource(datasource_id: str, db: Session = Depends(get_db)):
    datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
    if not datasource:
        raise HTTPException(status_code=404, detail="DataSource not found")
    return datasource


@router.post(
    "/datasource_search", 
    response_model=models.DataSourceSearchResponse, 
    dependencies=[Depends(cookie)],
    description="Search data sources with various criteria"
)
async def search_datasources(
    search: models.DataSourceSearch,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(database.DataSource)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    
    # Apply filters based on the search criteria
    search_exclude = search.dict(exclude_unset=True)
    
    # Filter by search words
    if 'search_words' in search_exclude and not utils.is_empty(search_exclude['search_words']):
        search_pattern = f"%{search_exclude['search_words']}%"
        if 'search_scope' in search_exclude:
            filters = []
            if "name" in search_exclude['search_scope']:
                filters.append(database.DataSource.name.like(search_pattern))
            if "description" in search_exclude['search_scope']:
                filters.append(database.DataSource.description.like(search_pattern))
            query = query.filter(or_(*filters))
        else:
            query = query.filter(
                or_(
                    database.DataSource.name.like(search_pattern),
                    database.DataSource.description.like(search_pattern)
                )
            )
    
    # Filter by visibility
    if 'visibility' in search_exclude and len(search_exclude['visibility']) > 0:
        visibility_values = [v.value for v in search_exclude['visibility']]
        query = query.filter(database.DataSource.visibility.in_(visibility_values))
    
    # Filter by tags
    if 'tag_ids' in search_exclude and len(search_exclude['tag_ids']) > 0:
        query = query.filter(database.DataSource.tags.any(database.Tag.tag_id.in_(search_exclude['tag_ids'])))
    
    # Filter by users
    user_filter_basic = [
        database.DataSource.create_user == session_data.user_id ,
        database.DataSource.visibility == 'public'
    ]
    query = query.filter(or_(*user_filter_basic))
    if 'user_list' in search_exclude and len(search_exclude['user_list']) > 0:
        query = query.filter(database.DataSource.create_user.in_(search_exclude['user_list']))

    # Filter by data source types (multi-select)
    if 'datasource_types' in search_exclude and len(search_exclude['datasource_types']) > 0:
        # Extract the string values from the Enum
        datasource_type_values = [ds_type.value for ds_type in search_exclude['datasource_types']]
        query = query.filter(database.DataSource.datasource_type.in_(datasource_type_values))
        
    total_count = query.count()
    
    
    

    # Pagination
    query = query.order_by(database.DataSource.updated_at.desc(),database.DataSource.created_at.desc())  
    if 'skip' in search_exclude and 'limit' in search_exclude:
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit']) 
    
    query = query.options(
        selectinload(database.DataSource.tags),
        selectinload(database.DataSource.embeddings),
        selectinload(database.DataSource.create_user_info)
    )
    
    datasources = query.all()
    
    return models.DataSourceSearchResponse(
        total_count=total_count,
        list=datasources
    )
    
    
@router.post(
    "/embedding_search", 
    response_model=models.EmbeddingSearchResponse, 
    dependencies=[Depends(cookie)],
    description="Search embeddings with pagination")
async def search_embeddings(
    search: models.EmbeddingSearch,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(database.Embedding).filter(database.db_comment_endpoint)
    
    search_exclude = search.dict(exclude_unset=True)
    query = query.filter(database.Embedding.datasource_id == search.datasource_id)

    # Get the total count before pagination
    total_count = query.count()

    # Apply pagination
    query = query.order_by(database.Embedding.last_update_time.desc(),database.Embedding.started_at.desc())  
    if 'skip' in search_exclude and 'limit' in search_exclude:
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit']) 
    
    # Fetch the results
    embeddings = query.all()

    return models.EmbeddingSearchResponse(
        total_count=total_count,
        list=embeddings
    )
    
    
@router.get("/{datasource_id}/{embedding_id}", response_model=models.Embedding)
async def get_embedding_info(
    datasource_id: str,
    embedding_id: str,
    db: Session = Depends(get_db)
):
    # Fetch the datasource to ensure it exists
    db_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
    if not db_datasource:
        raise HTTPException(status_code=404, detail="Datasource not found")
    
    # Fetch the embedding
    db_embedding = db.query(database.Embedding).filter(
        database.Embedding.datasource_id == datasource_id, 
        database.Embedding.embedding_id == embedding_id,
        database.db_comment_endpoint
    ).first()
    
    if not db_embedding:
        raise HTTPException(status_code=404, detail="Embedding not found")
    
    # Return the embedding information
    return db_embedding



@router.delete("/{datasource_id}/{embedding_id}", response_model=models.Embedding)
async def delete_embedding(
    datasource_id: str,
    embedding_id: str,
    db: Session = Depends(get_db)
):
    # Verify the embedding exists
    db_embedding = db.query(database.Embedding).filter(
        database.Embedding.datasource_id == datasource_id,
        database.Embedding.embedding_id == embedding_id,
        database.db_comment_endpoint
    ).first()
    
    if not db_embedding:
        raise HTTPException(status_code=404, detail="Embedding not found")
    
    db_datasource = db_embedding.datasource
    db_llm_api = db_embedding.llm_api
    
    # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type,
        opensearch_hosts=opensearch_hosts,
        opensearch_auth=opensearch_auth
    )
    
    # 3. 데이터 소스에 컬렉션 추가
    collection_name = create_collection_name(data_source.id, db_embedding.embedding_model)
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider(db_llm_api.llm_api_provider),
        llm_api_key=db_llm_api.llm_api_key,
        llm_api_url=db_llm_api.llm_api_url,
        llm_embedding_model_name=db_embedding.embedding_model,
        last_update_succeeded_at=iso_8601_str_to_datetime(get_iso_8601_current_time())
    )    
    # # 실제인덱스를 찾는다. 여기서 해당 index는 하나만 나와야 한다.
    # op_client = collection.vectorstore.client
    # wildcard_name = f"{collection_name}*"
    # co_response = op_client.cat.indices(index=wildcard_name, format="json")
    # # 실제 index 로 바꿔준다.
    # collection.vectorstore.index_name = co_response[0]['index']
    
    collection.delete_collection()
    # embed_delete = asyncio.create_task(collection.adelete_collection())
    # Extract embedding data before deletion
    embedding_data = models.Embedding.from_orm(db_embedding)
    

    # Delete the embedding
    db.delete(db_embedding)
    
    # Return the deleted embedding's information
    return embedding_data


class DataSourceNameRequest(BaseModel):
    name: str

@router.post(
    "/check_name",
    response_model=models.DataSourceSearchResponse,  # Adjust the response model to the correct one for datasources
    dependencies=[Depends(cookie)],  # Adjust this dependency based on your authentication setup
    description="""
    지정된 이름을 가진 데이터 소스가 존재하는지 확인합니다. 
    결과로 일치하는 데이터 소스의 목록과 총 개수를 반환합니다.
    """
)
def check_datasource_name(request: DataSourceNameRequest, db: Session = Depends(get_db)):
    """
    지정된 이름을 가진 데이터 소스가 존재하는지 확인하고, 결과를 리스트 형식으로 반환합니다.
    """
    query = db.query(database.DataSource).filter(
        database.db_comment_endpoint ,
        database.DataSource.name == request.name
    )
    total_count = query.count()
    results = query.all()  # Fetch all matching records

    return models.DataSourceSearchResponse(
        total_count=total_count,
        list=results
    )