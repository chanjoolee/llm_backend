import asyncio
from datetime import datetime
from enum import Enum
import logging
import os
from typing import Annotated, Iterable, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile ,status,BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import Field
import requests
from sqlalchemy import insert, or_, text
from sqlalchemy.orm import Session , joinedload , selectinload
from ai_core.data_source.base import DataSourceType, create_data_source
from ai_core.data_source.splitter.base import SplitterType, create_splitter ,Language,TextSplitter
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

logger = logging.getLogger('sqlalchemy.engine')
router = APIRouter()
COLLECTION_DIR = os.path.join(os.getenv("DAISY_ROOT_FOLDER") ,'datasource/collection')
CHROMA_DB_DIR = os.path.join(COLLECTION_DIR ,'daisy_chromadb')

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
async def create_datasource(
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
    db: Session = Depends(database.get_db_async),
    session_data: SessionData = Depends(verifier)
):
    
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
    existing_datasource = db.query(database.DataSource).filter(database.DataSource.datasource_id == datasource_id).first()
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
            db_tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
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
        data_source_type=db_datasource.datasource_type
    )

    
    data_size = 0
    if datasource_type == DataSourceType.TEXT:
        data_size = len(db_datasource.raw_text.encode('utf-8'))
        save_task = asyncio.create_task(data_source.save_data(raw_text=[db_datasource.raw_text]))
        
    if datasource_type == DataSourceType.PDF_FILE:
        pdf_file_path = db_datasource.file_path
        if os.path.exists(pdf_file_path):
            data_size = os.path.getsize(pdf_file_path)  # Get PDF file size in bytes
        save_task = asyncio.create_task(data_source.save_data(pdf_file_path=db_datasource.file_path))
        
        
    if datasource_type == DataSourceType.DOC_FILE:
        doc_file_path = db_datasource.file_path
        if os.path.exists(doc_file_path):
            data_size = os.path.getsize(doc_file_path)  # Get DOC file size in bytesn bytes
        save_task = asyncio.create_task(data_source.save_data(doc_file_path=db_datasource.file_path))
        
    if datasource_type == DataSourceType.CONFLUENCE:
        if utils.is_empty(session_data.token_confluence):
            raise HTTPException(status_code=422, detail=f"Token of Confluence is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = asyncio.create_task(data_source.save_data(url=db_datasource.url ,access_token=session_data.token_confluence,space_key=db_datasource.space_key)) 
        
    if datasource_type == DataSourceType.JIRA:
        if utils.is_empty(db_datasource.token):
            raise HTTPException(status_code=422, detail=f"Token of Jira is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = asyncio.create_task(
            data_source.save_data(
                url=db_datasource.url ,
                access_token=db_datasource.token,
                project_key=db_datasource.project_key,
                start=db_datasource.start, 
                limit=db_datasource.limit
            )
        ) 
        
    if datasource_type == DataSourceType.GITLAB:
        if utils.is_empty(session_data.token_gitlab):
            raise HTTPException(status_code=422, detail=f"Token of Gitlab is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = asyncio.create_task(data_source.save_data(
            url=db_datasource.url ,
            namespace=db_datasource.namespace, 
            project_name = db_datasource.project_name,
            branch=db_datasource.branch,
            private_token=session_data.token_gitlab)) 
        
    if datasource_type == DataSourceType.GITLAB_DISCUSSION:
        if utils.is_empty(session_data.token_gitlab):
            raise HTTPException(status_code=422, detail=f"Token of Gitlab is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = asyncio.create_task(data_source.save_data(
            url=db_datasource.url ,
            namespace=db_datasource.namespace, 
            project_name = db_datasource.project_name,
            private_token=session_data.token_gitlab)) 
        
    if datasource_type == DataSourceType.URL:
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = asyncio.create_task(
            data_source.save_data(
                url=db_datasource.url ,
                base_url=db_datasource.base_url,
                max_depth=db_datasource.max_depth 
            )
        ) 

        
    logger.info("Data saving task started")  
    def save_callback(future):
        try:
            if future.exception():
                logger.info("Datasource task failed: ", future.exception())
                db_datasource.status = database.DatasourceDownloadStatus.error
            else:
                # embeded, total = future.result()
                # logger.info("Embedding task completed. Number of chunks embedded / Total: ", str(embeded), " / ", str(total))
                db_datasource.downloaded_at = datetime.now(database.KST)
                db_datasource.updated_at = datetime.now(database.KST)
                db_datasource.status = database.DatasourceDownloadStatus.downloaded
            
            # update_message = {
            #     "type": "datasource_update",
            #     "data": utils.sqlalchemy_to_dict(db_datasource)
            # }
            update_message = {
                "type": "datasource_update",
                "data": models.Datasource.from_orm(db_datasource).dict()
            }
            asyncio.create_task(broadcast_update(update_message))
        except Exception as e:
            db_datasource.status = database.DatasourceDownloadStatus.error
            logger.error(f"Save task failed: {e}")
            
        finally:
            db.commit()
            db.close()  
        
    if save_task:
        save_task.add_done_callback(save_callback) 
    
    
    db.commit()
    return db_datasource

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
    description="Update an existing data source using multipart form data."
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
    raw_text: Optional[str] = Form(None, description="Raw text data"),
    url: Optional[str] = Form(None, description="URL for URL data source"),
    base_url: Optional[str] = Form(None, description="Base URL for URL data source"),
    max_depth: Optional[int] = Form(None, description="Maximum depth for URL crawling"),
    tag_ids: Optional[str] = Form(None, description="Comma-separated list of tag IDs"),
    file: Annotated[UploadFile, File(description="File for document or PDF")] = None,  # File upload
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    # Fetch the existing datasource from the database
    db_datasource = db.query(database.DataSource).filter(database.DataSource.datasource_id == datasource_id).first()
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

    # updates to the database
    db.flush()
    db.refresh(db_datasource)
    
    
    # 데이터를 텍스트 파일로 저장
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type
    )

        
    data_size = 0
    if datasource_type1 == DataSourceType.TEXT:
        data_size = len(db_datasource.raw_text.encode('utf-8'))
        save_task = asyncio.create_task(data_source.save_data(raw_text=[db_datasource.raw_text]))
        
    if datasource_type1 == DataSourceType.PDF_FILE:
        pdf_file_path = db_datasource.file_path
        if os.path.exists(pdf_file_path):
            data_size = os.path.getsize(pdf_file_path)  # Get PDF file size in bytes
        save_task = asyncio.create_task(data_source.save_data(pdf_file_path=db_datasource.file_path))
        
        
    if datasource_type1 == DataSourceType.DOC_FILE:
        doc_file_path = db_datasource.file_path
        if os.path.exists(doc_file_path):
            data_size = os.path.getsize(doc_file_path)  # Get DOC file size in bytesn bytes
        save_task = asyncio.create_task(data_source.save_data(doc_file_path=db_datasource.file_path))
        
    if datasource_type1 == DataSourceType.CONFLUENCE:
        if utils.is_empty(session_data.token_confluence):
            raise HTTPException(status_code=422, detail=f"Token of Confluence is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = asyncio.create_task(data_source.save_data(url=db_datasource.url ,access_token=session_data.token_confluence,space_key=db_datasource.space_key)) 
        
    if datasource_type1 == DataSourceType.JIRA:
        if utils.is_empty(db_datasource.token):
            raise HTTPException(status_code=422, detail=f"Token of Jira is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = asyncio.create_task(
            data_source.save_data(
                url=db_datasource.url ,
                access_token=db_datasource.token,
                project_key=db_datasource.project_key,
                start=db_datasource.start, 
                limit=db_datasource.limit
            )
        ) 
        
    if datasource_type1 == DataSourceType.GITLAB:
        if utils.is_empty(session_data.token_gitlab):
            raise HTTPException(status_code=422, detail=f"Token of Gitlab is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = asyncio.create_task(data_source.save_data(
            url=db_datasource.url ,
            namespace=db_datasource.namespace, 
            project_name = db_datasource.project_name,
            branch=db_datasource.branch,
            private_token=session_data.token_gitlab)) 
        
    if datasource_type1 == DataSourceType.GITLAB_DISCUSSION:
        if utils.is_empty(session_data.token_gitlab):
            raise HTTPException(status_code=422, detail=f"Token of Gitlab is required for {datasource_type.name} datasource type")
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = asyncio.create_task(data_source.save_data(
            url=db_datasource.url ,
            namespace=db_datasource.namespace, 
            project_name = db_datasource.project_name,
            private_token=session_data.token_gitlab)) 
        
    if datasource_type1 == DataSourceType.URL:
        # if os.path.exists(data_source.data_dir_path + '/0'):
        #     data_size = get_total_file_size(data_source.data_dir_path + '/0')
        save_task = asyncio.create_task(
            data_source.save_data(
                url=db_datasource.url ,
                base_url=db_datasource.base_url,
                max_depth=db_datasource.max_depth 
            )
        ) 

        
    logger.info("Data saving task started")  
    def save_callback(future):
        if future.exception():
            logger.info("Datasource task failed: ", future.exception())
            db_datasource.status = database.DatasourceDownloadStatus.error
        else:
            # embeded, total = future.result()
            # logger.info("Embedding task completed. Number of chunks embedded / Total: ", str(embeded), " / ", str(total))
            db_datasource.downloaded_at = datetime.now(database.KST)
            db_datasource.updated_at = datetime.now(database.KST)
            db_datasource.status = database.DatasourceDownloadStatus.downloaded
        
    save_task.add_done_callback(save_callback)
    await save_task
    

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
    existing_datasource = db.query(database.DataSource).filter(database.DataSource.datasource_id == datasource_id).first()
        
    info_datasource = None
    if existing_datasource:
        # 파일때문에 기존의 path를 그대로 사용하기 위해서.
        info_datasource = models.Datasource.from_orm(existing_datasource)
    else:
        # create_user_info = models.User.from_orm(session_data)
        db_user = db.query(database.User).filter(database.User.user_id == session_data.user_id).first()
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
        data_source_type=info_datasource.datasource_type
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
    # Query the database to find the data source by its ID
    db_datasource = db.query(database.DataSource).filter(database.DataSource.datasource_id == datasource_id).first()

    # If the data source doesn't exist, raise a 404 error
    if not db_datasource:
        raise HTTPException(status_code=404, detail="Data source not found")


    # 물리적으로 데이타소스를 삭제한다.    
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type
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
            persist_directory=CHROMA_DB_DEFAULT_PERSIST_DIR)
        
        collection.delete_collection()
    data_source.delete_data_directory()
    
    
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
    # Example to show how you can process the data


    payload_data = payload.dict(exclude_unset=True)

    # Retrieve the data source
    db_datasource = db.query(database.DataSource).filter(database.DataSource.datasource_id == payload.datasource_id).first()
    if not db_datasource:
        raise HTTPException(status_code=404, detail="DataSource not found")

    # Retrieve LLM API configuration based on the llm_api_id
    # Here you can use your models/logic to get API config details based on llm_api_id
    # Example API data retrieval
    db_llm_api = db.query(database.LlmApi).filter(database.LlmApi.llm_api_id == payload.llm_api_id).first()
    if not db_llm_api:
        raise HTTPException(status_code=404, detail="LLM API configuration not found")
    
    
    data_source_type = DataSourceType(db_datasource.datasource_type)
    # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type
    )
    
    # 3. 데이터 소스에 컬렉션 추가
    collection_name = create_collection_name(data_source.id, payload.embedding_model)
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider(db_llm_api.llm_api_provider),
        llm_api_key=db_llm_api.llm_api_key,
        llm_api_url=db_llm_api.llm_api_url,
        llm_embedding_model_name=payload.embedding_model,
        persist_directory=CHROMA_DB_DEFAULT_PERSIST_DIR)
    
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

    data = data_source.read_data()
    
    # Filter out keys from payload_data
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
            db_embedding.last_update_time = datetime.now(database.KST)
            db_embedding.success_at = datetime.now(database.KST)
            db_embedding.completed_at = datetime.now(database.KST)
            db_embedding.status="updated"
            
    embed_task.add_done_callback(embed_callback)

    await embed_task
    # End embedding

    db.flush()
    db.refresh(db_embedding)
    return db_embedding

@router.put(
    "/{datasource_id}/{embedding_id}",
    response_model=models.Embedding,
    dependencies=[Depends(cookie)],
    description="""
    Update embedding for a given data source and stores it in ChromaDB.
    
    This endpoint performs the following:
    
    1. **Data Retrieval**: The data is retrieved from the specified data source based on its type (e.g., text, PDF, DOC, URL).
    
    2. **Embedding Process**: The documents from the data source are split into chunks using the specified text splitter and embedded using the selected LLM model. The embeddings are then stored in ChromaDB.

    3. **Status Tracking**: The status of the embedding process is tracked, and the result is returned once the task is completed. The status can be "updating", "updated", or "failed".

    ### Request Body (Models: EmbeddingCreate)
    
    - **datasource_id** (Optional): Please Do not input. The unique identifier of the data source.
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
      "llm_api_id": 1,        
      "embedding_model": "text-embedding-3-small",
      "splitter": "RecursiveCharacterTextSplitter",
      "chunk_size": 1000,
      "chunk_overlap": 50
    }
    {
      "llm_api_id": 1,        
      "embedding_model": "text-embedding-3-large",
      "splitter": "CharacterTextSplitter",
      "chunk_size": 1000,
      "chunk_overlap": 50,
      "separator": "\\n",
      "is_separator_regex": false
    }
    ```
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
        database.Embedding.embedding_id == embedding_id
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
    db_llm_api = db.query(database.LlmApi).filter(database.LlmApi.llm_api_id == db_embedding.llm_api_id).first()
    if not db_llm_api:
        raise HTTPException(status_code=404, detail="LLM API configuration not found")
    
    
    data_source_type = DataSourceType(db_datasource.datasource_type)
    # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=db_datasource.name,
        created_by=db_datasource.create_user_info.nickname,
        description=db_datasource.description,
        data_source_type=db_datasource.datasource_type
    )
    
    # 3. 데이터 소스에 컬렉션 추가
    collection_name = create_collection_name(data_source.id, db_embedding.embedding_model)
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider(db_llm_api.llm_api_provider),
        llm_api_key=db_llm_api.llm_api_key,
        llm_api_url=db_llm_api.llm_api_url,
        llm_embedding_model_name=payload.embedding_model,
        persist_directory=CHROMA_DB_DEFAULT_PERSIST_DIR
    )
    collection = data_source.collections[collection_name]
    
    # db_embedding.embedding_id = collection_name
    db.flush()
    
    data_size = 0


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
            db_embedding.last_update_time = datetime.now(database.KST)
            db_embedding.success_at = datetime.now(database.KST)
            db_embedding.completed_at = datetime.now(database.KST)
            db_embedding.status="updated"
            
    embed_task.add_done_callback(embed_callback)

    await embed_task
    # End embedding

    db.flush()
    db.refresh(db_embedding)
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
    db_datasource = db.query(database.DataSource).filter(database.DataSource.datasource_id == payload.datasource_id).first()
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
    
    def read_data(preview_data) -> Iterable[Document]:
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
    splitted_documents: Iterable[Document] = split_texts(preview_data_i, splitter)
   
   
    result: List[dict] = []
    for document in splitted_documents:
        # Convert each Document object to a dictionary with 'content' and 'metadata'
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
    db_embedding = db.query(database.Embedding).filter(database.Embedding.embedding_id == payload.embedding_id).first()
    if not db_embedding:
        raise HTTPException(status_code=404, detail="Embedding is not found")

    
    data_source_type = DataSourceType(db_embedding.datasource.datasource_type)
    # 2. 데이터 소스 생성
    data_source = create_data_source(
        data_source_name=db_embedding.datasource.name,
        created_by=db_embedding.datasource.create_user_info.nickname,
        description=db_embedding.datasource.description,
        data_source_type=db_embedding.datasource.datasource_type
    )
    # 3. 데이터 소스에 컬렉션 추가

    collection_name = create_collection_name(db_embedding.datasource.datasource_id, db_embedding.embedding_model)
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider(db_embedding.llm_api.llm_api_provider),
        llm_api_key=db_embedding.llm_api.llm_api_key,
        llm_api_url=db_embedding.llm_api.llm_api_url,
        llm_embedding_model_name=db_embedding.embedding_model,
        persist_directory=CHROMA_DB_DEFAULT_PERSIST_DIR)
    
    query_results = collection.similarity_search(query=payload.query, k=4)
    return query_results

@router.get(
    "/{datasource_id}",
    response_model=models.Datasource,
    dependencies=[Depends(cookie)],
    summary="Get a specific data source by ID",
    description="Retrieve detailed information about a specific data source using its ID."
)
async def get_datasource(datasource_id: str, db: Session = Depends(get_db)):
    datasource = db.query(database.DataSource).filter(database.DataSource.datasource_id == datasource_id).first()
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
    query = db.query(database.Embedding)
    
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
    db_datasource = db.query(database.DataSource).filter(database.DataSource.datasource_id == datasource_id).first()
    if not db_datasource:
        raise HTTPException(status_code=404, detail="Datasource not found")
    
    # Fetch the embedding
    db_embedding = db.query(database.Embedding).filter(
        database.Embedding.datasource_id == datasource_id, 
        database.Embedding.embedding_id == embedding_id
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
        database.Embedding.embedding_id == embedding_id
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
        data_source_type=db_datasource.datasource_type
    )
    
    # 3. 데이터 소스에 컬렉션 추가
    collection_name = create_collection_name(data_source.id, db_embedding.embedding_model)
    collection = data_source.add_collection(
        collection_name=collection_name,
        llm_api_provider=LlmApiProvider(db_llm_api.llm_api_provider),
        llm_api_key=db_llm_api.llm_api_key,
        llm_api_url=db_llm_api.llm_api_url,
        llm_embedding_model_name=db_embedding.embedding_model,
        persist_directory=CHROMA_DB_DEFAULT_PERSIST_DIR)    
    
    collection.delete_collection()
    # Extract embedding data before deletion
    embedding_data = models.Embedding.from_orm(db_embedding)

    # Delete the embedding
    db.delete(db_embedding)
    db.commit()
    
    # Return the deleted embedding's information
    return embedding_data
