import logging
from urllib.parse import quote
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import pydash
from sqlalchemy.orm import Session , joinedload
from sqlalchemy import insert, delete, or_, text
from typing import List, Union
from datetime import datetime
from git import Repo  # You need to install GitPython
import os
from ai_core.tool.base import load_tool
from app import models, database
from app.database import SessionLocal, get_db
from app.endpoint.login import cookie, SessionData, verifier
import tempfile
import shutil
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()


def convert_db_tool_to_pydantic(db_tool: database.Tool) -> models.Tool:
    # return models.Tool(
    #     tool_id=db_tool.tool_id,
    #     name=db_tool.name,
    #     description=db_tool.description,
    #     visibility=db_tool.visibility,
    #     tool_configuration=db_tool.tool_configuration,
    #     code=db_tool.code,
    #     git_url=db_tool.git_url,
    #     git_branch=db_tool.git_branch,
    #     git_path=db_tool.git_path,
    #     tags=db_tool.tags,
    #     create_user=db_tool.create_user,
    #     update_user=db_tool.update_user,
    #     created_at=db_tool.created_at,
    #     updated_at=db_tool.updated_at,
    #     create_user_info=db_tool.create_user_info
    # )
    return models.Tool.from_orm(db_tool)

def sanitize_git_url(git_url: str) -> str:
    # Replace special characters with underscores or other safe characters
    return git_url.replace("://", "_").replace("/", "_").replace(".", "_")


def construct_file_save_path(tool: models.Tool) -> str:
    """
    파일경로
    """
    root_path = os.getenv("DAISY_ROOT_FOLDER", "/var/webapps")  # Default to /var/webapps if DAISY_ROOT_FOLDER is not set
    if tool.tool_configuration == 'code':
        # If the configuration is code, use tool name and code file name
        file_save_path = os.path.join(root_path, 'tools/code_input', str(tool.tool_id)) + '.py'
    else:
        sanitized_git_url = sanitize_git_url(tool.git_url)
        encoded_git_path = requests.utils.quote(tool.git_path, safe='')
        file_save_path = os.path.join(root_path, "tools/git", sanitized_git_url, tool.git_branch, tool.git_path)
    return file_save_path

# Create a new tool
@router.post(
    "/tools/",
    response_model=models.Tool,
    dependencies=[Depends(cookie)],
    description="""
    Creates a new tool with optional code or Git configuration and tags.
    
    Request Body:
    - name (str): The name of the tool.
    - description (Optional[str]): The description of the tool.
    - visibility (str): The visibility of the tool (private/public).
    - tool_configuration (str): The configuration type of the tool (code/git).
    - code (Optional[str]): The tool code if the configuration is code.
    - git_url (Optional[str]): The Git URL if the configuration is git.
    - git_branch (Optional[str]): The Git branch if the configuration is git.
    - git_path (Optional[str]): The Git path if the configuration is git.
    - tag_ids  (List[int]): A list of tag IDs to associate with the tool.
    """
)
def create_tool(
    tool: models.ToolCreate, 
    db: Session = Depends(get_db), 
    session_data: SessionData = Depends(verifier)
):
    db_tool = database.Tool(create_user=session_data.user_id, update_user=session_data.user_id)
    params_ex = tool.dict(exclude_unset=True)
    for key, value in params_ex.items():
        if key not in ["tags","tag_ids"]:
            setattr(db_tool, key, value)

    db.add(db_tool)
    # db.flush()
    db.flush()
    db.refresh(db_tool)
    
    # Add tags to the tool
    if 'tag_ids' in params_ex and len(params_ex['tag_ids']) > 0 :
        for tag_id in params_ex['tag_ids']:
            db_tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
            if db_tag:
                # 이 부분이 맞는지. 에러가 나면 아랫구문활용.
                # stmt = insert(
                db_tool.tags.append(db_tag)

    # Convert db_tool to pydantic tool model
    pydantic_tool = convert_db_tool_to_pydantic(db_tool)
    # Fetch and save the file from Git if git_path is provided
    if 'tool_configuration' in params_ex and params_ex['tool_configuration'] == "git":
        try:
            token = get_gitlab_token(session_data=session_data)
            project_id_response = get_project_id(git_url=params_ex['git_url'],session_data=session_data)
            project_id = project_id_response["project_id"]
            repo_domain = params_ex['git_url'].split('/')[2]
            encoded_file_path = requests.utils.quote(params_ex['git_path'], safe='')
            api_url = f"https://{repo_domain}/api/v4/projects/{project_id}/repository/files/{encoded_file_path}/raw"
            headers = {"Private-Token": token}
            params = {"ref": params_ex['git_branch']}

            response = requests.get(api_url, headers=headers, params=params)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            file_content = response.text

            # Determine the path to save the file
            # Use the utility function to construct the file save path
            file_save_path = construct_file_save_path(pydantic_tool)
            dir_save_path = os.path.dirname(file_save_path)
            os.makedirs(dir_save_path, exist_ok=True)
            with open(file_save_path, "w") as file:
                file.write(file_content)
            tool = load_tool(db_tool.name, session_data.nickname, file_save_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Save the provided code if tool_configuration is "code"
    if 'tool_configuration' in params_ex and params_ex['tool_configuration'] == "code":
        try:
            # Use the utility function to construct the file save path
            file_save_path = construct_file_save_path(pydantic_tool)
            dir_save_path = os.path.dirname(file_save_path)
            os.makedirs(dir_save_path, exist_ok=True)
            with open(file_save_path, "w", encoding="utf-8") as file:
                file.write(params_ex['code'])
            tool = load_tool(db_tool.name, session_data.nickname, file_save_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        
    db.flush()
    db.refresh(db_tool)
    return db_tool

# Read all tools
@router.get("/tools/", response_model=List[models.Tool])
def read_tools(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    tools = db.query(database.Tool).filter(database.db_comment_endpoint).offset(skip).limit(limit).all()
    return tools

# Read a single tool
@router.get("/tools/{tool_id}", response_model=models.Tool)
def read_tool(tool_id: int, db: Session = Depends(get_db)):
    db_tool = db.query(database.Tool).filter(database.db_comment_endpoint).filter(database.Tool.tool_id == tool_id).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    return db_tool

# Update a tool
@router.put(
    "/tools/{tool_id}", 
    response_model=models.Tool,
    dependencies=[Depends(cookie)],
    description="""
    Updates an existing tool with optional code or Git configuration and tags.
    
    Request Body:
    - name (str): The name of the tool.
    - description (Optional[str]): The description of the tool.
    - visibility (str): The visibility of the tool (private/public).
    - tool_configuration (str): The configuration type of the tool (code/git).
    - code (Optional[str]): The tool code if the configuration is code.
    - git_url (Optional[str]): The Git URL if the configuration is git.
    - git_branch (Optional[str]): The Git branch if the configuration is git.
    - git_path (Optional[str]): The Git path if the configuration is git.
    - tag_ids  (List[int]): A list of tag IDs to associate with the tool.
    """
)
def update_tool(
    tool_id: int, 
    tool: models.ToolUpdate, 
    db: Session = Depends(get_db), 
    session_data: SessionData = Depends(verifier)
):
    db_tool = db.query(database.Tool).filter(database.db_comment_endpoint).filter(database.Tool.tool_id == tool_id).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    params_ex = tool.dict(exclude_unset=True)
    for key, value in params_ex.items():
        if key not in ["tags"]:
            setattr(db_tool, key, value)
    db_tool.update_user = session_data.user_id
    
    # Update tags
    if 'tag_ids' in params_ex and len(params_ex['tag_ids']) > 0 :
        # mapping many to many
        db_tool.tags = []
        for tag_id in params_ex['tag_ids']:
            db_tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
            if db_tag:
                db_tool.tags.append(db_tag)

    # Convert db_tool to pydantic tool model
    pydantic_tool = convert_db_tool_to_pydantic(db_tool)
    # Fetch and save the file from Git if git_path is provided
    if 'tool_configuration' in params_ex and params_ex['tool_configuration'] == "git":
        try:
            token = get_gitlab_token(session_data=session_data)
            project_id_response = get_project_id(git_url=params_ex['git_url'],session_data=session_data)
            project_id = project_id_response["project_id"]
            repo_domain = params_ex['git_url'].split('/')[2]
            encoded_file_path = requests.utils.quote(params_ex['git_path'], safe='')
            api_url = f"https://{repo_domain}/api/v4/projects/{project_id}/repository/files/{encoded_file_path}/raw"
            headers = {"Private-Token": token}
            params = {"ref": params_ex['git_branch']}

            response = requests.get(api_url, headers=headers, params=params)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            file_content = response.text

            # Determine the path to save the file
            # Use the utility function to construct the file save path
            file_save_path = construct_file_save_path(pydantic_tool)
            dir_save_path = os.path.dirname(file_save_path)
            os.makedirs(dir_save_path, exist_ok=True)
            with open(file_save_path, "w") as file:
                file.write(file_content)
            tool = load_tool(db_tool.name, session_data.nickname, file_save_path)

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Save the provided code if tool_configuration is "code"
    if 'tool_configuration' in params_ex and params_ex['tool_configuration'] == "code":
        try:
            # Use the utility function to construct the file save path
            file_save_path = construct_file_save_path(pydantic_tool)
            dir_save_path = os.path.dirname(file_save_path)
            os.makedirs(dir_save_path, exist_ok=True)
            with open(file_save_path, "w", encoding="utf-8") as file:
                file.write(params_ex['code'])
            tool = load_tool(db_tool.name, session_data.nickname, file_save_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    db.flush()
    db.refresh(db_tool)
    return db_tool

# Delete a tool
@router.delete("/tools/{tool_id}", response_model=models.Tool)
def delete_tool(tool_id: int, db: Session = Depends(get_db)):
    db_tool = db.query(database.Tool).filter(database.db_comment_endpoint).filter(database.Tool.tool_id == tool_id).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    tool_data = models.Tool.from_orm(db_tool)
    db.delete(db_tool)
    db.flush()
    return tool_data

# Associate a tool with a conversation
@router.post("/conversations/{conversation_id}/tools/", response_model=models.ConversationToolLink)
def associate_tool_with_conversation(conversation_id: str, tool_id: int, db: Session = Depends(get_db)):
    db_conversation = db.query(database.Conversation).filter(database.db_comment_endpoint).filter(database.Conversation.conversation_id == conversation_id).first()
    if db_conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    db_tool = db.query(database.Tool).filter(database.db_comment_endpoint).filter(database.Tool.tool_id == tool_id).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    stmt = insert(database.conversation_tools).values(
        conversation_id=conversation_id,
        tool_id=tool_id
    )
    db.execute(stmt)
    db.flush()
    return {"conversation_id": conversation_id, "tool_id": tool_id}

# Function to get branches from a given Git URL (if possible)
@router.post(
    "/tools/git/branches_by_repo",
    response_model=List[str],
    dependencies=[Depends(cookie)],
    description="Test Returns the list of branches for a given Git URL."
)
def branches_by_repo(git_url: str):
    try:
        # username = os.getenv('GIT_USERNAME')
        # password = os.getenv('GIT_PASSWORD')
        username = 'P196214'
        password = 'tde2-D7GDczPA8fyxYvo-H5jC'

        # Create a temporary directory to clone the repository
        temp_dir = tempfile.mkdtemp()
        branches = []
        
        if username and password:
            # Handle authenticated clone
            git_url_with_creds = git_url.replace('https://', f'https://{username}:{password}@')
            Repo.clone_from(git_url_with_creds, temp_dir)
        else:
            # Handle unauthenticated clone
            Repo.clone_from(git_url, temp_dir)

        repo = Repo(temp_dir)
        branches = [head.name for head in repo.heads]
        

        return branches
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get(
    "/get_gitlab_token",
    dependencies=[Depends(cookie)]
)
def get_gitlab_token(
    session_data: SessionData = Depends(verifier)
):
    """
    추후 table에서 가져오도록한다.
    """
    # token = os.getenv("GITLAB_TOKEN")
    token = session_data.token_gitlab
    
    if not token:
        raise HTTPException(status_code=500, detail="GitLab token not found")
    return token

@router.get(
    "/get_confluence_token",
    dependencies=[Depends(cookie)]
)
def get_confluence_token(
    session_data: SessionData = Depends(verifier)
):
    """
    추후 table에서 가져오도록한다.
    """
    # token = os.getenv("GITLAB_TOKEN")
    token = session_data.token_confluence
    
    if not token:
        raise HTTPException(status_code=500, detail="GitLab token not found")
    return token


@router.get(
    "/get_project_id",
    dependencies=[Depends(cookie)],
    description="""
    repository 로 부터 project id 를 가져온다.
    """
)
def get_project_id(
    git_url: str ,
    db: Session = Depends(get_db), 
    session_data: SessionData = Depends(verifier)
):
    token = get_gitlab_token(session_data=session_data)
    repo_damin = git_url.split('/')[2]
    repo_name = git_url.split('/')[-1]
    # api_url = f"https://gitlab.tde.sktelecom.com/api/v4/projects?search={repo_name}"
    api_url = f"https://{repo_damin}/api/v4/projects?search={repo_name}"
    
    headers = {
        "Private-Token": token
    }
    
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    
    projects = response.json()
    
    
    for project in projects:
        # if project['path_with_namespace'] == repo_name:
            # return {"project_id": project['id']}
        if project['name'] == repo_name:
            return {"project_id": project['id']}
    
    raise HTTPException(status_code=404, detail="Project not found")


    # get file info
    # encoded_file_path = requests.utils.quote(file_path, safe='')
    # %2F
    # https://gitlab.tde.sktelecom.com/api/v4/projects/15516/repository/files/app%2Fendpoint%2FchatbotConfig.py/raw?ref=develop

@router.post(
    "/tools/git/branches",
    response_model=List[str],
    dependencies=[Depends(cookie)],
    description="Returns the list of branches for a given Git URL."
)
def get_branches(
    git_url: str ,
    db: Session = Depends(get_db), 
    session_data: SessionData = Depends(verifier)
):
    token = get_gitlab_token(session_data=session_data)
    project_id_response = get_project_id(git_url=git_url,session_data=session_data)
    project_id = project_id_response["project_id"]
    # git test

    repo_damin = git_url.split('/')[2]
    repo_name = git_url.split('/')[-1]

    # api_url = f"https://gitlab.tde.sktelecom.com/api/v4/projects/{project_id}/repository/branches"
    api_url = f"https://{repo_damin}/api/v4/projects/{project_id}/repository/branches"
    
    headers = {
        "Private-Token": token
    }
    
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    
    branches = [branch['name'] for branch in response.json()]
    return branches

@router.post(
    "/tools/git/sources",
    response_model=List[dict],
    dependencies=[Depends(cookie)],
    description="Returns the list of sources for a given branch"
)
def get_gitlab_files_recursive(
    git_url: str,
    branch:str,
    db: Session = Depends(get_db), 
    session_data: SessionData = Depends(verifier)
):
    token = get_gitlab_token(session_data=session_data)
    project_id_response = get_project_id(git_url=git_url,session_data=session_data)
    project_id = project_id_response["project_id"]

    repo_damin = git_url.split('/')[2]
    repo_name = git_url.split('/')[-1]

    api_url = f"https://{repo_damin}/api/v4/projects/{project_id}/repository/tree"
    headers = {"Private-Token": token}
    files_list = []

    def fetch_directory_contents(path=''):
        params = {'ref': branch, 'path': path}
        response = requests.get(api_url, headers=headers, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch directory contents: {response.json()}")
        
        contents = response.json()
        
        for item in contents:
            if item['type'] == 'blob':
                if item['path'].endswith(('.py')):  # Filter for .py and .bin files
                    files_list.append({
                        'id': item['id'],
                        'path': item['path'],
                        'type': item['type'],
                    })
            elif item['type'] == 'tree':
                fetch_directory_contents(item['path'])

    fetch_directory_contents()
    return files_list


@router.post(
    "/tools/git/file_content",
    response_model=dict,
    dependencies=[Depends(cookie)],
    description="Returns the content of a file for a given path"
)
def get_gitlab_file_content(
    git_url: str,
    branch: str,
    file_path: str,
    db: Session = Depends(get_db), 
    session_data: SessionData = Depends(verifier)
):
    token = get_gitlab_token(session_data=session_data)
    project_id_response = get_project_id(git_url=git_url,session_data=session_data)
    project_id = project_id_response["project_id"]

    repo_domain = git_url.split('/')[2]
    encoded_file_path = requests.utils.quote(file_path, safe='')
    api_url = f"https://{repo_domain}/api/v4/projects/{project_id}/repository/files/{encoded_file_path}/raw"
    headers = {"Private-Token": token}
    params = {"ref": branch}

    response = requests.get(api_url, headers=headers, params=params)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    return {"content": response.text}

# Search tools
@router.post(
    "/tools/search/",
    response_model=models.SearchToolsResponse,
    dependencies=[Depends(cookie)],
    description="""
    Searches for tools based on various criteria.
    
    Request Body:
    - search_words (Optional[str]): The search words to filter tools by name or description.
    - search_scope (Optional[List[str]]): The scope of the search (e.g., ["name", "description"]).
    - visibility (Optional[str]): The visibility of the tools (private/public).
    - tag_ids (Optional[List[int]]): A list of tag IDs to filter tools.
    - user_list (Optional[List[str}]): The user who created the tools.
    - skip (Optional[int]): The number of records to skip for pagination.
    - limit (Optional[int]): The maximum number of records to return for pagination.
    """
)
def search_tools(
    search: models.ToolSearch, 
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(database.Tool)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 

    search_exclude = search.dict(exclude_unset=True)

    # Filter by search words
    if 'search_words' in search_exclude and not pydash.is_blank(search_exclude['search_words']):
        search_pattern = f"%{search_exclude['search_words']}%"
        if 'search_scope' in search_exclude:
            filters = []
            if "name" in search_exclude['search_scope']:
                filters.append(database.Tool.name.like(search_pattern))
            if "description" in search_exclude['search_scope']:
                filters.append(database.Tool.description.like(search_pattern))
            query = query.filter(or_(*filters))
        else:
            query = query.filter(
                or_(
                    database.Tool.name.like(search_pattern),
                    database.Tool.description.like(search_pattern)
                )
            )
    
    # Filter by visibility
    if 'visibility' in search_exclude and len(search_exclude['visibility']) > 0:
        query = query.filter(database.Tool.visibility.in_(search_exclude['visibility']))
    
    # Filter by tags
    if 'tag_ids' in search_exclude and len(search_exclude['tag_ids']) > 0:
        query = query.filter(database.Tool.tags.any(database.Tag.tag_id.in_(search_exclude['tag_ids'])))
    
    # 사용자 필터링
    user_filter_basic = [
        database.Tool.create_user == session_data.user_id ,
        database.Tool.visibility == 'public'
    ]
    query = query.filter(or_(*user_filter_basic))

    if 'user_list' in search_exclude and len(search_exclude['user_list']) > 0:
        query = query.filter(database.Tool.create_user .in_(search_exclude['user_list']))
    # 사용자 End
    
    total_count = query.count()
    query = query.order_by(database.Tool.updated_at.desc(), database.Tool.created_at.desc())   
    # Pagination
    if 'skip' in search_exclude and 'limit' in search_exclude:
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit'])
    
    # tools = query.options(
    #     joinedload(database.Tool.tags)
    # ).all()
    tools = query.all()

    return models.SearchToolsResponse(totalCount=total_count, list=tools)



class ToolNameRequest(BaseModel):
    name: str

@router.post(
    "/tools/check_name",
    response_model=models.SearchToolsResponse,  # Adjust the response model to the correct one for tools
    dependencies=[Depends(cookie)],  # Adjust this dependency based on your authentication setup
    description="""
    지정된 이름을 가진 도구가 존재하는지 확인합니다. 
    결과로 일치하는 도구의 목록과 총 개수를 반환합니다.
    """
)
def check_tool_name(request: ToolNameRequest, db: Session = Depends(get_db)):
    """
    지정된 이름을 가진 도구가 존재하는지 확인하고, 결과를 리스트 형식으로 반환합니다.
    """
    query = db.query(database.Tool).filter(
        database.db_comment_endpoint ,
        database.Tool.name == request.name
    )
    total_count = query.count()
    results = query.all()  # Fetch all matching records

    return models.SearchToolsResponse(
        totalCount=total_count,
        list=results
    )