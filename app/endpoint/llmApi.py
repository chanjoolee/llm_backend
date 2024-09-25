from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_ , and_, text
from datetime import datetime
from app.endpoint.login import cookie , SessionData , verifier

from app import models, database
import pydash
from app.database import SessionLocal, get_db
router = APIRouter()



@router.post(
    "/",
    response_model=models.LlmApi,
    dependencies=[Depends(cookie)],
    tags=["LLM APIs"],
    description="""
    Creates a new LLM API record.
    
    Request Body:
    - llm_api_name (Optional[str]): LLM API 이름 sk-sandbox
    - llm_api_type (Optional[str]): 오픈종료 private, public
    - llm_api_provider (Optional[str]): LLM API 제공자 'openai'
    - llm_api_url (Optional[str]): LLM Url 'https://aihub-api.sktelecom.com/aihub/v1/sandbox'
    - llm_model (Optional[str]): LLM Model. 쉼표로 구분하여 입력. gpt-4o,gpt-3.5
    - llm_api_key (Optional[str]): LLM Api Key
    """
)
def create_llm_api(llm_api: models.LlmApiCreate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_llm_api = database.LlmApi(**llm_api.dict())
    db_llm_api.create_user = session_data.user_id
    db.add(db_llm_api)
    db.flush()
    db.refresh(db_llm_api)
    return db_llm_api

@router.get(
    "/",
    response_model=List[models.LlmApi],
    dependencies=[Depends(cookie)],
    # tags=["LLM APIs"],
    description="""
    Retrieves a list of LLM API records with pagination.
    
    Query Parameters:
    - skip (Optional[int]): The number of records to skip for pagination.
    - limit (Optional[int]): The maximum number of records to return for pagination.
    """
)
def read_llm_apis(skip: Optional[int] = 0, limit: Optional[int] = 10, llm_api_type : Optional[str] = '' , db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    query = db.query(database.LlmApi)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 

    if not pydash.is_empty(llm_api_type):
        query = query.filter(database.LlmApi.llm_api_type == llm_api_type)
    if skip is not None  and limit is not None :
        query = query.offset(skip).limit(limit)
    llm_apis = query.all()
    return llm_apis

@router.get(
    "/public_and_private",
    response_model=List[models.LlmApi],
    dependencies=[Depends(cookie)],
    # tags=["LLM APIs"],
    description="""
    공용 LLM API 와 개인적으로 등록된 LLM API 를 가져온다.
    """
)
def read_public_and_private( db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    query = db.query(database.LlmApi)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 
    
    filter_llm_api_type = []
    filter_llm_api_type.append(database.LlmApi.llm_api_type == 'public')
    filter_llm_api_type.append(
        and_(
        database.LlmApi.llm_api_type == 'private',
        database.LlmApi.create_user == session_data.user_id
    )
    )
    query = query.filter(or_(*filter_llm_api_type))
    llm_apis = query.all()
    return llm_apis

@router.get(
    "/{llm_api_id}",
    response_model=models.LlmApi,
    dependencies=[Depends(cookie)],
    # tags=["LLM APIs"],
    description="""
    Retrieves a specific LLM API record by its ID.
    """
)
def read_llm_api(llm_api_id: int, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_llm_api = db.query(database.LlmApi).filter(database.db_comment_endpoint).filter(database.LlmApi.llm_api_id == llm_api_id).first()
    if db_llm_api is None:
        raise HTTPException(status_code=404, detail="LLM API not found")
    return db_llm_api

@router.put(
    "/{llm_api_id}",
    response_model=models.LlmApi,
    dependencies=[Depends(cookie)],
    # tags=["LLM APIs"],
    description="""
    Updates a specific LLM API record by its ID.
    
    Request Body:
    - llm_api_name (Optional[str]): LLM API 이름
    - llm_api_type (Optional[str]): 오픈종료 private, public
    - llm_api_provider (Optional[str]): LLM API 제공자
    - llm_api_url (Optional[str]): LLM Url
    - llm_model (Optional[str]): LLM Model. 쉼표로 구분하여 입력
    - llm_api_key (Optional[str]): LLM Api Key
    """
)
def update_llm_api(llm_api_id: int, llm_api: models.LlmApiUpdate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_llm_api = db.query(database.LlmApi).filter(database.db_comment_endpoint).filter(database.LlmApi.llm_api_id == llm_api_id).first()
    if db_llm_api is None:
        raise HTTPException(status_code=404, detail="LLM API not found")
    update_data = llm_api.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_llm_api, key, value)
    db_llm_api.update_user = session_data.user_id
    db.flush()
    db.refresh(db_llm_api)
    return db_llm_api

@router.delete(
    "/{llm_api_id}",
    response_model=models.LlmApi,
    dependencies=[Depends(cookie)],
    # tags=["LLM APIs"],
    description="""
    Deletes a specific LLM API record by its ID.
    """
)
def delete_llm_api(llm_api_id: int, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_llm_api = db.query(database.LlmApi).filter(database.db_comment_endpoint).filter(database.LlmApi.llm_api_id == llm_api_id).first()
    if db_llm_api is None:
        raise HTTPException(status_code=404, detail="LLM API not found")

    if db_llm_api.conversation:
        raise HTTPException(status_code=403, detail="Conversation is aleady exists. You can not delete llm api")
    
    if db_llm_api.agents:
        raise HTTPException(status_code=403, detail="Agent is aleady exists. You can not delete agent")


    # Convert to Pydantic model before deleting
    deleted_data = models.LlmApi.from_orm(db_llm_api)
    db.delete(db_llm_api)
    db.flush()
    return deleted_data

@router.delete(
    "/",
    response_model=List[models.LlmApi],
    dependencies=[Depends(cookie)],
    # tags=["LLM APIs"],
    description="""
    Deletes multiple llm api by their IDs.
    
    Request Body:
    - llm_api_id (List[int]): The list of llm api IDs to delete.
    - skip (Optional[int]): The number of records to skip for pagination after deletion.
    - limit (Optional[int]): The maximum number of records to return for pagination after deletion.
    """
)
def delete_multiple_llm_api(param_delete:models.LlmApiDelete , db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    llm_apis_to_delete = db.query(database.LlmApi).filter(database.db_comment_endpoint).filter(database.LlmApi.llm_api_id.in_(param_delete.llm_api_ids)).all()
    if not llm_apis_to_delete:
        raise HTTPException(status_code=404, detail="Tags not found")
    
    for llm_api in llm_apis_to_delete:
        if llm_api.conversation:
            raise HTTPException(status_code=403, detail="Conversation is aleady exists. You can not delete agent")
    
        if llm_api.agents:
            raise HTTPException(status_code=403, detail="Agent is aleady exists. You can not delete agent")
        db.delete(llm_api)
        db.flush()
    
    
    query = db.query(database.LlmApi).filter(database.db_comment_endpoint)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 

    if param_delete.skip is not None  and  param_delete.limit is not None :
        query = query.offset(param_delete.skip).limit(param_delete.limit)

    db.flush()
    remaining_tags = query.all()
    return remaining_tags

@router.post(
    "/search",
    response_model=List[models.LlmApi],
    dependencies=[Depends(cookie)],
    # tags=["LLM APIs"],
    description="""
    Searches for LLM API records based on various criteria.
    
    Request Body:
    - search_field (Optional[str]): Field to search in (
        ''  => 전체
        , llm_api_name  => LLM API 이름
        , llm_api_provider => LLM API 제공자
        , llm_api_url => LLM API 주소
        , llm_api_key  => LLM API 키
        , llm_model =>  텍스트 생성 모델
        , embedding_model => 임베딩 모델
    ).
    - search_words (Optional[str]): 검색어
    - llm_api_type (Optional[str]): The type of LLM API to filter by.
    - skip (Optional[int]): The number of records to skip for pagination.
    - limit (Optional[int]): The maximum number of records to return for pagination.
    """
)
def search_llm_api(   
    search : models.LlmApiSearch, 
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(database.LlmApi).filter(database.db_comment_endpoint)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 
    
    if not pydash.is_empty(search.search_field) and not pydash.is_empty(search.search_words):
        if search.search_field == '전체':
            query = query.filter(
                or_(
                    database.LlmApi.llm_api_name.ilike(f"%{search.search_words}%"),
                    database.LlmApi.llm_api_provider.ilike(f"%{search.search_words}%"),
                    database.LlmApi.llm_api_url.ilike(f"%{search.search_words}%"),
                    database.LlmApi.llm_api_key.ilike(f"%{search.search_words}%"),
                    database.LlmApi.llm_model.ilike(f"%{search.search_words}%"),
                    database.LlmApi.embedding_model.ilike(f"%{search.search_words}%"),
                )
            )
        elif search.search_field == 'llm_api_name': # LLM API 이름
            query = query.filter(database.LlmApi.llm_api_name.ilike(f"%{search.search_words}%"))
        elif search.search_field == 'llm_api_provider': # LLM API 제공자
            query = query.filter(database.LlmApi.llm_api_provider.ilike(f"%{search.search_words}%"))
        elif search.search_field == 'llm_api_url': # LLM API 주소
            query = query.filter(database.LlmApi.llm_api_url.ilike(f"%{search.search_words}%"))
        elif search.search_field == 'llm_api_key': # LLM API 키
            query = query.filter(database.LlmApi.llm_api_key.ilike(f"%{search.search_words}%"))
        elif search.search_field == 'llm_model': # 텍스트 생성 모델
            query = query.filter(database.LlmApi.llm_model.ilike(f"%{search.search_words}%"))
        elif search.search_field == 'embedding_model': # 임베딩 모델
            query = query.filter(database.LlmApi.embedding_model.ilike(f"%{search.search_words}%"))
    
    # Apply llm_api_type filter
    if not pydash.is_empty(search.llm_api_type):
        query = query.filter(database.LlmApi.llm_api_type == search.llm_api_type)
    # Pagination
    if search.skip is not None and search.limit is not None:
        query = query.offset(search.skip).limit(search.limit)
    results = query.all()
    return results