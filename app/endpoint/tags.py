import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException , Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import or_, text
from app.endpoint.login import cookie , SessionData , verifier

from app import database
from app.model import model_llm
from app.schema import schema_llm
from app.database import SessionLocal, get_db
import pydash

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()


@router.post(
    "/",
    response_model=schema_llm.Tag,
    dependencies=[Depends(cookie)],
    # tags=["Tags"],
    description="""
    Creates a new tag record.
    
    Request Body:
    - name (str): The name of the tag.
    - background_color (Optional[str]): The background color of the tag.
    """
)
def create_tag(tag: schema_llm.TagCreate, db: Session = Depends(get_db),session_data: SessionData = Depends(verifier)):
    db_tag = model_llm.Tag(name=tag.name, background_color=tag.background_color)
    db_tag.create_user = session_data.user_id
    db.add(db_tag)
    db.flush()
    db.refresh(db_tag)
    return db_tag

@router.get(
    "/",
    response_model=schema_llm.SearchTagResponse,
    dependencies=[Depends(cookie)],
    # tags=["Tags"],
    description="""
    Retrieves a list of tag records with pagination.
    
    Query Parameters:
    - skip (Optional[int]): The number of records to skip for pagination.
    - limit (Optional[int]): The maximum number of records to return for pagination.
    """
)
def read_tags(skip: Optional[int] = 0, limit: Optional[int] = 10, db: Session = Depends(get_db),session_data: SessionData = Depends(verifier)):
    query = db.query(model_llm.Tag)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 
    
    total_count = query.count()
    query = query.order_by(model_llm.Tag.updated_at.desc(),model_llm.Tag.created_at.desc())
    if skip is not None  and limit is not None :
        query = query.offset(skip).limit(limit)
        
    results = query.all()
    tags =query.all()
    
    return schema_llm.SearchTagResponse(totalCount=total_count, list=results)

@router.get(
    "/{tag_id}",
    response_model=schema_llm.Tag,
    dependencies=[Depends(cookie)],
    # tags=["Tags"],
    description="Retrieves a specific tag by its ID."
)
def read_tag(tag_id: int, db: Session = Depends(get_db),session_data: SessionData = Depends(verifier)):
    tag = db.query(model_llm.Tag).filter(model_llm.db_comment_endpoint).filter(model_llm.Tag.tag_id == tag_id).first()
    if tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")
    return tag

@router.put(
    "/{tag_id}",
    response_model=schema_llm.Tag,
    dependencies=[Depends(cookie)],
    # tags=["Tags"],
    description="""
    Updates a specific tag by its ID.
    
    Request Body:
    - name (str): The name of the tag.
    - background_color (Optional[str]): The background color of the tag.
    - update_user (Optional[str]): The user updating the tag.
    """
)
def update_tag(tag_id: int, tag: schema_llm.TagUpdate, db: Session = Depends(get_db),session_data: SessionData = Depends(verifier)):
    db_tag = db.query(model_llm.Tag).filter(model_llm.db_comment_endpoint).filter(model_llm.Tag.tag_id == tag_id).first()
    if db_tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")
    
    db_tag.name = tag.name
    db_tag.background_color = tag.background_color

    update_data = tag.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_tag, key, value)

    db_tag.update_user = session_data.user_id
    db.flush()
    db.refresh(db_tag)
    return db_tag

@router.delete(
    "/{tag_id}",
    response_model=schema_llm.Tag,
    tags=["Tags"],
    description="Deletes a specific tag by its ID."
)
def delete_tag(tag_id: int, db: Session = Depends(get_db)):
    db_tag = db.query(model_llm.Tag).filter(model_llm.db_comment_endpoint).filter(model_llm.Tag.tag_id == tag_id).first()
    if db_tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")
    db.delete(db_tag)
    db.flush()
    return db_tag

@router.delete(
    "/",
    response_model=List[schema_llm.Tag],
    tags=["Tags"],
    description="""
    Deletes multiple tags by their IDs.
    
    Request Body:
    - tag_ids (List[int]): The list of tag IDs to delete.
    - skip (Optional[int]): The number of records to skip for pagination after deletion.
    - limit (Optional[int]): The maximum number of records to return for pagination after deletion.
    """
)
def delete_multiple_tags(tag_delete:schema_llm.TagDelete , db: Session = Depends(get_db)):
    tags_to_delete = db.query(model_llm.Tag).filter(model_llm.Tag.tag_id.in_(tag_delete.tag_ids)).all()
    if not tags_to_delete:
        raise HTTPException(status_code=404, detail="Tags not found")
    
    for tag in tags_to_delete:
        db.delete(tag)
    db.flush()
    
    query = db.query(model_llm.Tag)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 

    if tag_delete.skip is not None  and tag_delete.limit is not None :
        query = query.offset(tag_delete.skip).limit(tag_delete.limit)

    remaining_tags = query.all()
    return remaining_tags

@router.post(
    "/search",
    response_model=schema_llm.SearchTagResponse,
    dependencies=[Depends(cookie)],
    # tags=["LLM APIs"],
    description="""
    Searches for Tags records based on various criteria.
    
    Request Body:
    - search_words (Optional[str]): 검색어
    """
)
def search_tags(   
    search : schema_llm.TagSearch, 
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    search_exclude = search.dict(exclude_unset=True)
    
    query = db.query(model_llm.Tag)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 
    
    if 'search_words' in search_exclude and search_exclude['search_words']:
        query = query.filter(model_llm.Tag.name.ilike(f"%{search_exclude["search_words"]}%"))
    
    total_count = query.count()
    query = query.order_by(model_llm.Tag.updated_at.desc(),model_llm.Tag.created_at.desc())
    if 'skip' in search_exclude and 'limit' in search_exclude:
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit'])
    
    results = query.all()
    
    return schema_llm.SearchTagResponse(totalCount=total_count, list=results)



class TagNameRequest(BaseModel):
    name: str

@router.post(
    "/check_name",
    response_model=schema_llm.SearchTagResponse,  # Adjust the response model to the correct one for tags
    dependencies=[Depends(cookie)],  # Adjust this dependency based on your authentication setup
    description="""
    지정된 이름을 가진 태그가 존재하는지 확인합니다. 
    결과로 일치하는 태그의 목록과 총 개수를 반환합니다.
    """
)
def check_tag_name(request: TagNameRequest, db: Session = Depends(get_db)):
    """
    지정된 이름을 가진 태그가 존재하는지 확인하고, 결과를 리스트 형식으로 반환합니다.
    """
    query = db.query(model_llm.Tag).filter(
        model_llm.db_comment_endpoint ,
        model_llm.Tag.name == request.name
    )
    total_count = query.count()
    results = query.all()  # Fetch all matching records

    return schema_llm.SearchTagResponse(
        totalCount=total_count,
        list=results
    )