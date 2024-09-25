from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException , Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, text
from app.endpoint.login import cookie , SessionData , verifier

from app import models, database
from app.database import SessionLocal, get_db
import pydash


router = APIRouter()


@router.post(
    "/",
    response_model=models.Tag,
    dependencies=[Depends(cookie)],
    # tags=["Tags"],
    description="""
    Creates a new tag record.
    
    Request Body:
    - name (str): The name of the tag.
    - background_color (Optional[str]): The background color of the tag.
    """
)
def create_tag(tag: models.TagCreate, db: Session = Depends(get_db),session_data: SessionData = Depends(verifier)):
    db_tag = database.Tag(name=tag.name, background_color=tag.background_color)
    db_tag.create_user = session_data.user_id
    db.add(db_tag)
    db.flush()
    db.refresh(db_tag)
    return db_tag

@router.get(
    "/",
    response_model=List[models.Tag],
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
    query = db.query(database.Tag)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 
    
    if skip is not None  and limit is not None :
        query = query.offset(skip).limit(limit)
    tags =query.all()
    return tags

@router.get(
    "/{tag_id}",
    response_model=models.Tag,
    dependencies=[Depends(cookie)],
    # tags=["Tags"],
    description="Retrieves a specific tag by its ID."
)
def read_tag(tag_id: int, db: Session = Depends(get_db),session_data: SessionData = Depends(verifier)):
    tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
    if tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")
    return tag

@router.put(
    "/{tag_id}",
    response_model=models.Tag,
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
def update_tag(tag_id: int, tag: models.TagUpdate, db: Session = Depends(get_db),session_data: SessionData = Depends(verifier)):
    db_tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
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
    response_model=models.Tag,
    tags=["Tags"],
    description="Deletes a specific tag by its ID."
)
def delete_tag(tag_id: int, db: Session = Depends(get_db)):
    db_tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
    if db_tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")
    db.delete(db_tag)
    db.flush()
    return db_tag

@router.delete(
    "/",
    response_model=List[models.Tag],
    tags=["Tags"],
    description="""
    Deletes multiple tags by their IDs.
    
    Request Body:
    - tag_ids (List[int]): The list of tag IDs to delete.
    - skip (Optional[int]): The number of records to skip for pagination after deletion.
    - limit (Optional[int]): The maximum number of records to return for pagination after deletion.
    """
)
def delete_multiple_tags(tag_delete:models.TagDelete , db: Session = Depends(get_db)):
    tags_to_delete = db.query(database.Tag).filter(database.Tag.tag_id.in_(tag_delete.tag_ids)).all()
    if not tags_to_delete:
        raise HTTPException(status_code=404, detail="Tags not found")
    
    for tag in tags_to_delete:
        db.delete(tag)
    db.flush()
    
    query = db.query(database.Tag)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 

    if tag_delete.skip is not None  and tag_delete.limit is not None :
        query = query.offset(tag_delete.skip).limit(tag_delete.limit)

    remaining_tags = query.all()
    return remaining_tags

@router.post(
    "/search",
    response_model=List[models.Tag],
    dependencies=[Depends(cookie)],
    # tags=["LLM APIs"],
    description="""
    Searches for Tags records based on various criteria.
    
    Request Body:
    - search_words (Optional[str]): 검색어
    """
)
def search_tags(   
    search : models.TagSearch, 
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(database.Tag)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 
    
    if not pydash.is_empty(search.search_words) and not pydash.is_empty(search.search_words):
        query = query.filter(database.Tag.name.ilike(f"%{search.search_words}%"))
    
    if search.skip is not None  and search.limit is not None :
        query = query.offset(search.skip).limit(search.limit)
    results = query.all()
    return results