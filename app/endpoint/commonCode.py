import logging
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from app.database import SessionLocal, get_db
from app import database
from app.model import model_llm
from app.schema import schema_llm
from fastapi import APIRouter
from uuid import UUID, uuid4
from app.endpoint.login import cookie , SessionData , verifier
from passlib.context import CryptContext
from app.utils.utils import pwd_context , hash_password ,  verify_password


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()
# app = FastAPI()



@router.get("/codegroups/{code_group}", response_model=schema_llm.CommonCodeGroup, dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def read_code_group(code_group: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    
    query = db.query(model_llm.CommonCodeGroup)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    query = query.filter(model_llm.CommonCodeGroup.code_group == code_group)
    db_code_group = query.first()
    
    if db_code_group is None:
        raise HTTPException(status_code=404, detail="Code Group not found")
    return db_code_group

@router.get("/codegroups", response_model=List[schema_llm.CommonCodeGroup], dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def read_code_groups(db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):

    query = db.query(model_llm.CommonCodeGroup)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    db_code_group = query.all()

    if db_code_group is None:
        raise HTTPException(status_code=404, detail="Code Group not found")
    return db_code_group


@router.post("/codegroups/", response_model=schema_llm.CommonCodeGroup, dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def create_code_group(code_group: schema_llm.CommonCodeGroupCreate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_code_group = model_llm.CommonCodeGroup(**code_group.model_dump())
    db_code_group.create_user = session_data.user_id
    db.add(db_code_group)
    db.flush()
    db.refresh(db_code_group)
    return db_code_group

@router.put("/codegroups/{code_group}", response_model=schema_llm.CommonCodeGroup, dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def update_code_group(code_group: str, updated_group: schema_llm.CommonCodeGroupCreate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    query = db.query(model_llm.CommonCodeGroup)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    query = query.filter(model_llm.CommonCodeGroup.code_group == code_group)
    db_code_group = query.first()
    
    if db_code_group is None:
        raise HTTPException(status_code=404, detail="Code Group not found")

    for key, value in updated_group.dict().items():
        setattr(db_code_group, key, value)

    db_code_group.update_user = session_data.user_id
    db.flush()
    db.refresh(db_code_group)
    return db_code_group

@router.delete("/codegroups/{code_group}", response_model=dict, dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def delete_code_group(code_group: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    query = query.filter(model_llm.CommonCodeGroup.code_group == code_group)
    db_code_group = query.first()

    if db_code_group is None:
        raise HTTPException(status_code=404, detail="Code Group not found")

    db.delete(db_code_group)
    db.flush()
    return {"detail": "Code Group deleted"}

@router.get("/codegroups/{code_group}/details", response_model=List[schema_llm.CommonCodeDetail], dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def read_code_group_details(code_group: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):

    query = db.query(model_llm.CommonCodeGroup).filter(model_llm.db_comment_endpoint)
    query = query.filter(model_llm.CommonCodeGroup.code_group == code_group)
    db_code_group = query.first()
    
    if db_code_group is None:
        raise HTTPException(status_code=404, detail="Code Group not found")

    return db_code_group.commonCode    

@router.get("/codedetails/{code_group}/{code_value}", response_model=schema_llm.CommonCodeDetail, dependencies=[Depends(cookie)], tags=["CommonCodeDetail"])
def read_code_detail(code_group: str, code_value: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    comment = text("/* is_endpoint_query */ 1=1")
    db_code_detail = db.query(model_llm.CommonCodeDetail).filter(
        model_llm.CommonCodeDetail.code_group == code_group,
        model_llm.CommonCodeDetail.code_value == code_value
    ).filter(comment).first()
    if db_code_detail is None:
        raise HTTPException(status_code=404, detail="Code Detail not found")
    return db_code_detail

@router.post("/codedetails/", response_model=schema_llm.CommonCodeDetail, dependencies=[Depends(cookie)], tags=["CommonCodeDetail"])
def create_code_detail(code_detail: schema_llm.CommonCodeDetailCreate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_code_detail = model_llm.CommonCodeDetail(**code_detail.model_dump())
    db_code_detail.create_user = session_data.user_id
    db.add(db_code_detail)
    db.flush()
    db.refresh(db_code_detail)
    return db_code_detail

@router.put("/codedetails/{code_group}/{code_value}", response_model=schema_llm.CommonCodeDetail, dependencies=[Depends(cookie)],tags=["CommonCodeDetail"])
def update_code_detail(code_group: str, code_value: str, updated_detail: schema_llm.CommonCodeDetailCreate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    comment = text("/* is_endpoint_query */ 1=1")
    db_code_detail = db.query(model_llm.CommonCodeDetail).filter(
        model_llm.CommonCodeDetail.code_group == code_group,
        model_llm.CommonCodeDetail.code_value == code_value
    ).filter(comment).first()
    if db_code_detail is None:
        raise HTTPException(status_code=404, detail="Code Detail not found")

    for key, value in updated_detail.dict().items():
        setattr(db_code_detail, key, value)

    db_code_detail.update_user = session_data.user_id
    # db.add(db_code_detail)
    db.flush()
    db.refresh(db_code_detail)
    return db_code_detail

@router.delete("/codedetails/{code_group}/{code_value}", response_model=dict, dependencies=[Depends(cookie)], tags=["CommonCodeDetail"])
def delete_code_detail(code_group: str, code_value: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    comment = text("/* is_endpoint_query */ 1=1")
    db_code_detail = db.query(model_llm.CommonCodeDetail).filter(
        model_llm.CommonCodeDetail.code_group == code_group,
        model_llm.CommonCodeDetail.code_value == code_value
    ).filter(comment).first()
    if db_code_detail is None:
        raise HTTPException(status_code=404, detail="Code Detail not found")

    db.delete(db_code_detail)
    db.flush()
    return {"detail": "Code Detail deleted"}