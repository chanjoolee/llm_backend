from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from app.models import CommonCodeGroup , CommonCodeGroupCreate , CommonCodeDetail ,CommonCodeDetailCreate
from app.database import SessionLocal, get_db
from app import models, database
from fastapi import APIRouter
from uuid import UUID, uuid4
from app.endpoint.login import cookie , SessionData , verifier
from passlib.context import CryptContext
from app.utils.utils import pwd_context , hash_password ,  verify_password

router = APIRouter()
# app = FastAPI()



@router.get("/codegroups/{code_group}", response_model=CommonCodeGroup, dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def read_code_group(code_group: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    
    query = db.query(database.CommonCodeGroup)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    query = query.filter(database.CommonCodeGroup.code_group == code_group)
    db_code_group = query.first()
    
    if db_code_group is None:
        raise HTTPException(status_code=404, detail="Code Group not found")
    return db_code_group

@router.get("/codegroups", response_model=List[CommonCodeGroup], dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def read_code_groups(db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):

    query = db.query(database.CommonCodeGroup)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    db_code_group = query.all()

    if db_code_group is None:
        raise HTTPException(status_code=404, detail="Code Group not found")
    return db_code_group


@router.post("/codegroups/", response_model=CommonCodeGroup, dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def create_code_group(code_group: CommonCodeGroupCreate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_code_group = database.CommonCodeGroup(**code_group.model_dump())
    db_code_group.create_user = session_data.user_id
    db.add(db_code_group)
    db.flush()
    db.refresh(db_code_group)
    return db_code_group

@router.put("/codegroups/{code_group}", response_model=CommonCodeGroup, dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def update_code_group(code_group: str, updated_group: CommonCodeGroupCreate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    query = query.filter(database.CommonCodeGroup.code_group == code_group)
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
    query = query.filter(database.CommonCodeGroup.code_group == code_group)
    db_code_group = query.first()

    if db_code_group is None:
        raise HTTPException(status_code=404, detail="Code Group not found")

    db.delete(db_code_group)
    db.flush()
    return {"detail": "Code Group deleted"}

@router.get("/codegroups/{code_group}/details", response_model=List[CommonCodeDetail], dependencies=[Depends(cookie)], tags=["CommonCodeGroup"])
def read_code_group_details(code_group: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):

    query = db.query(database.CommonCodeGroup).filter(database.db_comment_endpoint)
    query = query.filter(database.CommonCodeGroup.code_group == code_group)
    db_code_group = query.first()
    
    if db_code_group is None:
        raise HTTPException(status_code=404, detail="Code Group not found")

    return db_code_group.commonCode    

@router.get("/codedetails/{code_group}/{code_value}", response_model=CommonCodeDetail, dependencies=[Depends(cookie)], tags=["CommonCodeDetail"])
def read_code_detail(code_group: str, code_value: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    comment = text("/* is_endpoint_query */ 1=1")
    db_code_detail = db.query(database.CommonCodeDetail).filter(
        database.CommonCodeDetail.code_group == code_group,
        database.CommonCodeDetail.code_value == code_value
    ).filter(comment).first()
    if db_code_detail is None:
        raise HTTPException(status_code=404, detail="Code Detail not found")
    return db_code_detail

@router.post("/codedetails/", response_model=CommonCodeDetail, dependencies=[Depends(cookie)], tags=["CommonCodeDetail"])
def create_code_detail(code_detail: CommonCodeDetailCreate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_code_detail = database.CommonCodeDetail(**code_detail.model_dump())
    db_code_detail.create_user = session_data.user_id
    db.add(db_code_detail)
    db.flush()
    db.refresh(db_code_detail)
    return db_code_detail

@router.put("/codedetails/{code_group}/{code_value}", response_model=CommonCodeDetail, dependencies=[Depends(cookie)],tags=["CommonCodeDetail"])
def update_code_detail(code_group: str, code_value: str, updated_detail: CommonCodeDetailCreate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    comment = text("/* is_endpoint_query */ 1=1")
    db_code_detail = db.query(database.CommonCodeDetail).filter(
        database.CommonCodeDetail.code_group == code_group,
        database.CommonCodeDetail.code_value == code_value
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
    db_code_detail = db.query(database.CommonCodeDetail).filter(
        database.CommonCodeDetail.code_group == code_group,
        database.CommonCodeDetail.code_value == code_value
    ).filter(comment).first()
    if db_code_detail is None:
        raise HTTPException(status_code=404, detail="Code Detail not found")

    db.delete(db_code_detail)
    db.flush()
    return {"detail": "Code Detail deleted"}