from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from app.models import ChatbotConfig , ConfigCreate , ConfigUpdate
from app.database import SessionLocal, get_db
from app import models, database
from app.model import model_llm
from app.schema import schema_llm
from fastapi import APIRouter
from app.endpoint.login import cookie , SessionData , verifier

router = APIRouter()


# Pydantic models
# class ConfigCreate(BaseModel):
#     config_key: str
#     config_value: str

# class ConfigUpdate(BaseModel):
#     config_value: str

# CRUD operations for ChatbotConfig
@router.post("/config", response_model=ChatbotConfig, dependencies=[Depends(cookie)],tags=["ChatbotConfig"])
def create_config(config: ConfigCreate, db: Session = Depends(get_db)):
    db_config = model_llm.ChatbotConfig(config_key=config.config_key, config_value=config.config_value)
    db.add(db_config)
    db.flush()
    db.refresh(db_config)
    return db_config

@router.get("/config/{config_id}", response_model=ChatbotConfig, dependencies=[Depends(cookie)], tags=["ChatbotConfig"])
def read_config(config_id: int, db: Session = Depends(get_db)):
    db_config = db.query(model_llm.ChatbotConfig).filter(model_llm.ChatbotConfig.config_id == config_id).first()
    if db_config is None:
        raise HTTPException(status_code=404, detail="Config not found")
    return db_config

@router.put("/config/{config_id}", response_model=ChatbotConfig,dependencies=[Depends(cookie)], tags=["ChatbotConfig"])
def update_config(config_id: int, config: ConfigUpdate, db: Session = Depends(get_db)):
    db_config = db.query(model_llm.ChatbotConfig).filter(model_llm.ChatbotConfig.config_id == config_id).first()
    if db_config is None:
        raise HTTPException(status_code=404, detail="Config not found")
    db_config.config_value = config.config_value
    db.flush()
    db.refresh(db_config)
    return db_config

@router.delete("/config/{config_id}", response_model=ChatbotConfig,dependencies=[Depends(cookie)], tags=["ChatbotConfig"])
def delete_config(config_id: int, db: Session = Depends(get_db)):
    db_config = db.query(model_llm.ChatbotConfig).filter(model_llm.ChatbotConfig.config_id == config_id).first()
    if db_config is None:
        raise HTTPException(status_code=404, detail="Config not found")
    db.delete(db_config)
    db.flush()
    return db_config
