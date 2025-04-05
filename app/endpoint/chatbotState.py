from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from app.models import ChatbotState
from app.database import SessionLocal, get_db
from app import models, database
from app.model import model_llm
from app.schema import schema_llm
from fastapi import APIRouter
from app.endpoint.login import cookie , SessionData , verifier

router = APIRouter()



# Pydantic models
class StateCreate(BaseModel):
    conversation_id: int
    state_key: str
    state_value: str

class StateUpdate(BaseModel):
    state_value: str

# CRUD operations for ChatbotState
@router.post("/state", response_model=ChatbotState, dependencies=[Depends(cookie)], tags=["ChatbotState"])
def create_state(state: StateCreate, db: Session = Depends(get_db)):
    db_state = model_llm.ChatbotState(conversation_id=state.conversation_id, state_key=state.state_key, state_value=state.state_value)
    db.add(db_state)
    db.flush()
    db.refresh(db_state)
    return db_state

@router.get("/state/{state_id}", response_model=ChatbotState, dependencies=[Depends(cookie)],tags=["ChatbotState"])
def read_state(state_id: int, db: Session = Depends(get_db)):
    db_state = db.query(model_llm.ChatbotState).filter(model_llm.ChatbotState.state_id == state_id).first()
    if db_state is None:
        raise HTTPException(status_code=404, detail="State not found")
    return db_state

@router.get("/state/", response_model=List[ChatbotState], dependencies=[Depends(cookie)],tags=["ChatbotState"])
def read_states(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    states = db.query(model_llm.ChatbotState).offset(skip).limit(limit).all()
    return states

@router.put("/state/{state_id}", response_model=ChatbotState, dependencies=[Depends(cookie)],tags=["ChatbotState"])
def update_state(state_id: int, state: StateUpdate, db: Session = Depends(get_db)):
    db_state = db.query(model_llm.ChatbotState).filter(model_llm.ChatbotState.state_id == state_id).first()
    if db_state is None:
        raise HTTPException(status_code=404, detail="State not found")
    db_state.state_value = state.state_value
    db.flush()
    db.refresh(db_state)
    return db_state

@router.delete("/state/{state_id}", response_model=ChatbotState, dependencies=[Depends(cookie)], tags=["ChatbotState"])
def delete_state(state_id: int, db: Session = Depends(get_db)):
    db_state = db.query(model_llm.ChatbotState).filter(model_llm.ChatbotState.state_id == state_id).first()
    if db_state is None:
        raise HTTPException(status_code=404, detail="State not found")
    db.delete(db_state)
    db.flush()
    return db_state
