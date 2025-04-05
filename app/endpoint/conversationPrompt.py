import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Optional
from app.database import SessionLocal, get_db
from app import models, database
from app.model import model_llm
from app.schema import schema_llm
from sqlalchemy import func , text , select , exists


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()



# Pydantic models for request/response
class ConversationPromptBase(BaseModel):
    conversation_id: str = Field(..., description="ID of the conversation")
    prompt_id: int = Field(..., description="ID of the prompt")
    

class ConversationPromptCreate(ConversationPromptBase):
    pass

class ConversationPromptUpdate(BaseModel):
    sort_order: Optional[int] = Field(None, description="Updated sort order of the prompt in the conversation")

class ConversationPrompt(ConversationPromptBase):
    sort_order: Optional[int] = Field(None, description="Sort order of the prompt in the conversation")

    class Config:
        from_attributes = True

# Create ConversationPrompt entry
@router.post(
    "/conversation_prompts/", 
    response_model=ConversationPrompt, 
    tags=["Conversation Prompt"],
    description="""
    <pre>
    Creates a new association between a conversation and a prompt.
    Request Body:
        - conversation_id (required): The ID of the conversation.
        - prompt_id (required): The ID of the prompt.
    </pre>
    """
)
def create_conversation_prompt(cp: ConversationPromptCreate, db: Session = Depends(get_db)):
    db_cp = schema_llm.ConversationPrompt(**cp.dict())
    # Fetch the current maximum sort_order for the given conversation_id
    max_sort_order = db.query(func.max(model_llm.conversation_prompt.c.sort_order)).filter(model_llm.db_comment_endpoint).filter(model_llm.conversation_prompt.c.conversation_id == cp.conversation_id).scalar()

    # Increment the sort_order
    next_sort_order = 1 if max_sort_order is None else max_sort_order + 1
    db_cp.sort_order= next_sort_order

    db.add(db_cp)
    db.flush()
    db.refresh(db_cp)
    return db_cp

# Read ConversationPrompt entries
@router.get(
    "/conversation_prompts/", 
    response_model=List[ConversationPrompt], 
    tags=["Conversation Prompt"],
    description="""
    <pre>
    Retrieves a list of conversation-prompt associations with pagination.
    Query Parameters:
        - skip (optional): The number of records to skip for pagination.
        - limit (optional): The maximum number of records to return.
    </pre>
    """
)
def read_conversation_prompts(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    results = db.query(schema_llm.ConversationPrompt).filter(model_llm.db_comment_endpoint).offset(skip).limit(limit).all()
    return results

# Update ConversationPrompt entry
@router.put(
    "/conversation_prompts/{conversation_id}/{prompt_id}", 
    response_model=ConversationPrompt, 
    tags=["Conversation Prompt"],
    description="""
    <pre>
    Updates an existing association between a conversation and a prompt.
    Path Parameters:
        - conversation_id: The ID of the conversation.
        - prompt_id: The ID of the prompt.
    Request Body:
        - sort_order (optional): The updated order in which the prompt appears in the conversation.
    </pre>
    """
)
def update_conversation_prompt(conversation_id: str, prompt_id: int, cp: ConversationPromptUpdate, db: Session = Depends(get_db)):
    db_cp = db.query(schema_llm.ConversationPrompt).filter(
        schema_llm.ConversationPrompt.conversation_id == conversation_id,
        schema_llm.ConversationPrompt.prompt_id == prompt_id
    ).filter(model_llm.db_comment_endpoint).first()
    
    if db_cp is None:
        raise HTTPException(status_code=404, detail="Association not found")
    
    update_data = cp.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_cp, key, value)
    
    db.flush()
    db.refresh(db_cp)
    return db_cp

# Delete ConversationPrompt entry
@router.delete(
    "/conversation_prompts/{conversation_id}/{prompt_id}", 
    tags=["Conversation Prompt"],
    description="""
    <pre>
    Deletes a specific association between a conversation and a prompt.
    Path Parameters:
        - conversation_id: The ID of the conversation.
        - prompt_id: The ID of the prompt.
    </pre>
    """
)
def delete_conversation_prompt(conversation_id: str, prompt_id: int, db: Session = Depends(get_db)):
    db_cp = db.query(schema_llm.ConversationPrompt).filter(
        schema_llm.ConversationPrompt.conversation_id == conversation_id,
        schema_llm.ConversationPrompt.prompt_id == prompt_id
    ).filter(model_llm.db_comment_endpoint).first()
    
    if db_cp is None:
        raise HTTPException(status_code=404, detail="Association not found")
    
    db.delete(db_cp)
    db.flush()
    return {"message": "Entry deleted successfully"}
