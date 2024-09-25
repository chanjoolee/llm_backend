from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session , joinedload
from sqlalchemy import or_ , insert , delete , select, text
from typing import List
from app import models, database
from app.database import SessionLocal, get_db
from app.endpoint.login import cookie, SessionData, verifier
import pydash
import re

router = APIRouter()


# VARIABLE_PATTERN = re.compile(r'\{(\w+)\}')
VARIABLE_PATTERN = re.compile(r'(?<!\\)\{(\w+)\}(?!\\)')

def extract_variables(message):
    return VARIABLE_PATTERN.findall(message)

def replace_variables(message, variables_dict):
    """Replace variables in the message using a dictionary of values."""
    
    def replacement(match):
        # Extract the variable name from the match group
        var_name = match.group(1)
        # Replace with the corresponding value from the dictionary
        return variables_dict.get(var_name, match.group(0))  # Default to original if not found

    # Replace all occurrences of the pattern with the corresponding values
    return VARIABLE_PATTERN.sub(replacement, message)

@router.post(
    "/prompts/",
    response_model=models.Prompt,
    dependencies=[Depends(cookie)],
    description="""
    Creates a new prompt with optional messages and tags.
    
    Request Body:
    - prompt_title (str): The title of the prompt.
    - prompt_desc (str): The description of the prompt.
    - open_type (str): The type of visibility (private/public).
    - tag_ids (List[int]): A list of tag IDs to associate with the prompt.
    - promptMessages (Optional[List[PromptMessageCreate]]): A list of prompt messages to create.
    """
)
def create_prompt(prompt: models.PromptCreate, db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    db_prompt = database.Prompt(
        prompt_title=prompt.prompt_title,
        prompt_desc=prompt.prompt_desc,
        open_type=prompt.open_type,
        create_user=session_data.user_id,
        update_user=session_data.user_id
    )
    db.add(db_prompt)
    db.flush()
    db.refresh(db_prompt)
    
    # Add tags to the prompt
    if prompt.tag_ids is not None:
        for tag_id in prompt.tag_ids:
            db_tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
            if db_tag:
                db_prompt.tags.append(db_tag)
    
    # Add prompt messages and extract variables
    existing_variables = set()
    if prompt.promptMessages:
        for message_data in prompt.promptMessages:
            new_message = database.PromptMessages(
                prompt_id=db_prompt.prompt_id,
                message_type=message_data.message_type,
                message=message_data.message,
                create_user=session_data.user_id,
                update_user=session_data.user_id
            )
            db.add(new_message)
            db.flush()
            db.refresh(new_message)
            
            variables = extract_variables(message_data.message)
            for variable in variables:
                if variable not in existing_variables:
                    db_variable = db.query(database.PromptVariable).filter(database.db_comment_endpoint).filter_by(
                        prompt_id=db_prompt.prompt_id,
                        variable_name=variable
                    ).first()
                    if not db_variable:
                        new_variable = database.PromptVariable(
                            prompt_id=db_prompt.prompt_id,
                            variable_name=variable
                        )
                        db.add(new_variable)
                        existing_variables.add(variable)
    
    db.flush()
    db.refresh(db_prompt)
    return db_prompt

@router.get(
    "/prompts/{prompt_id}",
    response_model=models.Prompt,
    dependencies=[Depends(cookie)],
    description="Retrieves a specific prompt by its ID, including related tags and prompt messages."
)
def get_prompt(prompt_id: int, db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    db_prompt = db.query(database.Prompt).options(
        joinedload(database.Prompt.tags),
        joinedload(database.Prompt.promptMessage),
        joinedload(database.Prompt.variables)
    ).filter(database.db_comment_endpoint).filter(database.Prompt.prompt_id == prompt_id).first()
    
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    # Fetch variables
    # variables = db.query(database.PromptVariable).filter(database.PromptVariable.prompt_id == prompt_id).all()
    # db_prompt.variables = [variable.variable_name for variable in variables]

    return db_prompt

@router.get(
    "/prompts/{prompt_id}/variables",
    response_model=List[str],
    dependencies=[Depends(cookie)],
    description="Retrieves variables for a specific prompt by its ID."
)
def get_prompt_variables(prompt_id: int, db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    variables = db.query(database.PromptVariable).filter(database.db_comment_endpoint).filter(database.PromptVariable.prompt_id == prompt_id).all()
    return [variable.variable_name for variable in variables]


@router.put(
    "/prompts/{prompt_id}",
    response_model=models.Prompt,
    dependencies=[Depends(cookie)],
    description="""
    Updates an existing prompt, with optional messages and tags.
    
    Request Body:
    - prompt_title (Optional[str]): The title of the prompt.
    - prompt_desc (Optional[str]): The description of the prompt.
    - open_type (Optional[str]): The type of visibility (private/public).
    - tag_ids (Optional[List[int]]): A list of tag IDs to associate with the prompt.
    - promptMessages (Optional[List[PromptMessageUpdate]]): A list of prompt messages to update or create.
    """
)
def update_prompt(prompt_id: int, prompt: models.PromptUpdate, db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    db_prompt = db.query(database.Prompt).filter(database.db_comment_endpoint).filter(database.Prompt.prompt_id == prompt_id).first()
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    # # If there is conversation message raise 403 error. 수정이 되도록. 다만 기존대화에는 영향을 미치지 않는다.
    # if db_prompt.conversations:
    #     for conversation in db_prompt.conversations:
    #         db_message_filter = pydash.filter_(conversation.messages, {'input_path': 'conversation'})
    #         if db_message_filter:
    #             raise HTTPException(
    #                 status_code=403, 
    #                 detail=f"Conversation (ID: {conversation.conversation_id}, Title: {conversation.conversation_title}) is already started. You cannot update the prompt"
    #             )

    # Update prompt fields
    # params_ex = models.Prompt.dict(exclude_unset=True)
    params_ex = prompt.dict(exclude_unset=True)
    for key, value in params_ex.items():
        if key not in ["tag_ids", "promptMessages"]:
            setattr(db_prompt, key, value)

    db_prompt.update_user = session_data.user_id

    # Add tags to the prompt
    if 'tag_ids' in params_ex and len(params_ex['tag_ids']) > 0 :
        db_prompt.tags = []
        for tag_id in prompt.tag_ids:
            db_tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
            if db_tag:
                db_prompt.tags.append(db_tag)
        
    existing_variables = set()
    if 'promptMessages' in params_ex and len(params_ex['promptMessages']) > 0 :
        # db_prompt.promptMessage = []
        # db_prompt.variables = []
        db.query(database.PromptMessages).filter(database.PromptMessages.prompt_id == prompt_id).delete()
        db.query(database.PromptVariable).filter(database.PromptVariable.prompt_id == prompt_id).delete()

        for message_data in params_ex['promptMessages']:
            new_message = database.PromptMessages(
                prompt_id=prompt_id,
                message_type=message_data['message_type'],
                message=message_data['message'],
                create_user=session_data.user_id,
                update_user=session_data.user_id
            )
            db.add(new_message)
            db.flush()
            
            variables = extract_variables(new_message.message)
            for variable in variables:
                if variable not in existing_variables:
                    db_variable = database.PromptVariable(
                        prompt_id=prompt_id,
                        variable_name=variable
                    )
                    db.add(db_variable)
                    existing_variables.add(variable)

    db.flush()
    db.refresh(db_prompt)
    # # Fetch the updated prompt with related data and variables
    # db_prompt = db.query(database.Prompt).options(
    #     joinedload(database.Prompt.tags),
    #     joinedload(database.Prompt.promptMessage)
    # ).filter(database.Prompt.prompt_id == prompt_id).first()
    
    # variables = db.query(database.PromptVariable).filter(database.PromptVariable.prompt_id == prompt_id).all()
    # # db_prompt.variables = [variable.variable_name for variable in variables]
    
    return db_prompt


@router.delete(
    "/prompts/{prompt_id}",
    response_model=models.Prompt,
    dependencies=[Depends(cookie)],
    description="Deletes a specific prompt by its ID."
)
def delete_prompt(prompt_id: int, db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    db_prompt = db.query(database.Prompt).filter(database.db_comment_endpoint).filter(database.Prompt.prompt_id == prompt_id).first()
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    prompt_data = models.Prompt.from_orm(db_prompt)
    
    # Delete related prompt messages
    # db.query(database.PromptMessages).filter(database.db_comment_endpoint).filter(database.PromptMessages.prompt_id == prompt_id).delete()
    
    # Clear tags associated with the prompt
    # db_prompt.tags = []
    
    db.delete(db_prompt)
    db.flush()
    
    return prompt_data


@router.post(
    "/prompts/search",
    response_model=models.SearchPromptResponse,
    dependencies=[Depends(cookie)],
    description="""
    Searches for prompts based on various criteria.
    
    Request Body:
    - search_words (Optional[str]): The search words to filter prompts by title, description, or message.
    - search_scope (Optional[List[str]]): The scope of the search (e.g., ["name", "desc", "message"]).
    - open_type (Optional[str]): The type of visibility (private/public).
    - tag_ids (Optional[List[int]]): A list of tag IDs to filter prompts.
    - user (Optional[str]): The user who created the prompts.
    - skip (Optional[int]): The number of records to skip for pagination.
    - limit (Optional[int]): The maximum number of records to return for pagination.
    """
)
def search_prompts(search: models.PromptSearch, db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    query = db.query(database.Prompt).filter(database.db_comment_endpoint)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 

    search_exclude = search.dict(exclude_unset=True)
    
    # Filter by search words
    if 'search_words' in search_exclude:
        search_pattern = f"%{search_exclude['search_words']}%"
        if 'search_scope' in search_exclude:
            filters = []
            if "title" in search_exclude['search_scope']:
                filters.append(database.Prompt.prompt_title.like(search_pattern))
            if "description" in search_exclude['search_scope']:
                filters.append(database.Prompt.prompt_desc.like(search_pattern))
            if "message" in search_exclude['search_scope']:
                match_query = text("MATCH(prompt_messages.message) AGAINST(:search_words IN BOOLEAN MODE)")
                subquery = select(database.PromptMessages.prompt_id).filter(
                    match_query.params(search_words=search_exclude['search_words'])
                    , database.Prompt.prompt_id == database.PromptMessages.prompt_id
                ).correlate(database.Prompt).exists()
                filters.append(subquery)
                
            query = query.filter(or_(*filters))
        else:
            query = query.filter(
                or_(
                    database.Prompt.prompt_title.like(search_pattern),
                    database.Prompt.prompt_desc.like(search_pattern),
                    database.Prompt.promptMessage.any(database.PromptMessages.message.like(search_pattern))
                )
            )
    
    # Filter by open type
    if 'open_type' in search_exclude and len(search_exclude['open_type']) > 0:
        query = query.filter(database.Prompt.open_type.in_(search_exclude['open_type']))
    
    # Filter by tags
    if 'tag_ids' in search_exclude and len(search_exclude['tag_ids']) > 0:
        query = query.filter(database.Prompt.tags.any(database.Tag.tag_id.in_(search_exclude['tag_ids'])))
    

    # 사용자 필터링
    user_filter_basic = [
        database.Prompt.create_user == session_data.user_id ,
        database.Prompt.open_type == 'public'
    ]
    query = query.filter(or_(*user_filter_basic))
    if 'user' in search_exclude and len(search_exclude['user']) > 0:
        query = query.filter(database.Prompt.create_user .in_(search_exclude['user']))

    # 사용자 End

    total_count = query.count()
    # Pagination
    if 'skip' in search_exclude and 'limit' in search_exclude:
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit'])
    
    prompts = query.options(
        joinedload(database.Prompt.tags),
        joinedload(database.Prompt.promptMessage)
    ).all()
    return models.SearchPromptResponse(totalCount=total_count, list=prompts)
