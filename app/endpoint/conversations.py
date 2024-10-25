import logging
from fastapi import FastAPI, Depends, HTTPException, Query, logger
from sqlalchemy.orm import Session , joinedload , selectinload
from sqlalchemy import and_, func, or_ , text , select , exists , insert , delete 
from pydantic import BaseModel
from typing import List , Optional
# import ai_core.conversation
from ai_core.tool.base import load_tool
from app.endpoint.prompt import replace_variables
from app.endpoint.tools import construct_file_save_path, convert_db_tool_to_pydantic
from app.models import Conversation , ConversationCreate , ConversationCopy, ConversationSearch, MessageCreate, Message
from app.database import SessionLocal, get_db
from app import models, database
from fastapi import APIRouter
from urllib.parse import quote
from app.endpoint.login import cookie , SessionData , verifier
# import ai_core
from ai_core.conversation.base import ConversationFactory
import uuid
import os
from app.utils import utils
from app.utils.utils import pwd_context , hash_password ,  verify_password
import pydash

router = APIRouter()



@router.post(
    "/create_conversation",
    response_model=models.Conversation,
    dependencies=[Depends(cookie)],
    tags=["Conversations"],
    description="""<pre>
    <h3>
    Creates a new conversation with prompts and variable replacements.
    Request Body:
        - conversation_title (str): The title of the conversation.
        - conversation_type (str): The type of conversation (private/public).
        - llm_api_id (Optional[int]): The ID of the LLM API.
        - llm_model (str): The LLM Model. One of the values from the LLM API.
        - temperature (float): The temperature setting for the LLM model.
        - max_tokens (int): The maximum number of tokens for the LLM model.
        - component_configuration : (none,component,agent,all)
        - prompts (Optional[List[ConversatonPromptCreate]]): A list of prompts with variable replacements.
            - prompt_id (int): The ID of the prompt.
            - variables (Optional[List[PromptVariableValue]]): Optional. A list of variables for the prompt.
                - variable_name (str): The name of the variable.
                - value (str): The value of the variable.
        - tools (List[int]): A list of tool IDs to associate with the conversation.
        - agents (List[int]): A list of agent IDs to associate with the conversation.
        - datasources (List[int]): A list of agent IDs to associate with the conversation.
    Returns:
        - Conversation: The created conversation object.
    </h3>
    </pre>
    """
)
def create_conversation(
    conversation: models.ConversationCreate,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    
    # coversation_id 성성
    conversation_id = utils.generate_conversation_id()
    # converation insert
    # db_conversation = database.Conversation(
    #     **conversation.model_dump() ,
    #     user_id = session_data.user_id ,
    #     conversation_id = conversation_id
    # )
    # db_conversation.conversation_id = conversation_id
    db_conversation = database.Conversation(
        user_id = session_data.user_id ,
        conversation_id = conversation_id
    )
    update_data = conversation.dict(exclude_unset=True)
    for key, value in update_data.items():
        if key not in ["prompts", "tools","agents","datasources"]:
            setattr(db_conversation, key, value)

    db.add(db_conversation)
    db.flush()
    db.refresh(db_conversation)

    # 프롬프트 저정
    
    if 'prompts' in update_data:
        for i, prompt in enumerate(conversation.prompts):
            db_prompt = db.query(database.Prompt).options(
                joinedload(database.Prompt.promptMessage)
            ).filter(database.db_comment_endpoint).filter(database.Prompt.prompt_id == prompt.prompt_id).first()
            
            if db_prompt:
                # insert conversation_prompt
                # db_conversation.prompts.append(db_prompt)
                stmt = insert(database.conversation_prompt).values(
                    conversation_id=db_conversation.conversation_id,
                    prompt_id=db_prompt.prompt_id,
                    sort_order=i+1
                )
                db.execute(stmt)

                # variable
                existing_variables = set()
                for variable in prompt.variables:
                    # 여기서 replace 하는 로직은 제거
                    # message = message.replace(f"{{{variable.variable_name}}}", variable.value)
                    
                    # conversation variable 에 저장한다.
                    if variable.variable_name not in existing_variables:
                        db_variable = database.ConversationVariable(
                            conversation_id=db_conversation.conversation_id,
                            variable_name=variable.variable_name,
                            variable_value=variable.value
                        )
                        db.add(db_variable)
                        existing_variables.add((variable.variable_name,variable.value))
                    
                variables_dict = dict(existing_variables)
                # message add
                for db_prompt_message in db_prompt.promptMessage:
                    message = db_prompt_message.message
                    # for var, value in variables_dict.items():
                    #     message = message.replace(f'{{{var}}}', value)
                    replaced_message = replace_variables(message, variables_dict)
                    # 변수를 replace 하자
                    new_message = database.Message(
                        conversation_id=db_conversation.conversation_id,
                        message_type=db_prompt_message.message_type,
                        message=replaced_message,
                        input_path = 'prompt'
                        # , create_user=session_data.user_id
                    )
                    # add database

                    db.add(new_message)
    
    if update_data['component_configuration'] == 'all':
        """
        공개된 도구와 
        공개된 데이터 소스.
        """
        # tool 추가 공개된 tool 을 넣는다.
        logger.info(f"tool 추가")
        tools_public = db.query(database.Tool).filter(database.Tool.visibility=='public').all()
        
        for db_tool in tools_public:
            db_conversation.tools.append(db_tool)
                
        logger.info(f"datasource 추가")
        datasource_public = db.query(database.DataSource).filter(database.DataSource.visibility=='public').all()
        
        for db_datasouce in datasource_public:
            db_conversation.datasources.append(db_datasouce)
        
    else:

        # Add tools to the conversation
        if 'tools' in update_data:
            for tool_id in conversation.tools:
                db_tool = db.query(database.Tool).filter(database.db_comment_endpoint).filter(database.Tool.tool_id == tool_id).first()
                if db_tool:
                    db_conversation.tools.append(db_tool)

        # Add agents to the conversation
        if 'agents' in update_data:
            for agent_id in conversation.agents:
                db_agent = db.query(database.Agent).filter(database.db_comment_endpoint).filter(database.Agent.agent_id == agent_id).first()
                if db_agent:
                    db_conversation.agents.append(db_agent)
                    
        if 'datasources' in update_data:
            for datasource_id in conversation.datasources:
                db_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
                if db_datasource:
                    db_conversation.datasources.append(db_datasource)
    
    db.flush()
    db.refresh(db_conversation)

    return db_conversation

@router.post(
    "/copy_conversation",
    response_model=models.Conversation,
    dependencies=[Depends(cookie)],
    tags=["Conversations"],
    description="""
    <pre>
    Copies an existing conversation to a new one.

    Request Body:
    - conversation_id_origin (str): The ID of the original conversation to copy.

    Returns:
    - Conversation: The copied conversation object.
    </pre>
    """
)
async def copy_conversation(
    conversation: models.ConversationCopy,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    # 원본
    db_conversation_origin = db.query(database.Conversation).filter(database.db_comment_endpoint).filter(database.Conversation.conversation_id == conversation.conversation_id_origin).first()
    if not db_conversation_origin:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db_message_origin = db.query(database.Message).filter(database.db_comment_endpoint).filter(database.Message.conversation_id == conversation.conversation_id_origin).order_by(database.Message.message_id.asc()).all()
    
    
    
    # coversation_id 성성
    conversation_id = utils.generate_conversation_id()

    # converation insert
    exclude_columns = {'conversation_id', 'conversation_title','user_id','last_message_id'}
    new_db_conversation = database.Conversation(
        **{
            column.name: getattr(db_conversation_origin, column.name)
            for column in database.Conversation.__table__.columns
            if column.name not in exclude_columns
        },
        user_id = session_data.user_id ,
        conversation_id = conversation_id,
        conversation_title = f'{db_conversation_origin.conversation_title} copy'
    )
    
    db.add(new_db_conversation)
    db.flush()
    db.refresh(new_db_conversation)


    # Insert the related messages with the new conversation ID
    last_message_id = None
    new_db_messages = []
    for message in db_message_origin:
        new_message = database.Message(
            conversation_id=conversation_id,
            message_type=message.message_type,
            message=message.message,
            sent_at=message.sent_at
        )
        new_db_messages.append(new_message)
        db.add(new_message)
        db.flush()
        db.refresh(new_message)
        last_message_id = new_message.message_id

    # Update the new conversation with the last message ID
    new_db_conversation.last_message_id = last_message_id
    db.flush()
    db.refresh(new_db_conversation)

    conversation_instance = ConversationFactory.create_conversation(
        llm_api_provider=db_conversation_origin.llm_api.llm_api_provider,
        llm_model=db_conversation_origin.llm_model,
        llm_api_key=db_conversation_origin.llm_api.llm_api_key,
        llm_api_url=db_conversation_origin.llm_api.llm_api_url,
        temperature=db_conversation_origin.temperature,
        max_tokens=db_conversation_origin.max_tokens,
        sync_conn_pool=database.sync_conn_pool, 
        async_conn_pool=database.async_conn_pool
    )    
    await conversation_instance.create_agent()
    await conversation_instance.copy_conversation(conversation_id=conversation.conversation_id_origin,new_conversation_id=conversation_id)

    # 프롬프트

    # 도구

    return new_db_conversation

@router.post(
    "/convert_to_private",
    response_model=models.Conversation,
    dependencies=[Depends(cookie)],
    tags=["Conversations"],
    description="""
    <pre>    
    Public 대화를 Private 대화로 바꾼다.
    공용대화인데 본인것이 아닌대화를 이어나갈 경우에 해당한다.

    Request Body:
    - conversation_id_origin (str): The ID of the original conversation to copy.

    Returns:
    - Conversation: The copied conversation object.
    </pre>
    """
)
async def convert_to_private(
    conversation: models.ConversationCopy,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    # 원본
    db_conversation_origin = db.query(database.Conversation).filter(database.db_comment_endpoint).filter(database.Conversation.conversation_id == conversation.conversation_id_origin).first()
    if not db_conversation_origin:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db_message_origin = db.query(database.Message).filter(database.db_comment_endpoint).filter(database.Message.conversation_id == conversation.conversation_id_origin).order_by(database.Message.message_id.asc()).all()
    
    # coversation_id 성성
    conversation_id = utils.generate_conversation_id()

    # converation insert
    exclude_columns = {'conversation_id', 'conversation_title','user_id','last_message_id', 'conversation_type'}
    new_db_conversation = database.Conversation(
        **{
            column.name: getattr(db_conversation_origin, column.name)
            for column in database.Conversation.__table__.columns
            if column.name not in exclude_columns
        },
        conversation_type = 'private',
        user_id = session_data.user_id ,
        conversation_id = conversation_id,
        conversation_title = f'{db_conversation_origin.conversation_title} copy'
    )
    
    db.add(new_db_conversation)
    db.flush()
    db.refresh(new_db_conversation)

    # prompt
    for i, db_prompt in enumerate(db_conversation_origin.prompts):
        # Insert conversation_prompt
        # stmt = insert(database.conversation_prompt).values(
        #     conversation_id=db_conversation.conversation_id,
        #     prompt_id=db_prompt.prompt_id,
        #     sort_order=i+1
        # )
        # db.execute(stmt)
        new_db_conversation.prompts.append(db_prompt)

    # variable
    for i, db_variable in enumerate(db_conversation_origin.variables):
        new_db_variable = database.ConversationVariable(
            conversation_id=new_db_conversation.conversation_id,
            variable_name=db_variable.variable_name,
            variable_value=db_variable.variable_value
        )
        db.add(new_db_variable)

    # tools 
    tools_copy = db_conversation_origin.tools
    if db_conversation_origin.user_id == 'system': 
        # 시스템이면 
        tools_copy = db.query(database.Tool).filter(database.Tool.visibility=='public').all()
        for db_tool in tools_copy:
            try:
                pydantic_tool = convert_db_tool_to_pydantic(db_tool)
                file_save_path = construct_file_save_path(pydantic_tool)
                tool = load_tool(db_tool.name, session_data.nickname, file_save_path)
                new_db_conversation.tools.append(db_tool)
                # new_db_conversation.tools.append(db_tool)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
            

    # agents
    for db_agent in db_conversation_origin.agents:
        new_db_conversation.agents.append(db_agent)
        
    for db_datasource in db_conversation_origin.datasources:
        new_db_conversation.datasources.append(db_datasource)

    # db.refresh(new_db_conversation)

    # Insert the related messages with the new conversation ID
    last_message_id = None
    new_db_messages = []
    for message in db_message_origin:
        input_path = message.input_path

        if input_path == 'system':
            input_path = 'system_copy'

        new_message = database.Message(
            conversation_id=new_db_conversation.conversation_id,
            message_type=message.message_type,
            message=message.message,
            sent_at=message.sent_at,
            input_path=input_path
        )
        new_db_messages.append(new_message)
        db.add(new_message)

    # Update the new conversation with the last message ID
    db.flush()
    if len(db_message_origin) > 0 :
        last_message_id = new_db_messages[-1].message_id
    
    new_db_conversation.last_message_id = last_message_id
    db.flush()
    db.refresh(new_db_conversation)

    conversation_instance = ConversationFactory.create_conversation(
        llm_api_provider=db_conversation_origin.llm_api.llm_api_provider,
        llm_model=db_conversation_origin.llm_model,
        llm_api_key=db_conversation_origin.llm_api.llm_api_key,
        llm_api_url=db_conversation_origin.llm_api.llm_api_url,
        temperature=db_conversation_origin.temperature,
        max_tokens=db_conversation_origin.max_tokens,
        sync_conn_pool=database.sync_conn_pool, 
        async_conn_pool=database.async_conn_pool
    )    
    await conversation_instance.create_agent()
    await conversation_instance.copy_conversation(conversation_id=conversation.conversation_id_origin,new_conversation_id=conversation_id)


    db.flush()
    return new_db_conversation



# 대화제목자동생성
@router.post(
    "/generate_title", 
    response_model=Conversation, 
    description='''<pre>
    
    conversation_id(필수)
        타이틀을 자동생성하고자 하는 conversation_id
        ai_core.conversation.Conversation.generate_title 을 사용함.
    
</pre>''', 
    dependencies=[Depends(cookie)], 
    tags=["Conversations"]
)
async def generate_title(conversation: models.ConversationGenerateTitle, db: Session = Depends(get_db), session_data: SessionData = Depends(verifier)):
    # 원본
    db_conversation = db.query(database.Conversation).options(
        joinedload(database.Conversation.messages)
        , joinedload(database.Conversation.llm_api)
    ).filter(database.db_comment_endpoint).filter(database.Conversation.conversation_id == conversation.conversation_id).first()
    if not db_conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation_instance = ConversationFactory.create_conversation(
        llm_api_provider=db_conversation.llm_api.llm_api_provider,
        llm_model=db_conversation.llm_model,
        llm_api_key=db_conversation.llm_api.llm_api_key,
        llm_api_url=db_conversation.llm_api.llm_api_url,
        temperature=db_conversation.temperature,
        max_tokens=db_conversation.max_tokens,
        sync_conn_pool=database.sync_conn_pool, 
        async_conn_pool=database.async_conn_pool
    )
    await conversation_instance.create_agent()

    conversation_title_new = await conversation_instance.generate_title(conversation_id=conversation.conversation_id)
    db_conversation.conversation_title = conversation_title_new
    db.flush()
    db.refresh(db_conversation)
    return db_conversation


@router.get(
    "/conversations/{conversation_id}",
    response_model=Conversation, 
    dependencies=[Depends(cookie)], 
    tags=["Conversations"]
)
def read_conversation(conversation_id: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    query = db.query(database.Conversation)
    query = query.filter(database.db_comment_endpoint)
    query = query.filter(database.Conversation.conversation_id == conversation_id)
    
    query = query.options(
        # selectinload(database.Conversation.messages),
        selectinload(database.Conversation.prompts),
        selectinload(database.Conversation.tools)
            .selectinload(database.Tool.tags),
        selectinload(database.Conversation.llm_api)
            .selectinload(database.LlmApi.create_user_info),
        selectinload(database.Conversation.user_info),
        selectinload(database.Conversation.variables),
        selectinload(database.Conversation.agents)
    )

    db_conversation = query.first()

    if db_conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return db_conversation

@router.get(
    "/conversations", 
    response_model=models.SearchConversationsResponse, 
    dependencies=[Depends(cookie)],
    tags=["Conversations"]
)
def read_conversations(
    skip: int = Query(0, description="Number of records to skip for pagination"), 
    limit: int = Query(10, description="Maximum number of records to return"), 
    db: Session = Depends(get_db), 
    session_data: SessionData = Depends(verifier)
):
    query = db.query(database.Conversation)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment) 
    total_count = query.count()

    if skip is not None and limit is not None:
        query = query.offset(skip).limit(limit)
    
    conversations = query.options(
        joinedload(database.Conversation.messages) ,
        joinedload(database.Conversation.llm_api)
    ).all()
    # return conversations
    return models.SearchConversationsResponse(totalCount=total_count, list=conversations)




@router.post(
    "/search_conversation",
    response_model=models.SearchConversationsResponse,
    dependencies=[Depends(cookie)],
    tags=["Conversations"],
    description="""<pre>
    Searches for conversations based on various criteria.
    <h3>Request Body:</h3>
        - search_words (Optional): Words to search for in titles and messages.
        - started_at (Optional): The start date from which to search for conversations.
        - last_conversation_time (Optional): The end date until which to search for conversations.
        - search_range_list (Optional): List of fields to search within 
            (e.g., 'title', 'message').
        - conversation_type_list (Optional): List of conversation types to filter by.
            - (e.g., 'public','private')
        - component_list (Optional): List of components to filter by.
        - user_list (Optional): List of users to filter by.
        - llm_api_list (Optional): List of LLM API to filter by.
            (e.g., [1,2,10]).
        - llm_model_list (Optional): List of LLM models to filter by.
            (e.g., 'gpt-4o','gpt-4','gpt-3.5').
        - datasource_list (Optional): List of Data Source to filter by.
            (e.g., ['ds-user1-confluence_0920_001','ds-user1-docx_0919_010','ds-user1-text_0909']).
        - skip (Optional): The number of records to skip for pagination.
        - limit (Optional): The maximum number of records to return for pagination.
    </pre>
    """
)
def search_conversation(
    search: ConversationSearch,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(database.Conversation)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    search_exclude = search.dict(exclude_unset=True)
    

    if 'started_at' in search_exclude:
        query = query.filter(database.Conversation.started_at >= search_exclude['started_at'])
    
    if 'last_conversation_time' in search_exclude:
        query = query.filter(database.Conversation.last_conversation_time <= search_exclude['last_conversation_time'])
    
    # 검색범위 제목, 메세지
    if (
        'search_words' in search_exclude and 
        search_exclude['search_words'] != "" and 
        'search_range_list' in search_exclude  and 
        len(search_exclude['search_range_list']) > 0 
    ) :
        search_filter_word = []
        if 'title' in search_exclude['search_range_list']: 
            filter = database.Conversation.conversation_title.like(f'%{search_exclude['search_words']}%')
            search_filter_word.append(filter)
            # query = query.filter(database.Conversation.conversation_title.like(f'%{search_exclude['search_words']}%'))

        if 'message' in search_exclude['search_range_list']:                    
            # in this place I will join message and query message.message
            # query = query.join(database.Message).filter(func.MATCH(database.Message.message).AGAINST(search.search_words))
            match_query = text("MATCH(messages.message) AGAINST(:search_words IN BOOLEAN MODE)")
            # query = query.join(database.Message).filter(match_query.params(search_words=search.search_words))

            subquery = select(database.Message.conversation_id).filter(
                match_query.params(search_words=search.search_words)
                , database.Conversation.conversation_id == database.Message.conversation_id
            ).correlate(database.Conversation).exists()
            # query = query.filter(subquery)
            search_filter_word.append(subquery)
        query = query.filter(or_(*search_filter_word))

        # else:
        #     match_query = text("MATCH(messages.message) AGAINST(:search_words IN BOOLEAN MODE)")
        #     subquery = select(database.Message.conversation_id).filter(
        #         match_query.params(search_words=search.search_words)
        #         , database.Conversation.conversation_id == database.Message.conversation_id
        #     ).correlate(database.Conversation).exists()
        #     query = query.filter(or_(
        #         match_query,
        #         subquery
        #     ))
    # 공개 여부
    if 'conversation_type_list' in search_exclude and len(search_exclude['conversation_type_list']) > 0:
        query = query.filter(database.Conversation.conversation_type.in_(search_exclude['conversation_type_list']))
    
    # # 콤포넌트 나중에 
    # if search.component_list:
    #     query = query.filter(Conversation.components.in_(search.component_list))
    
    # 사용자 필터링
    user_filter_basic = [
        database.Conversation.user_id == session_data.user_id,
        database.Conversation.conversation_type == 'public'
    ]
    query = query.filter(or_(*user_filter_basic))
    if 'user_list' in search_exclude and len(search_exclude['user_list']) > 0:
        query = query.filter(database.Conversation.user_id.in_(search_exclude['user_list']))
        # 참고
        # public_user_filter = text("""(
        # conversations.user_id in :user_list 
        # and conversations.conversation_type = 'public'
        # )""")
        # user_filter.append(public_user_filter.params(user_list=search_exclude['user_list']))
    
    # 사용자 End
    
    if 'llm_api_list' in search_exclude and len(search_exclude['llm_api_list']) > 0:
        query = query.filter(database.Conversation.llm_api_id.in_(search_exclude['llm_api_list']))
        # # sub쿼리로 한다.  여기할차례  llm_api에 exits 로 한다.
        # subquery = select(database.LlmApi.llm_api_id).filter(
        #     database.Conversation.llm_api_id == database.LlmApi.llm_api_id
        #     , database.LlmApi.llm_api_id.in_(search.llm_api_list)
        # ).correlate(database.Conversation).exists()
        # query = query.filter(subquery)
    
    if 'llm_model_list' in search_exclude and len(search_exclude['llm_model_list']) > 0:
        # llm api 에서는 쉼표로 구분하여 입력한다.
        # subquery = select(database.LlmApi.llm_api_id).filter(
        #     database.Conversation.llm_api_id == database.LlmApi.llm_api_id
        #     , database.LlmApi.llm_model.in_(search.llm_model_list)
        # ).correlate(database.Conversation).exists()
        # query = query.filter(subquery)

        # 본테이블에서 검색한다.
        query = query.filter(database.Conversation.llm_model.in_(search_exclude['llm_model_list']))
        
    if 'datasource_list' in search_exclude and len(search_exclude['datasource_list']) > 0:
        query = query.filter(database.Conversation.datasources.in_(search_exclude['datasource_list']))
 
    total_count = query.count()



    query = query.order_by(database.Conversation.last_conversation_time.desc())   
    # Apply pagination
    if 'skip' in search_exclude and 'limit' in search_exclude :
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit'])

    # conversations = query.options(
    #     joinedload(database.Conversation.messages) ,
    #     joinedload(database.Conversation.llm_api)
    # ).all()
    query = query.options(
        selectinload(database.Conversation.messages),
        selectinload(database.Conversation.prompts),
        selectinload(database.Conversation.tools)
            .selectinload(database.Tool.tags),
        selectinload(database.Conversation.llm_api)
            .selectinload(database.LlmApi.create_user_info),
        selectinload(database.Conversation.user_info),
        selectinload(database.Conversation.variables),
        selectinload(database.Conversation.agents),
        selectinload(database.Conversation.datasources)
    )
    conversations = query.all()

    # Convert SQLAlchemy models to Pydantic models
    # pydantic_conversations = [Conversation.model_validate(conversation) for conversation in conversations]

    return models.SearchConversationsResponse(totalCount=total_count, list=conversations)
    # return conversations
    # return [models.Conversation.model_validate(conversation) for conversation in conversations]

@router.post(
    "/search_conversation_all",
    response_model=models.SearchConversationsResponse,
    dependencies=[Depends(cookie)],
    tags=["Conversations"],
    description="""<pre>
    Searches for conversations based on various criteria.
    <h3>Request Body:</h3>
        - search_words (Optional): Words to search for in titles and messages.
        - started_at (Optional): The start date from which to search for conversations.
        - last_conversation_time (Optional): The end date until which to search for conversations.
        - search_range_list (Optional): List of fields to search within 
            (e.g., 'title', 'message').
        - conversation_type_list (Optional): List of conversation types to filter by.
            - (e.g., 'public','private')
        - component_list (Optional): List of components to filter by.
        - user_list (Optional): List of users to filter by.
        - llm_api_list (Optional): List of LLM API to filter by.
            (e.g., [1,2,10]).
        - llm_model_list (Optional): List of LLM models to filter by.
            (e.g., 'gpt-4o','gpt-4','gpt-3.5').
        - datasource_list (Optional): List of Data Source to filter by.
            (e.g., ['ds-user1-confluence_0920_001','ds-user1-docx_0919_010','ds-user1-text_0909']).
        - skip (Optional): The number of records to skip for pagination.
        - limit (Optional): The maximum number of records to return for pagination.
    </pre>
    """
)
def search_conversation_all(
    search: ConversationSearch,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(database.Conversation)
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    search_exclude = search.dict(exclude_unset=True)
    

    if 'started_at' in search_exclude:
        query = query.filter(database.Conversation.started_at >= search_exclude['started_at'])
    
    if 'last_conversation_time' in search_exclude:
        query = query.filter(database.Conversation.last_conversation_time <= search_exclude['last_conversation_time'])
    
    # 검색범위 제목, 메세지
    if (
        'search_words' in search_exclude and 
        search_exclude['search_words'] != "" and 
        'search_range_list' in search_exclude  and 
        len(search_exclude['search_range_list']) > 0 
    ) :
        search_filter_word = []
        if 'title' in search_exclude['search_range_list']: 
            filter = database.Conversation.conversation_title.like(f'%{search_exclude['search_words']}%')
            search_filter_word.append(filter)
            # query = query.filter(database.Conversation.conversation_title.like(f'%{search_exclude['search_words']}%'))

        if 'message' in search_exclude['search_range_list']:                    
            # in this place I will join message and query message.message
            # query = query.join(database.Message).filter(func.MATCH(database.Message.message).AGAINST(search.search_words))
            match_query = text("MATCH(messages.message) AGAINST(:search_words IN BOOLEAN MODE)")
            # query = query.join(database.Message).filter(match_query.params(search_words=search.search_words))

            subquery = select(database.Message.conversation_id).filter(
                match_query.params(search_words=search.search_words)
                , database.Conversation.conversation_id == database.Message.conversation_id
            ).correlate(database.Conversation).exists()
            # query = query.filter(subquery)
            search_filter_word.append(subquery)
        query = query.filter(or_(*search_filter_word))

    # 공개 여부
    if 'conversation_type_list' in search_exclude and len(search_exclude['conversation_type_list']) > 0:
        query = query.filter(database.Conversation.conversation_type.in_(search_exclude['conversation_type_list']))
    
    # # 콤포넌트 나중에 
    # if search.component_list:
    #     query = query.filter(Conversation.components.in_(search.component_list))
    

    if 'user_list' in search_exclude and len(search_exclude['user_list']) > 0:
        query = query.filter(database.Conversation.user_id.in_(search_exclude['user_list']))
    
    # 사용자 End
    
    if 'llm_api_list' in search_exclude and len(search_exclude['llm_api_list']) > 0:
        query = query.filter(database.Conversation.llm_api_id.in_(search_exclude['llm_api_list']))
    
    if 'llm_model_list' in search_exclude and len(search_exclude['llm_model_list']) > 0:
        query = query.filter(database.Conversation.llm_model.in_(search_exclude['llm_model_list']))
        
    if 'datasource_list' in search_exclude and len(search_exclude['datasource_list']) > 0:
        query = query.filter(database.Conversation.datasources.in_(search_exclude['datasource_list']))
        
    total_count = query.count()

    query = query.options(
        selectinload(database.Conversation.messages),
        selectinload(database.Conversation.prompts),
        selectinload(database.Conversation.tools)
            .selectinload(database.Tool.tags),
        selectinload(database.Conversation.llm_api)
            .selectinload(database.LlmApi.create_user_info),
        selectinload(database.Conversation.user_info),
        selectinload(database.Conversation.variables),
        selectinload(database.Conversation.agents),
        selectinload(database.Conversation.datasources)
    )

    query = query.order_by(database.Conversation.last_conversation_time.desc())   
    # Apply pagination
    if 'skip' in search_exclude and 'limit' in search_exclude :
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit'])

    # conversations = query.options(
    #     joinedload(database.Conversation.messages) ,
    #     joinedload(database.Conversation.llm_api)
    # ).all()
    conversations = query.all()

    # Convert SQLAlchemy models to Pydantic models
    # pydantic_conversations = [Conversation.model_validate(conversation) for conversation in conversations]

    return models.SearchConversationsResponse(totalCount=total_count, list=conversations)
    # return conversations
    # return [models.Conversation.model_validate(conversation) for conversation in conversations]


@router.put(
    "/conversations/{conversation_id}", 
    response_model=Conversation, 
    dependencies=[Depends(cookie)],
    tags=["Conversations"],
    description="""<pre>
    <h3>
    Request Body:
        - user_id (Optional[str]): 사용자아이디
        - conversation_title (Optional[str]): 대화제목
        - conversation_type (Optional[str]): 대화종류 private public
        - llm_api_id (Optional[int]): 대화종류 private public
        - llm_model (Optional[str]): LLM Model. LLM API에서 입력된 값중 하나만 입력
        - temperature (Optional[float]): 
        - max_tokens (Optional[int]): 최대토큰
        - used_tokens (Optional[int]): 토큰사용량
        - last_conversation_time (Optional[datetime]): 마지막 대화시간
        - last_message_id (Optional[int]): 메세지ID
        - started_at (Optional[datetime]): 생성일
        - component_configuration : (none,component,agent,all)
        - prompts (List[ConversatonPromptCreate]): A list of prompts with variable replacements.
            - prompt_id (int): The ID of the prompt.
            - variables (Optional[List[PromptVariableValue]]): Optional. A list of variables for the prompt.
                - variable_name (str): The name of the variable.
                - value (str): The value of the variable.
        - tools (List[int]): A list of tool IDs to associate with the conversation.
        - agents (List[int]): A list of agent IDs to associate with the conversation.
        - datasources (List[int]): A list of agent IDs to associate with the conversation.
    </h3>
    </pre>
    """
)
def update_conversation(conversation_id: str, conversation: models.ConversationUpdate, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    """
    01. 프롬프트 데이타가 있다면, 대화가 시작되었으면 중단한다.
    02. 기존 테이블로직 삭제
        - conversation_prompts
        - conversation_variables
    03. 신규생성 로직과 동일하게 진행
        conversation.prompts
            promptMessage

    """
    
    db_conversation = db.query(database.Conversation).filter(database.db_comment_endpoint).filter(database.Conversation.conversation_id == conversation_id).first()
    if db_conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    params_ex = conversation.model_dump(exclude_unset=True)

    # 프롬프트 데이타가 있다면, 대화가 시작되었으면 중단한다.
    if 'prompt' in params_ex and len(params_ex['prompt']) > 0 :
        db_message_filter = pydash.filter_(db_conversation.messages, {'input_path':'conversation'})
        if db_message_filter:
            raise HTTPException(status_code=403, detail="Conversation is aleady started. You can not update conversation")

    
    for key, value in params_ex.items():
        if key not in ["prompts", "tools","agents","datasources"]:
            setattr(db_conversation, key, value)
    


    """
    03. Proceed in the same way as the new creation logic
        conversation.prompts
            promptMessage
    """
    # Save prompts
    # [] 로 보낸다면 삭제하겠다는 의미
    if 'prompts' in params_ex :
        """
        02. 기존 테이블로직 삭제
            - conversation_prompts
            - conversation_variables
        """
        # Delete related conversation_prompts
        # stmt = delete(database.conversation_prompt).where(database.conversation_prompt.c.conversation_id == conversation_id)
        # db.execute(stmt)
        db_conversation.prompts = []
        # Delete related conversation_variables. many to many 가 아니므로 명시적으로 삭제한다.
        db.query(database.ConversationVariable).filter(database.db_comment_endpoint).filter(database.ConversationVariable.conversation_id == conversation_id).delete(synchronize_session='fetch')
        # db.flush()
        db.flush()

        for i, prompt in enumerate(params_ex['prompts']):
            db_prompt = db.query(database.Prompt).options(
                joinedload(database.Prompt.promptMessage)
            ).filter(database.db_comment_endpoint).filter(database.Prompt.prompt_id == prompt.prompt_id).first()
            
            if db_prompt:
                # Insert conversation_prompt
                stmt = insert(database.conversation_prompt).values(
                    conversation_id=db_conversation.conversation_id,
                    prompt_id=db_prompt.prompt_id,
                    sort_order=i+1
                )
                db.execute(stmt)

                # variable
                existing_variables = set()
                for variable in prompt.variables:
                    if variable.variable_name not in existing_variables:
                        db_variable = database.ConversationVariable(
                            conversation_id=db_conversation.conversation_id,
                            variable_name=variable.variable_name,
                            variable_value=variable.value
                        )
                        db.add(db_variable)
                        existing_variables.add((variable.variable_name,variable.value))

                variables_dict = dict(existing_variables)
                # message add. 대화가 시작된 이후에는 대화수정 불가
                for db_prompt_message in db_prompt.promptMessage:
                    message = db_prompt_message.message
                    # for var, value in variables_dict.items():
                    #     message = message.replace(f'{{{var}}}', value)
                    replaced_message = replace_variables(message, variables_dict)
                    # 변수를 replace 하자
                    new_message = database.Message(
                        conversation_id=db_conversation.conversation_id,
                        message_type=db_prompt_message.message_type,
                        message=replaced_message,
                        input_path='prompt'
                    )
                    db.add(new_message)

                
    # db.flush()

    # Remove existing tools and add new ones
    # [] 로 보낸다면 삭제하겠다는 의미
    if params_ex['component_configuration'] == 'all':
        """
        공개된 도구와 
        공개된 데이터 소스.
        """
        # tool 추가 공개된 tool 을 넣는다.
        logger.info(f"tool 추가")
        tools_public = db.query(database.Tool).filter(database.Tool.visibility=='public').all()
        db_conversation.tools = []
        for db_tool in tools_public:
            db_conversation.tools.append(db_tool)
                
        logger.info(f"datasource 추가")
        datasource_public = db.query(database.DataSource).filter(database.DataSource.visibility=='public').all()
        db_conversation.datasources = []
        for db_datasouce in datasource_public:
            db_conversation.datasources.append(db_datasouce)
        
    else:
        if 'tools' in params_ex:
            # 이 부분이 먹히는지 테스트한다. many to many 는 먹힌다.
            db_conversation.tools = []
            for tool_id in params_ex['tools']:
                db_tool = db.query(database.Tool).filter(database.db_comment_endpoint).filter(database.Tool.tool_id == tool_id).first()
                if db_tool:
                    db_conversation.tools.append(db_tool)
        
        # Update agents to the conversation
        # Remove and update
        if 'agents' in params_ex:
            db_conversation.agents = []
            for agent_id in conversation.agents:
                db_agent = db.query(database.Agent).filter(database.db_comment_endpoint).filter(database.Agent.agent_id == agent_id).first()
                if db_agent:
                    db_conversation.agents.append(db_agent)
                    
        if 'datasources' in params_ex:
            db_conversation.datasources = []
            for datasource_id in conversation.datasources:
                db_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
                if db_datasource:
                    db_conversation.datasources.append(db_datasource)

    db.flush()
    db.refresh(db_conversation)
    return db_conversation

@router.delete("/conversations/{conversation_id}", response_model=Conversation, dependencies=[Depends(cookie)],tags=["Conversations"])
async def delete_conversation(conversation_id: str, db: Session = Depends(get_db) ,session_data: SessionData = Depends(verifier)):
    db_conversation = db.query(database.Conversation).filter(database.db_comment_endpoint).filter(database.Conversation.conversation_id == conversation_id).first()
    if db_conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    
    # Convert to Pydantic model before deleting
    conversation_data = Conversation.from_orm(db_conversation)

    db_conversation.agents = []
    db_conversation.datasources = []
    db.delete(db_conversation)
    db.flush()
    # db.refresh(db_conversation)
    
    # checkpoint 대화삭제
    conversation_instance = ConversationFactory.create_conversation(
        llm_api_provider=db_conversation.llm_api.llm_api_provider,
        llm_model=db_conversation.llm_model,
        llm_api_key=db_conversation.llm_api.llm_api_key,
        llm_api_url=db_conversation.llm_api.llm_api_url,
        temperature=db_conversation.temperature,
        max_tokens=db_conversation.max_tokens,
        sync_conn_pool=database.sync_conn_pool, 
        async_conn_pool=database.async_conn_pool
    )  
    await conversation_instance.clear(conversation_id)
    
    return conversation_data


