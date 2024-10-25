import asyncio
import json
from fastapi import Body, FastAPI, Depends, HTTPException, Header, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from ai_core import CHROMA_DB_DEFAULT_PERSIST_DIR
from ai_core.data_source.base import create_data_source, create_data_source_tool
from ai_core.data_source.utils.utils import create_collection_name
from ai_core.llm.base import create_chat_model
from ai_core.llm_api_provider import LlmApiProvider
from ai_core.tool.base import load_tool
from app.endpoint.tools import construct_file_save_path, convert_db_tool_to_pydantic
from app.models import Message , MessageCreate
from app.database import SessionLocal, get_db , KST
from app import models, database
from fastapi import APIRouter
from app.endpoint.login import cookie , SessionData , verifier
import os
from urllib.parse import quote
from ai_core.conversation.base import ConversationFactory # , DaisyChunkSyncIterator ,HISTORY_VARIABLE_NAME, QUESTION_VARIABLE_NAME
from datetime import datetime  # Import datetime module
import pydash
from typing import Optional ,List,ForwardRef
from app.utils import utils
import logging

logger = logging.getLogger('sqlalchemy.engine')

router = APIRouter()


# stream 같은것을 사용할 때 
def get_db_async():
    db = SessionLocal()
    try:
        yield db
    finally:
        pass
        # db.commit()
        # db.close()




@router.post("/ask", 
    # dependencies=[Depends(cookie)], 
    response_model=List[Message],
    tags=["System Message"],
    description="""
    시스템 대화를 위한 주소입니다.
    """
)
async def ask_system_message(
    message: models.SystemMessageCreate ,
    db: Session = Depends(get_db)
):
    # check user
    db_owner = db.query(database.User).filter(database.User.user_roll=='SYSTEM').first()
    if db_owner is None:
        raise HTTPException(status_code=404, detail="User of 'SYSTEM' is not found")
    
    # check llm api
    db_llm_api = db.query(database.LlmApi).filter(
        database.LlmApi.llm_api_provider=='smart_bee'
        and database.LlmApi.llm_model=='gpt-4o'
        and database.LlmApi.llm_api_type=='public'
    ).first()
    if db_llm_api is None:
        raise HTTPException(status_code=404, detail="llm api with smart_bee is not found")
    
    # create conversation
    conversation_id = str(utils.generate_conversation_id())
    db_conversation = database.Conversation(
        user_id = db_owner.user_id ,
        conversation_id = conversation_id
    )
    db_conversation.conversation_title = "conversation_system_" + datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    db_conversation.conversation_type = 'public'
    db_conversation.llm_api_id = db_llm_api.llm_api_id
    db_conversation.llm_model = db_llm_api.llm_model.split('/')[-1]
    db_conversation.temperature = 0
    db_conversation.max_tokens = 4096
    db_conversation.component_configuration = 'component'  # agent?

    db.add(db_conversation)
    db.flush()
    db.refresh(db_conversation)

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


    # # tool 추가 공개된 tool 을 넣는다.
    # logger.info(f"tool 추가")
    # tools_public = db.query(database.Tool).filter(database.Tool.visibility=='public').all()

    

    # for db_tool in tools_public:
    #     pydantic_tool = convert_db_tool_to_pydantic(db_tool)
    #     file_save_path = construct_file_save_path(pydantic_tool)
    #     try: 
    #         db_conversation.tools.append(db_tool)
    #         conversation_instance.add_tool(db_tool.name,db_owner.nickname,file_save_path)
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code=500, 
    #             detail=f"""
    #             tool id : {db_tool.tool_id}
    #             tool name : {db_tool.name}
    #             file path : {file_save_path}
    #             error message: toolName {e}
    #             """
    #         )
    
    # agent 추가
    agent_public = db.query(database.Agent).filter(database.Agent.visibility=='public').all()
    for db_agent in agent_public:
        # chat_model
        chat_model = create_chat_model(
            llm_api_provider=db_agent.llm_api.llm_api_provider,
            llm_model=db_agent.llm_model,
            llm_api_key=db_agent.llm_api.llm_api_key,
            llm_api_url=db_agent.llm_api.llm_api_url,
            temperature=db_conversation.temperature,
            max_tokens=db_conversation.max_tokens
        )

        # prompt. please complete below area

        # tool
        agent_tools = []
        for db_agent_tool in db_agent.tools:
            
            pydantic_tool_agent = convert_db_tool_to_pydantic(db_agent_tool)
            file_save_path_agent = construct_file_save_path(pydantic_tool_agent)
            logger.info(f"file_save_path_agent: {file_save_path_agent}")
            # agent_tool = load_tool("add_two_numbers", "egnarts", '../tool/add.py')
            agent_tool = load_tool(db_agent_tool.name, db_agent_tool.create_user_info.nickname, file_save_path_agent)
            # conversation_instance.add_tool(db_agent_tool.name,session_data.nickname,file_save_path_agent)
            agent_tools.append(agent_tool)
            # logger.info(f"functionName for debug: {conversation_instance.tools[0].name}")
        
        # datasources
        for db_agent_datasource in db_agent.datasources:
            agent_data_source = create_data_source(
                data_source_name=db_agent_datasource.name,
                created_by=db_agent_datasource.create_user_info.nickname,
                description=db_agent_datasource.description,
                data_source_type=db_agent_datasource.datasource_type
            )
            for db_agent_embedding in db_agent_datasource.embeddings:
                # 데이터 소스에 컬렉션 추가
                agent_collection_name = create_collection_name(agent_data_source.id, db_agent_embedding.embedding_model)
                agent_collection = agent_data_source.add_collection(
                    collection_name=agent_collection_name,
                    llm_api_provider=LlmApiProvider(db_agent_embedding.llm_api.llm_api_provider),
                    llm_api_key=db_agent_embedding.llm_api.llm_api_key,
                    llm_api_url=db_agent_embedding.llm_api.llm_api_url,
                    llm_embedding_model_name=db_agent_embedding.embedding_model,
                    persist_directory=CHROMA_DB_DEFAULT_PERSIST_DIR
                    ,last_update_succeeded_at = db_agent_embedding.success_at
                )
            latest_collection_agent = agent_data_source.get_latest_collection()
            if latest_collection_agent is not None:    
                agent_datasource_tool = create_data_source_tool(
                    name=db_agent_datasource.name,
                    username=db_agent_datasource.create_user_info.nickname,
                    datasource=agent_data_source
                )
                agent_tools.append(agent_datasource_tool)
        # add agent
        conversation_instance.add_agent(
            name=db_agent.name,
            description=db_agent.description,
            chat_model=chat_model,
            tools=agent_tools
        )
       
    # 내부함수를 직접 쓴다.
    return_response = []
    db_message_human = database.Message(
        conversation_id = conversation_id,
        message_type = 'human',
        message = message.message,
        input_path = 'system'
    )
    db.add(db_message_human)
    db.flush()
    db_conversation.last_conversation_time = datetime.now(KST)  # Set current datetime
    db_conversation.last_message_id = db_message_human.message_id
    return_response.append(db_message_human)
    messages = []

        
    await conversation_instance.create_agent(debug=True)
    # conversation_instance.clear(message.conversation_id)
    async for message_ai in conversation_instance.invoke(conversation_id, message.message):
        messages.append(message_ai)


    for message_ai in messages:
        if isinstance(message_ai.message, list):
            # If message_ai.message is a list, join the text attributes
            joined_texts = ''.join([msg['text'].encode('utf-8').decode('utf-8') for msg in message_ai.message if 'text' in msg])
        elif isinstance(message_ai.message, str):
            # If message_ai.message is already a string, use it directly
            joined_texts = message_ai.message
        else:
            # Handle other unexpected types if necessary
            joined_texts = message_ai.message

        db_message_ai = database.Message(
            conversation_id = db_conversation.conversation_id,
            message_type = message_ai.role.value,
            # message_type = message_ai.raw_message.type,
            message = joined_texts,
            input_path = 'system'
        )
        db.add(db_message_ai)
        return_response.append(db_message_ai)
    # used tokend
    total_tokens_usage = pydash.chain(messages) \
    .map_(lambda x: x.tokens_usage.total_tokens if x.tokens_usage and x.tokens_usage.total_tokens is not None else 0) \
    .sum_() \
    .value()
    used_tokens = db_conversation.used_tokens
    if used_tokens is None:
        used_tokens = 0
    db_conversation.used_tokens = used_tokens + total_tokens_usage

    if len(return_response) > 1 :
        db.flush()
        db_conversation.last_conversation_time = datetime.now(KST)  # Set current datetime
        db_conversation.last_message_id = return_response[-1].message_id
        
    return return_response


@router.post("/alert/{sender}", 
    # dependencies=[Depends(cookie)], 
    response_model=List[Message],
    tags=["System Message"],
    description="""
    시스템 대화를 위한 주소입니다.
    알람주체설정
    """
)
async def alert_system_message(
    sender: str ,
    request: Request,
    content_type: str = Header(...),
    db: Session = Depends(get_db)
):
    # Handle both 'text/plain' and 'application/json' content types
    if content_type == 'text/plain':
        message = await request.body()
        message = message.decode("utf-8")
        
    elif content_type == 'application/json':
        message = await request.json()
        message = json.dumps(message,ensure_ascii=False)
    else:
        return PlainTextResponse("Invalid content type", status_code=400)
    
    
    # # Get the raw text data from the request body
    # request_body = await request.body()
    # message = request_body.decode('utf-8')  # Decoding as UTF-8
    
    # check user
    db_owner = db.query(database.User).filter(database.User.user_roll=='SYSTEM').first()
    if db_owner is None:
        raise HTTPException(status_code=404, detail="User of 'SYSTEM' is not found")
    
    db_sender = db.query(database.User).filter(database.User.nickname == sender).first()
    if db_sender is None:
        # creat user
        db_sender = database.User(
            user_id=sender, 
            password=db_owner.password,  
            nickname=sender,
            user_roll='ALERT'
        )
        db.add(db_sender)
        
    
    # check llm api
    db_llm_api = db.query(database.LlmApi).filter(
        database.LlmApi.llm_api_provider=='smart_bee'
        and database.LlmApi.llm_model=='gpt-4o'
        and database.LlmApi.llm_api_type=='public'
    ).first()
    if db_llm_api is None:
        raise HTTPException(status_code=404, detail="llm api with smart_bee is not found")
    
    # create conversation
    conversation_id = str(utils.generate_conversation_id())
    db_conversation = database.Conversation(
        user_id = db_sender.user_id ,
        conversation_id = conversation_id
    )
    db_conversation.conversation_title = "conversation_system_" + datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    db_conversation.conversation_type = 'public'
    db_conversation.llm_api_id = db_llm_api.llm_api_id
    db_conversation.llm_model = db_llm_api.llm_model.split('/')[-1]
    db_conversation.temperature = 0
    db_conversation.max_tokens = 4096
    db_conversation.component_configuration = 'component'  # agent?

    db.add(db_conversation)
    db.flush()
    db.refresh(db_conversation)

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


    # # tool 추가 공개된 tool 을 넣는다.
    # logger.info(f"tool 추가")
    # tools_public = db.query(database.Tool).filter(database.Tool.visibility=='public').all()

    

    # for db_tool in tools_public:
    #     pydantic_tool = convert_db_tool_to_pydantic(db_tool)
    #     file_save_path = construct_file_save_path(pydantic_tool)
    #     try: 
    #         db_conversation.tools.append(db_tool)
    #         conversation_instance.add_tool(db_tool.name,db_owner.nickname,file_save_path)
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code=500, 
    #             detail=f"""
    #             tool id : {db_tool.tool_id}
    #             tool name : {db_tool.name}
    #             file path : {file_save_path}
    #             error message: toolName {e}
    #             """
    #         )
    
    # agent 추가
    agent_public = db.query(database.Agent).filter(database.Agent.visibility=='public').all()
    for db_agent in agent_public:
        # chat_model
        chat_model = create_chat_model(
            llm_api_provider=db_agent.llm_api.llm_api_provider,
            llm_model=db_agent.llm_model,
            llm_api_key=db_agent.llm_api.llm_api_key,
            llm_api_url=db_agent.llm_api.llm_api_url,
            temperature=db_conversation.temperature,
            max_tokens=db_conversation.max_tokens
        )

        # prompt. please complete below area

        # tool
        agent_tools = []
        for db_agent_tool in db_agent.tools:
            
            pydantic_tool_agent = convert_db_tool_to_pydantic(db_agent_tool)
            file_save_path_agent = construct_file_save_path(pydantic_tool_agent)
            logger.info(f"file_save_path_agent: {file_save_path_agent}")
            # agent_tool = load_tool("add_two_numbers", "egnarts", '../tool/add.py')
            agent_tool = load_tool(db_agent_tool.name, db_agent_tool.create_user_info.nickname, file_save_path_agent)
            # conversation_instance.add_tool(db_agent_tool.name,session_data.nickname,file_save_path_agent)
            agent_tools.append(agent_tool)
            # logger.info(f"functionName for debug: {conversation_instance.tools[0].name}")
        
        # datasources
        for db_agent_datasource in db_agent.datasources:
            agent_data_source = create_data_source(
                data_source_name=db_agent_datasource.name,
                created_by=db_agent_datasource.create_user_info.nickname,
                description=db_agent_datasource.description,
                data_source_type=db_agent_datasource.datasource_type
            )
            for db_agent_embedding in db_agent_datasource.embeddings:
                # 데이터 소스에 컬렉션 추가
                agent_collection_name = create_collection_name(agent_data_source.id, db_agent_embedding.embedding_model)
                agent_collection = agent_data_source.add_collection(
                    collection_name=agent_collection_name,
                    llm_api_provider=LlmApiProvider(db_agent_embedding.llm_api.llm_api_provider),
                    llm_api_key=db_agent_embedding.llm_api.llm_api_key,
                    llm_api_url=db_agent_embedding.llm_api.llm_api_url,
                    llm_embedding_model_name=db_agent_embedding.embedding_model,
                    persist_directory=CHROMA_DB_DEFAULT_PERSIST_DIR
                    ,last_update_succeeded_at = db_agent_embedding.success_at
                )
            latest_collection_agent = agent_data_source.get_latest_collection()
            if latest_collection_agent is not None:    
                agent_datasource_tool = create_data_source_tool(
                    name=db_agent_datasource.name,
                    username=db_agent_datasource.create_user_info.nickname,
                    datasource=agent_data_source
                )
                agent_tools.append(agent_datasource_tool)
        # add agent
        conversation_instance.add_agent(
            name=db_agent.name,
            description=db_agent.description,
            chat_model=chat_model,
            tools=agent_tools
        )
       
    # 내부함수를 직접 쓴다.
    return_response = []

        
    db_message_human = database.Message(
        conversation_id = conversation_id,
        message_type = 'human',
        message = message,
        input_path = 'system'
    )
    db.add(db_message_human)
    db.flush()
    db_conversation.last_conversation_time = datetime.now(KST)  # Set current datetime
    db_conversation.last_message_id = db_message_human.message_id
    return_response.append(db_message_human)
    messages = []

        
    await conversation_instance.create_agent(debug=True)
    # conversation_instance.clear(message.conversation_id)
    async for message_ai in conversation_instance.invoke(conversation_id, message):
        messages.append(message_ai)


    for message_ai in messages:
        if isinstance(message_ai.message, list):
            # If message_ai.message is a list, join the text attributes
            joined_texts = "".join([msg['text'] for msg in message_ai.message if 'text' in msg])
        elif isinstance(message_ai.message, str):
            # If message_ai.message is already a string, use it directly
            joined_texts = message_ai.message
        else:
            # Handle other unexpected types if necessary
            joined_texts = message_ai.message

        db_message_ai = database.Message(
            conversation_id = db_conversation.conversation_id,
            message_type = message_ai.role.value,
            # message_type = message_ai.raw_message.type,
            message = joined_texts,
            input_path = 'system'
        )
        db.add(db_message_ai)
        return_response.append(db_message_ai)


    if len(return_response) > 1 :
        db.flush()
        db_conversation.last_conversation_time = datetime.now(KST)  # Set current datetime
        db_conversation.last_message_id = return_response[-1].message_id
        
    return return_response


