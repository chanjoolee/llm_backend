import asyncio
import traceback
from fastapi import FastAPI, Depends, HTTPException , status,logger as logger_fastapi
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from ai_core import CHROMA_DB_DEFAULT_PERSIST_DIR
from ai_core.conversation.message.base import DaisyMessageRole
from ai_core.data_source.base import create_data_source, create_data_source_tool
from ai_core.data_source.utils.utils import create_collection_name
from ai_core.llm_api_provider import LlmApiProvider
from app.endpoint.prompt import replace_variables
from app.endpoint.tools import construct_file_save_path, convert_db_tool_to_pydantic
from app.database import SessionLocal, get_db , KST
from app import database
from app.model import model_llm
from app.schema import schema_llm
from fastapi import APIRouter
from app.endpoint.login import cookie , SessionData , verifier
import os
from urllib.parse import quote
from ai_core.conversation.base import ConversationFactory , Conversation as Conversation_core# , DaisyChunkSyncIterator ,HISTORY_VARIABLE_NAME, QUESTION_VARIABLE_NAME

from ai_core.prompt.base import PromptComponent
from datetime import datetime  # Import datetime module
import pydash
from typing import Optional ,List,ForwardRef
from ai_core.checkpoint.mysql_saver import MySQLSaver
from langchain_core.messages import AIMessage, ToolMessage
import logging
from ai_core.llm.base import create_chat_model
from ai_core.tool.base import load_tool
import json
import app.config

logger_sql = logging.getLogger('sqlalchemy.engine')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

opensearch_hosts = os.getenv('opensearch_hosts')
opensearch_auth = (os.getenv('opensearch_username'), os.getenv('opensearch_password'))

# stream 같은것을 사용할 때 
def get_db_async():
    db = SessionLocal()
    try:
        yield db
    finally:
        pass
        # db.commit()
        # db.close()

# # Pydantic models
# class MessageCreate(BaseModel):
#     conversation_id: int
#     sender: str
#     message_text: str


# CRUD operations for Messages

def get_sum_token(collected_chunks, token_type):
    """
    Calculate the sum of a specific token type from the collected responses.

    :param collected_response_all: The list of response objects.
    :param token_type: The type of token to sum ('total_tokens', 'input_tokens', 'output_tokens').
    :return: The sum of the specified token type.
    """
    return pydash.chain(collected_chunks) \
        .map(lambda x: getattr(x.tokens_usage,token_type) if x.tokens_usage and hasattr(x.tokens_usage,token_type) else 0) \
        .sum() \
        .value()

def format_traceback(exc):
    """Convert the traceback into a more readable format."""
    tb_str = ''.join(traceback.format_exception(etype=type(exc), value=exc, tb=exc.__traceback__))
    return tb_str

@router.post("/messages", dependencies=[Depends(cookie)], tags=["Messages"])
async def create_message(
    message: schema_llm.MessageCreate, 
    db: Session = Depends(get_db_async),
    session_data: SessionData = Depends(verifier)
):
    logger.info("/messages 를 시작합니다.")
    db_conversation = db.query(model_llm.Conversation).filter(model_llm.db_comment_endpoint).filter(model_llm.Conversation.conversation_id==message.conversation_id).first()

    if db_conversation is None: 
        error_message = "Conversation does not exist"
        # raise HTTPException(
        #     status_code=200,
        #     detail={"error_code": 404, "message": error_message}
        # )
        raise HTTPException(status_code=404, detail=error_message)

    _llm_api_provider=pydash.clone_deep(db_conversation.llm_api.llm_api_provider)
    _llm_model=pydash.clone_deep(db_conversation.llm_model)
    _llm_api_key=pydash.clone_deep(db_conversation.llm_api.llm_api_key)
    _llm_api_url=pydash.clone_deep(db_conversation.llm_api.llm_api_url)
    _temperature=pydash.clone_deep(db_conversation.temperature)
    _max_tokens=pydash.clone_deep(db_conversation.max_tokens)

    logger.info(f"create_message: conversation instance ") 
    conversation_instance = ConversationFactory.create_conversation(
        llm_api_provider=_llm_api_provider,
        llm_model=_llm_model,
        llm_api_key=_llm_api_key,
        llm_api_url=_llm_api_url,
        temperature=_temperature,
        max_tokens=_max_tokens,
        sync_conn_pool=model_llm.sync_conn_pool, 
        async_conn_pool=model_llm.async_conn_pool
    )
    logger.warning(f"conversation instance info : {conversation_instance}") 

    # prompts 추가.  message 에 넣는다.
    if db_conversation.messages :
        # 메세지중 프롬프트를 찾는다.
        db_message_prompts = pydash.filter_(db_conversation.messages,{'input_path':'prompt'})
        if db_message_prompts: 
            add_to_prompt = []
            # db_prompt = db_conversation.prompts[0]
            for db_prompt in db_conversation.prompts:
                # variable
                variables ={}
                for db_variable in db_conversation.variables:
                    # setattr(variables, db_variable.variable_name, db_variable.variable_value)
                    variables[db_variable.variable_name] = db_variable.variable_value
                
                for db_message_prompt in db_message_prompts:
                    prompt = (db_message_prompt.message_type,db_message_prompt.message)
                    add_to_prompt.append(prompt)

                prompt_component = PromptComponent(
                    name=db_prompt.prompt_title,
                    description = db_prompt.prompt_desc,
                    messages=add_to_prompt,
                    input_values={}
                )
                logger.info(f"create_message: tool append {db_prompt.prompt_title}") 
                conversation_instance.add_prompt(prompt_component)     
    

    # tool 추가
    logger.info(f"tool 추가")
    for tool in db_conversation.tools:
        pydantic_tool = convert_db_tool_to_pydantic(tool)
        file_save_path = construct_file_save_path(pydantic_tool)
        logger.info(f"create_message: tool append {tool.name}") 
        conversation_instance.add_tool(tool.name,session_data.nickname,file_save_path)
        # logger.info(f"functionName for debug: {conversation_instance.tools[0].name}")

    # agent 추가
    for db_agent in db_conversation.agents:
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
        agent_prompt_messages_list = []
        agent_variables = set()
        for db_agent_variable in db_agent.variables:
            if db_agent_variable.variable_name not in agent_variables:
                agent_variables.add((db_agent_variable.variable_name,db_agent_variable.variable_value))
                
        agent_variables_dict = dict(agent_variables)    
        for db_agent_prompt in db_agent.prompts:
            for db_agent_prompt_message in db_agent_prompt.promptMessage:
                agent_prompt_message = replace_variables(db_agent_prompt_message.message ,agent_variables_dict)
                agent_prompt = (db_agent_prompt_message.message_type,agent_prompt_message)
                agent_prompt_messages_list.append(agent_prompt)
            
        
        # agent tool
        agent_tools = []
        for db_agent_tool in db_agent.tools:
            pydantic_tool_agent = convert_db_tool_to_pydantic(db_agent_tool)
            file_save_path_agent = construct_file_save_path(pydantic_tool_agent)
            logger.info(f"file_save_path_agent: {file_save_path_agent}")
            # agent_tool = load_tool("add_two_numbers", "egnarts", '../tool/add.py')
            agent_tool = load_tool(db_agent_tool.name, session_data.nickname, file_save_path_agent)
            # conversation_instance.add_tool(db_agent_tool.name,session_data.nickname,file_save_path_agent)
            logger.info(f"create_message: agent_tools append {db_agent_tool.name}") 
            agent_tools.append(agent_tool)
            # logger.info(f"functionName for debug: {conversation_instance.tools[0].name}")
        
        # agent datasources
        for db_agent_datasource in db_agent.datasources:
            agent_data_source = create_data_source(
                data_source_name=db_agent_datasource.name,
                created_by=db_agent_datasource.create_user_info.nickname,
                description=db_agent_datasource.description,
                data_source_type=db_agent_datasource.datasource_type,
                opensearch_hosts=opensearch_hosts,
                opensearch_auth=opensearch_auth
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
                    last_update_succeeded_at=db_agent_embedding.success_at
                )
            latest_collection_agent = agent_data_source.get_latest_collection()
            if latest_collection_agent is not None:
                agent_datasource_tool = create_data_source_tool(
                    name=db_agent_datasource.name,
                    username=db_agent_datasource.create_user_info.nickname,
                    datasource=agent_data_source
                )
                logger.info(f"create_message: agent_tools datasource append {db_agent_datasource.name}") 
                agent_tools.append(agent_datasource_tool)
        
        # add agent
        conversation_instance.add_agent(
            name=db_agent.name,
            description=db_agent.description,
            chat_model=chat_model,
            tools=agent_tools,
            prompt_messages=agent_prompt_messages_list
        )
        
    # datasource 추가
    for db_datasource in db_conversation.datasources:
        data_source = create_data_source(
            data_source_name=db_datasource.name,
            created_by=db_datasource.create_user_info.nickname,
            description=db_datasource.description,
            data_source_type=db_datasource.datasource_type,
            opensearch_hosts=opensearch_hosts,
            opensearch_auth=opensearch_auth
        )
        for db_embedding in db_datasource.embeddings:
            # # 3. 데이터 소스에 컬렉션 추가
            collection_name = create_collection_name(data_source.id, db_embedding.embedding_model)
            collection = data_source.add_collection(
                collection_name=collection_name,
                llm_api_provider=LlmApiProvider(db_embedding.llm_api.llm_api_provider),
                llm_api_key=db_embedding.llm_api.llm_api_key,
                llm_api_url=db_embedding.llm_api.llm_api_url,
                llm_embedding_model_name=db_embedding.embedding_model,
                last_update_succeeded_at=db_embedding.success_at
            )
        latest_collection = data_source.get_latest_collection()
        if latest_collection is not None:
            logger.info(f"create_message: add_datasource {db_datasource.name}")  
            conversation_instance.add_datasource(db_datasource.name,db_datasource.create_user_info.nickname,data_source)
            

    return_response = []
    db_message_human = model_llm.Message(
        conversation_id = message.conversation_id,
        message_type = 'human',
        message = message.message,
        input_path = 'conversation'
    )
    db.add(db_message_human)
    db.flush()
    db_conversation.last_conversation_time = datetime.now(KST)  # Set current datetime
    db_conversation.last_message_id = db_message_human.message_id
    return_response.append(db_message_human)
    messages = []
    

    processed_messages = []
    messages_all = []
    
    async def save_db():
            
        for message_ai in messages_all:
            if isinstance(message_ai.message, list):
                joined_texts = ''.join([msg['text'].encode('utf-8').decode('utf-8') for msg in message_ai.message if 'text' in msg])

            elif isinstance(message_ai.message, str):
                joined_texts = message_ai.message
            else:
                joined_texts = message_ai.message
            db_message_ai = model_llm.Message(
                conversation_id = db_conversation.conversation_id,
                message_type = message_ai.role.value,
                message = joined_texts,
                input_token = get_sum_token([message_ai],'input_tokens'),
                output_token =  get_sum_token([message_ai],'output_tokens'),
                total_token = get_sum_token([message_ai],'total_tokens'),
                input_path = 'conversation'
            )
            
            db.add(db_message_ai)
            db.flush()
            db_conversation.last_message_id = db_message_ai.message_id
            
            
        # used tokend
        db_conversation.used_tokens = (db_conversation.used_tokens or 0) + get_sum_token(messages_all, 'total_tokens')
        # input_token
        db_conversation.input_token = (db_conversation.input_token or 0) + get_sum_token(messages_all, 'input_tokens')
        # output token
        db_conversation.output_token = (db_conversation.output_token or 0) + get_sum_token(messages_all, 'output_tokens')
    async def message_generator():
        try: 
            logger.info("create_message: conversation invoke")
            await conversation_instance.create_agent(debug=True)
            logger.warning(f"conversation instance info : {conversation_instance}")
            async for message_ai in conversation_instance.invoke(db_conversation.conversation_id, message.message):
                if isinstance(message_ai.message, list):
                    joined_texts = ''.join([msg['text'].encode('utf-8').decode('utf-8') for msg in message_ai.message if 'text' in msg])

                elif isinstance(message_ai.message, str):
                    joined_texts = message_ai.message
                else:
                    joined_texts = message_ai.message

                try:
                    # Attempt to load `joined_texts` as JSON to check if it's already a JSON string
                    json.loads(joined_texts)
                    is_json = True
                except ValueError:
                    # If an error occurs, it means `joined_texts` is not a JSON string
                    is_json = False
                tokens_usage = 0
                if hasattr(message_ai,'tokens_usage') :
                    if message_ai.tokens_usage is not None:
                        tokens_usage = message_ai.tokens_usage.total_tokens
                processed_message = {
                    "conversation_id": db_conversation.conversation_id,
                    "message_type": message_ai.role.value,
                    "message": json.loads(joined_texts) if is_json else joined_texts ,
                    "sent_at": datetime.now(KST).strftime('%Y-%m-%dT%H:%M:%S'),
                    "input_path": "conversation",
                    "tokens_usage" : {
                        "total_tokens": get_sum_token([message_ai],'total_tokens'),
                        "input_tokens": get_sum_token([message_ai],'input_tokens'),
                        "output_tokens" : get_sum_token([message_ai],'output_tokens')
                    }
                }
                processed_messages.append(processed_message)
                messages_all.append(message_ai)
                # yield processed_message
                yield json.dumps(processed_message,ensure_ascii=False) + "\n"

            await save_db()
            db.commit()

        except Exception as e:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            if hasattr(e,'status_code') and e.status_code :
                status_code = e.status_code

            detail = f"Unhandled error {str(e)}"
            if hasattr(e, 'body') and e.body: 
                # Check if e.body is a dictionary
                if isinstance(e.body, dict):
                    detail = e.body
                    if 'body' in e.body:
                        detail = e.body['body']
                else:
                    # Handle the case where e.body is not a dictionary, if needed
                    detail = str(e.body)

            trace_str = traceback.format_exc()
            error_message = {
                "error_code" : status_code,                
                "detail": detail,
                "traceback": trace_str.split("\n")
                # "traceback": format_traceback(e)
            }
            logger.info(f"Error When message by invoke : {e}")
            logger.info(trace_str)
            
            yield json.dumps(error_message,ensure_ascii=False) + "\n"
            # yield json.dumps(e.body)
            # yield e

            db.rollback()
        
        finally:
            db.close()

    # return return_response
    # generator = message_generator(conversation_instance, message.conversation_id, message.message)
    return StreamingResponse(message_generator(), media_type="application/json")

@router.post(
    "/test_tool_call_db" , 
    description="프론트에서 참조X" ,
    tags=["Messages"],
    include_in_schema=False  # This hides the endpoint from Swagger
    )
async def test_tool_call_db(db: Session = Depends(get_db)):
    conversation_id = "1ed654f0-88d6-41f4-bc0b-d7eb760a2c8c"
    
    db_conversation = db.query(model_llm.Conversation).filter(model_llm.db_comment_endpoint).filter(model_llm.Conversation.conversation_id==conversation_id).first()
    
    _llm_api_provider=pydash.clone_deep(db_conversation.llm_api.llm_api_provider)
    _llm_model=pydash.clone_deep(db_conversation.llm_model)
    _llm_api_key=pydash.clone_deep(db_conversation.llm_api.llm_api_key)
    _llm_api_url=pydash.clone_deep(db_conversation.llm_api.llm_api_url)
    _temperature=pydash.clone_deep(db_conversation.temperature)
    _max_tokens=pydash.clone_deep(db_conversation.max_tokens)
    
    print(_llm_api_provider)
    print(_llm_model)
    print(_llm_api_key)
    print(_llm_api_url)
    print(_temperature)
    print(_max_tokens)


    smart_bee_conversation = ConversationFactory.create_conversation(
        llm_api_provider=_llm_api_provider,
        llm_model=_llm_model,
        llm_api_key=_llm_api_key,
        llm_api_url=_llm_api_url,
        temperature=_temperature,
        max_tokens=_max_tokens,
        sync_conn_pool=model_llm.sync_conn_pool, 
        async_conn_pool=model_llm.async_conn_pool
    )

    for tool in db_conversation.tools:
        pydantic_tool = convert_db_tool_to_pydantic(tool)
        file_save_path = construct_file_save_path(pydantic_tool)
        smart_bee_conversation.add_tool(tool.name,"chanjoo",file_save_path)
        
    
    
    # smart_bee_conversation.add_tool('add','chajoo',r'C:\project\sktelecom\gitlab\daisy_backend\ai_core\tests\integration_tests\tool\add.py')
    try:
        # conversation_id = "session2"
        await smart_bee_conversation.create_agent(debug=False)
        
        # smart_bee_conversation.clear("session2")
        # messages = smart_bee_conversation.invoke("session2", "3과 4를 더한 결과는?")
        messages = [] 
        await smart_bee_conversation.clear(conversation_id)
        async for m in smart_bee_conversation.invoke(conversation_id, "50과 12를 더한 결과는?"):
            print(m)
            messages.append(m)

        assert (messages[0].role == DaisyMessageRole.AI and messages[0].tool_call is not None
                and isinstance(messages[0].raw_message, AIMessage))
        assert (messages[1].role == DaisyMessageRole.AGENT and messages[1].tool_call is not None and messages[1].message == '7'
                and isinstance(messages[1].raw_message, ToolMessage))
        assert (messages[2].role == DaisyMessageRole.AI and messages[2].tool_call is None
                and isinstance(messages[2].raw_message, AIMessage))
    except Exception as e:
        # Handle exceptions here
        print(f"An error occurred: {e}")
        return {"result": "error", "details": str(e)}

    finally:
        # await smart_bee_conversation.close_connection_pools()
        
        
        return {
            "result": "success",
            "messages": messages
        }

@router.post(
    "/test_tool_call" 
    , description="프론트에서 참조X" 
    ,tags=["Messages"]
    ,include_in_schema=False  # This hides the endpoint from Swagger
)
async def test_tool_call(db: Session = Depends(get_db)):
    conversation_id = "session2"
    db_conversation = db.query(model_llm.Conversation).filter(model_llm.db_comment_endpoint).filter(model_llm.Conversation.conversation_id==conversation_id).first()

    smart_bee_conversation = ConversationFactory.create_conversation(
        llm_api_provider="smart_bee",
        llm_model='gpt-4o',
        llm_api_key='ba3954fe-9cbb-4599-966b-20b04b5d3441',
        llm_api_url='https://aihub-api.sktelecom.com/aihub/v1/sandbox',
        temperature=0.2,
        max_tokens=100
        , sync_conn_pool=model_llm.sync_conn_pool, 
        async_conn_pool=model_llm.async_conn_pool
    )

    smart_bee_conversation.add_tool('add','chajoo',r'C:\project\sktelecom\gitlab\daisy_backend\ai_core\tests\integration_tests\tool\add.py')
    try:
        await smart_bee_conversation.create_agent(debug=False)
        # smart_bee_conversation.clear("session2")
        # messages = smart_bee_conversation.invoke("session2", "3과 4를 더한 결과는?")
        messages = [] 
        await smart_bee_conversation.clear(conversation_id)
        async for m in smart_bee_conversation.invoke(conversation_id, "50 과 12 를 더한 결과는?"):
            print(m)
            messages.append(m)

        assert (messages[0].role == DaisyMessageRole.AI and messages[0].tool_call is not None
                and isinstance(messages[0].raw_message, AIMessage))
        assert (messages[1].role == DaisyMessageRole.AGENT and messages[1].tool_call is not None and messages[1].message == '7'
                and isinstance(messages[1].raw_message, ToolMessage))
        assert (messages[2].role == DaisyMessageRole.AI and messages[2].tool_call is None
                and isinstance(messages[2].raw_message, AIMessage))
    except Exception as e:
        # Handle exceptions here
        print(f"An error occurred: {e}")
        return {"result": "error", "details": str(e)}

    finally:
        # await smart_bee_conversation.close_connection_pools()
        
        
        return {
            "result": "success",
            "messages": messages
        }


@router.post("/messages_stream", dependencies=[Depends(cookie)], tags=["Messages"])
async def create_message_stream(
    message: schema_llm.MessageCreate, 
    db: Session = Depends(get_db_async),
    session_data: SessionData = Depends(verifier)
):
    db_conversation = db.query(model_llm.Conversation).filter(model_llm.db_comment_endpoint).filter(model_llm.Conversation.conversation_id==message.conversation_id).first()
    
    logger.info(f"create_message_stream: conversation instance ")
    logger.info(f"message: {message}")
    
    # Validation if db_conversation.llm_model is contained to conversation.llm_api.llm_model.
    # llm_api.llm_model is like "gpt-4o,gpt-3.5".
    # db_conversation.llm_model must be one of "gpt-4o,gpt-3.5"
    if db_conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    valid_models = db_conversation.llm_api.llm_model.split(",")
    if db_conversation.llm_model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid LLM model. Model '{db_conversation.llm_model}' is not in allowed models: {valid_models}"
        )
    
    
    conversation_instance = ConversationFactory.create_conversation(
        llm_api_provider=db_conversation.llm_api.llm_api_provider,
        llm_model=db_conversation.llm_model,
        llm_api_key=db_conversation.llm_api.llm_api_key,
        llm_api_url=db_conversation.llm_api.llm_api_url,
        temperature=db_conversation.temperature,
        max_tokens=db_conversation.max_tokens,
        sync_conn_pool=model_llm.sync_conn_pool, 
        async_conn_pool=model_llm.async_conn_pool
    )
    
    


    # prompts 추가.  message 에 넣는다.
    if db_conversation.messages :
        # 메세지중 프롬프트를 찾는다.
        db_message_prompts = pydash.filter_(db_conversation.messages,{'input_path':'prompt'})
        if db_message_prompts: 
            add_to_prompt = []
            # db_prompt = db_conversation.prompts[0]
            for db_prompt in db_conversation.prompts:
                # variable
                variables ={}
                for db_variable in db_conversation.variables:
                    # setattr(variables, db_variable.variable_name, db_variable.variable_value)
                    variables[db_variable.variable_name] = db_variable.variable_value
                
                for db_message_prompt in db_message_prompts:
                    prompt = (db_message_prompt.message_type,db_message_prompt.message)
                    add_to_prompt.append(prompt)

                prompt_component = PromptComponent(
                    name=db_prompt.prompt_title,
                    description = db_prompt.prompt_desc,
                    messages=add_to_prompt,
                    input_values={}
                )
                logger.info(f"create_message: tool append {db_prompt.prompt_title}") 
                conversation_instance.add_prompt(prompt_component)  


    # tool 추가
    logger.info(f"tool 추가")
    for tool in db_conversation.tools:
        pydantic_tool = convert_db_tool_to_pydantic(tool)
        file_save_path = construct_file_save_path(pydantic_tool)
        logger.info(f"create_message_stream: tool append {tool.name}") 
        conversation_instance.add_tool(tool.name,session_data.nickname,file_save_path)
        
        
    # agent 추가
    for db_agent in db_conversation.agents:
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
        agent_prompt_messages_list = []
        agent_variables = set()
        for db_agent_variable in db_agent.variables:
            if db_agent_variable.variable_name not in agent_variables:
                agent_variables.add((db_agent_variable.variable_name,db_agent_variable.variable_value))
                
        agent_variables_dict = dict(agent_variables)    
        for db_agent_prompt in db_agent.prompts:
            for db_agent_prompt_message in db_agent_prompt.promptMessage:
                agent_prompt_message = replace_variables(db_agent_prompt_message.message ,agent_variables_dict)
                agent_prompt = (db_agent_prompt_message.message_type,agent_prompt_message)
                agent_prompt_messages_list.append(agent_prompt)

        # agent tool
        agent_tools = []
        for db_agent_tool in db_agent.tools:
            pydantic_tool_agent = convert_db_tool_to_pydantic(db_agent_tool)
            file_save_path_agent = construct_file_save_path(pydantic_tool_agent)
            logger.info(f"file_save_path_agent: {file_save_path_agent}")
            # agent_tool = load_tool("add_two_numbers", "egnarts", '../tool/add.py')
            agent_tool = load_tool(db_agent_tool.name, session_data.nickname, file_save_path_agent)
            # conversation_instance.add_tool(db_agent_tool.name,session_data.nickname,file_save_path_agent)
            logger.info(f"create_message_stream: agent_tools append {db_agent_tool.name}") 
            agent_tools.append(agent_tool)
            # logger.info(f"functionName for debug: {conversation_instance.tools[0].name}")
        
        # agent  datasources
        for db_agent_datasource in db_agent.datasources:
            agent_data_source = create_data_source(
                data_source_name=db_agent_datasource.name,
                created_by=db_agent_datasource.create_user_info.nickname,
                description=db_agent_datasource.description,
                data_source_type=db_agent_datasource.datasource_type,
                opensearch_hosts=opensearch_hosts,
                opensearch_auth=opensearch_auth
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
                    last_update_succeeded_at=db_agent_embedding.success_at
                )
            latest_collection_agent = agent_data_source.get_latest_collection()
            if latest_collection_agent is not None:
                agent_datasource_tool = create_data_source_tool(
                    name=db_agent_datasource.name,
                    username=db_agent_datasource.create_user_info.nickname,
                    datasource=agent_data_source
                )
                logger.info(f"create_message_stream: agent_tools datasource append {db_agent_datasource.name}") 
                agent_tools.append(agent_datasource_tool)

        # add agent
        conversation_instance.add_agent(
            name=db_agent.name,
            description=db_agent.description,
            chat_model=chat_model,
            tools=agent_tools,
            prompt_messages=agent_prompt_messages_list
        )
        
    # datasource 추가
    for db_datasource in db_conversation.datasources:
        data_source = create_data_source(
            data_source_name=db_datasource.name,
            created_by=db_datasource.create_user_info.nickname,
            description=db_datasource.description,
            data_source_type=db_datasource.datasource_type,
            opensearch_hosts=opensearch_hosts,
            opensearch_auth=opensearch_auth
        )
        for db_embedding in db_datasource.embeddings:
            # # 3. 데이터 소스에 컬렉션 추가
            collection_name = create_collection_name(data_source.id, db_embedding.embedding_model)
            logger.info(f"add collection for message: {collection_name}  to {db_datasource.datasource_id}" )
            collection = data_source.add_collection(
                collection_name=collection_name,
                llm_api_provider=LlmApiProvider(db_embedding.llm_api.llm_api_provider),
                llm_api_key=db_embedding.llm_api.llm_api_key,
                llm_api_url=db_embedding.llm_api.llm_api_url,
                llm_embedding_model_name=db_embedding.embedding_model,
                last_update_succeeded_at=db_embedding.success_at
            )
        latest_collection = data_source.get_latest_collection()
        if latest_collection is not None:
            logger.info(f"create_message_stream: add_datasource {db_datasource.name}")  
            conversation_instance.add_datasource(db_datasource.name,db_datasource.create_user_info.nickname,data_source)
            
    
    db_message = model_llm.Message(
        conversation_id = message.conversation_id,
        message_type = 'human',
        message = message.message,
        input_path = 'conversation'
    )
    db.add(db_message)
    db.flush()  # Commit to get the message ID
    db.refresh(db_message)
    
    # 내부함수를 직접 쓴다.
    collected_response = []
    collected_response_all = []
    async def save_response_to_db():

        grouped_responses = pydash.group_by(collected_response_all, lambda x: x.id)
        added_messages = []
        for ai_message_id in grouped_responses:
            group_message = grouped_responses[ai_message_id]
            first_message = group_message[0]
            if first_message : 
                joined_messages = ''.join(
                    [
                        chunk.message.encode('utf-8').decode('utf-8') if isinstance(chunk.message, str)
                        else json.dumps(chunk.message, ensure_ascii=False) if isinstance(chunk.message, (list, dict))
                        else "" if chunk.message is None
                        else str(chunk.message)  # Fallback for unexpected types
                        for chunk in group_message
                    ]
                )
                
                db_message_ai = model_llm.Message(
                    conversation_id = db_conversation.conversation_id,
                    message_type = first_message.role.value,
                    # message_type = message_ai.raw_message.type,
                    message = joined_messages,
                    input_token = get_sum_token(group_message,'input_tokens'),
                    output_token =  get_sum_token(group_message,'output_tokens'),
                    total_token = get_sum_token(group_message,'total_tokens'),
                    input_path = 'conversation'
                )
                db.add(db_message_ai)
                added_messages.append(db_message_ai)
            
        # used tokend
        db_conversation.used_tokens = (db_conversation.used_tokens or 0) + get_sum_token(collected_response_all, 'total_tokens')
        # input token 
        db_conversation.input_token = (db_conversation.input_token or 0) + get_sum_token(collected_response_all, 'input_tokens')
        # output token
        db_conversation.output_token = (db_conversation.output_token or 0) + get_sum_token(collected_response_all, 'output_tokens')

        db.flush()  # Commit to get the message ID
        db_conversation.last_conversation_time = datetime.now(KST)  # Set current datetime
        if added_messages:
            db_conversation.last_message_id = added_messages[-1].message_id


    async def streaming_with_cleanup():
        try:
            # 스트리밍 방식
            await conversation_instance.create_agent(debug=True)
            logger.info("create_message_stream: conversation stream")
            logger.warning(f"conversation instance info : {conversation_instance}") 
            async for chunk in conversation_instance.stream(message.conversation_id, message.message):
                try: 
                    collected_response.append(chunk.message)
                    collected_response_all.append(chunk)
                    # logger.info(chunk.message)
                    # Inside your code where you process chunk.message
                    chunk_message = ""
                    if isinstance(chunk.message, str):  # Check if it's a string
                        chunk_message = chunk.message.encode('utf-8').decode('utf-8')
                    elif isinstance(chunk.message, (list, dict)):  # Handle list or dict types
                        chunk_message = json.dumps(chunk.message, ensure_ascii=False)  # Convert to JSON string
                    elif chunk.message is None:  # Handle None
                        chunk_message = ""
                    else:
                        # Handle other unexpected types gracefully
                        chunk_message = str(chunk.message)
                        
                    yield_chunk = {
                        "id" : chunk.id,
                        "message" : chunk_message,
                        "message_type" : chunk.role.value,
                        "sent_at": datetime.now(KST).strftime('%Y-%m-%dT%H:%M:%S'),
                        "tokens_usage" : {
                            "total_tokens": get_sum_token([chunk],'total_tokens'),
                            "input_tokens": get_sum_token([chunk],'input_tokens'),
                            "output_tokens" : get_sum_token([chunk],'output_tokens')
                        }
                    }                
                    yield json.dumps(yield_chunk,ensure_ascii=False)
                except ValueError as e:
                    logger.error(f"error when {message.conversation_id} ")
                    
                    yield_chunk = {
                        "id" : chunk.id,
                        "message" : f"{str(e)}",
                        "message_type" : chunk.role.value,
                        "sent_at": datetime.now(KST).strftime('%Y-%m-%dT%H:%M:%S')
                    }      
                    logger.error(f"error info {yield_chunk}")          
                    yield json.dumps(yield_chunk,ensure_ascii=False)
                    break
                    
                # yield chunk.message_aaa  # 에러유발
            await save_response_to_db()
            db.commit()
        except Exception as e:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            if hasattr(e,'status_code') and e.status_code :
                status_code = e.status_code

            detail = f"Unhandled error {str(e)}"
            if hasattr(e, 'body') and e.body: 
                # Check if e.body is a dictionary
                if isinstance(e.body, dict):
                    detail = e.body
                    if 'body' in e.body:
                        detail = e.body['body']
                else:
                    # Handle the case where e.body is not a dictionary, if needed
                    detail = str(e.body)

            trace_str = traceback.format_exc()
            error_message = {
                "error_code" : status_code,                
                "detail": detail,
                "traceback": trace_str.split("\n")
                # "traceback": format_traceback(e)
            }
            logger.info(f"Error When message by stream : {e}")
            logger.info(trace_str)
            
            yield json.dumps(error_message,ensure_ascii=False)
            # yield json.dumps(e.body)
            # yield e

            db.rollback()
        finally:
            db.close()
            

    return StreamingResponse(streaming_with_cleanup(), media_type="application/json")

@router.get("/messages/{message_id}", response_model=schema_llm.Message, dependencies=[Depends(cookie)], tags=["Messages"])
def read_message(message_id: int, db: Session = Depends(get_db)):
    db_message = db.query(model_llm.Message).filter(model_llm.db_comment_endpoint).filter(model_llm.Message.message_id == message_id).first()
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return db_message

@router.get(
    "/conversation_messages/{conversation_id}", 
    response_model=List[schema_llm.Message], 
    dependencies=[Depends(cookie)], 
    tags=["Messages"]
)
def read_message(
    conversation_id: str, 
    skip: Optional[int] = 0, 
    limit: Optional[int] = 10, 
    db: Session = Depends(get_db)
):
    db_message = db.query(model_llm.Message).filter(model_llm.db_comment_endpoint).filter(model_llm.Message.conversation_id == conversation_id).order_by(model_llm.Message.message_id.asc()).all()
    # if db_message is None:
    #     raise HTTPException(status_code=404, detail="Message not found")
    # Apply pagination
    if skip is not None  and limit is not None :
        query = query.offset(skip).limit(limit)

    return db_message

@router.get("/messages", response_model=List[schema_llm.Message], dependencies=[Depends(cookie)], tags=["Messages"])
def read_messages(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    messages = db.query(model_llm.Message).filter(model_llm.db_comment_endpoint).offset(skip).limit(limit).all()
    return messages

@router.put("/messages/{message_id}", response_model=schema_llm.Message, dependencies=[Depends(cookie)], tags=["Messages"])
def update_message(message_id: int, message: schema_llm.MessageCreate, db: Session = Depends(get_db)):
    db_message = db.query(model_llm.Message).filter(model_llm.db_comment_endpoint).filter(model_llm.Message.message_id == message_id).first()
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    db_message.conversation_id = message.conversation_id
    db_message.sender = message.sender
    db_message.message_text = message.message_text
    db.refresh(db_message)
    return db_message

@router.delete("/messages/{message_id}", response_model=schema_llm.Message, dependencies=[Depends(cookie)], tags=["Messages"])
def delete_message(message_id: int, db: Session = Depends(get_db)):
    db_message = db.query(model_llm.Message).filter(model_llm.db_comment_endpoint).filter(model_llm.Message.message_id == message_id).first()
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    db.delete(db_message)
    return db_message


# 이거왜이래