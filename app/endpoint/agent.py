import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import exists, insert, or_, select, text
from sqlalchemy.orm import Session
from app import database
from app.model import model_llm
from app.schema import schema_llm
from app.utils import utils
from app.endpoint.login import cookie , SessionData , verifier
from app.database import SessionLocal, get_db

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()


@router.post(
    "/create_agent",
    response_model=schema_llm.Agent,
    dependencies=[Depends(cookie)],
    tags=["Agents"],
    description="""<pre>
    Creates a new agent.
    <h3>Request Body:</h3>
        - name (str): The name of the agent.
        - description (Optional[str]): A description of the agent.
        - visibility (str): The visibility of the agent (private or public).
        - llm_api_id (Optional[int]): The ID of the LLM API.
        - llm_model (str): The LLM model.
        - prompts (Optional[List[int]]): List of prompt IDs associated with the agent.
        - tools (Optional[List[int]]): List of tool IDs associated with the agent.
        - tags (Optional[List[int]]): List of tag IDs associated with the agent.
        - sub_agents (Optional[List[int]]): List of child agent IDs.
        - datasources (List[int]): A list of agent IDs to associate with the agent.
    </pre>
    """
)
def create_agent(
    agent: schema_llm.AgentCreate,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    db_agent = model_llm.Agent(
        create_user=session_data.user_id
    )
    # Handle prompts, tools, and tags associations
    update_data = agent.dict(exclude_unset=True)
    for key, value in update_data.items():
        if key not in ["prompts", "tools", "tags", "sub_agents","datasources"]:
            setattr(db_agent, key, value)

    db.add(db_agent)
    db.flush()
    db.refresh(db_agent)


    # Handle prompts associations
    if 'prompts' in update_data:
        for i, prompt in enumerate(agent.prompts):
            db_prompt = db.query(model_llm.Prompt).filter(model_llm.Prompt.prompt_id == prompt.prompt_id).first()

            if db_prompt:
                stmt = insert(model_llm.agent_prompts).values(
                    agent_id=db_agent.agent_id,
                    prompt_id=db_prompt.prompt_id
                    # ,sort_order=i+1
                )
                db.execute(stmt)

                for db_prompt_message in db_prompt.promptMessage:
                    message = db_prompt_message.message
                    for variable in prompt.variables:
                        db_variable = model_llm.AgentVariable(
                            agent_id=db_agent.agent_id,
                            variable_name=variable.variable_name,
                            variable_value=variable.value
                        )
                        db.add(db_variable)

                    # 이 부분은 어떻게 처리를 하나 ==> conversation 을 만들때 처리하는 것이 좋을 듯. 어떻게? ...
                    # new_message = model_llm.Message(
                    #     conversation_id=db_agent.agent_id,
                    #     message_type=db_prompt_message.message_type,
                    #     message=message,
                    #     input_path='prompt'
                    # )
                    # db.add(new_message)


    # Handle tools
    if 'tools' in update_data:
        for tool_id in update_data['tools']:
            db_tool = db.query(model_llm.Tool).filter(model_llm.db_comment_endpoint).filter(model_llm.Tool.tool_id == tool_id).first()
            if db_tool:
                db_agent.tools.append(db_tool)

    # Handle tags
    if 'tags' in update_data:
        for tag_id in agent.tags:
            db_tag = db.query(model_llm.Tag).filter(model_llm.db_comment_endpoint).filter(model_llm.Tag.tag_id == tag_id).first()
            if db_tag:
                db_agent.tags.append(db_tag)

    # Handle child agents associations
    if 'sub_agents' in update_data:
        for child_agent_id in agent.sub_agents:
            db_child_agent = db.query(model_llm.Agent).filter(model_llm.Agent.agent_id == child_agent_id).first()
            if db_child_agent:
                db_child_agent.parent_agent_id = db_agent.agent_id
                db.add(db_child_agent)
    if 'datasources' in update_data:
        for datasource_id in agent.datasources:
            db_datasource = db.query(model_llm.DataSource).filter(model_llm.db_comment_endpoint).filter(model_llm.DataSource.datasource_id == datasource_id).first()
            if db_datasource:
                db_agent.datasources.append(db_datasource)
    db.flush()
    db.refresh(db_agent)

    return db_agent

@router.get(
    "/get_agent/{agent_id}",
    response_model=schema_llm.Agent,
    dependencies=[Depends(cookie)],
    tags=["Agents"],
    description="""<pre>
    Get a specific agent by ID.
    </pre>
    """
)
def get_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(model_llm.Agent).filter(model_llm.db_comment_endpoint)
    query = query.filter(model_llm.Agent.agent_id == agent_id)

    db_agent = query.first()
    if db_agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return db_agent


@router.put(
    "/update_agent/{agent_id}",
    response_model=schema_llm.Agent,
    dependencies=[Depends(cookie)],
    tags=["Agents"],
    description="""<pre>
    Updates an existing agent by its ID.
    <h3>Request Body:</h:</h3>
        - name (Optional[str]): The name of the agent.
        - description (Optional[str]): A description of the agent.
        - visibility (Optional[str]): The visibility of the agent (private or public).
        - llm_api_id (Optional[int]): The ID of the LLM API.
        - llm_model (Optional[str]): The LLM model.
        - prompts (Optional[List[int]]): List of prompt IDs associated with the agent.
        - tools (Optional[List[int]]): List of tool IDs associated with the agent.
        - tags (Optional[List[int]]): List of tag IDs associated with the agent.
        - sub_agents (Optional[List[int]]): List of child agent IDs.
        - datasources (List[int]): A list of agent IDs to associate with the agent.
    </pre>
    """
)
def update_agent(
    agent_id: int,
    agent: schema_llm.AgentUpdate,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(model_llm.Agent).filter(model_llm.db_comment_endpoint)
    query = query.filter(model_llm.Agent.agent_id == agent_id)
    db_agent = query.first()

    if db_agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    

    update_data = agent.dict(exclude_unset=True)
    for key, value in update_data.items():
        if key not in ["prompts", "tools", "tags", "sub_agents","datasources"]:
            setattr(db_agent, key, value)
    
    # Handle prompts associations
    if 'prompts' in update_data:
        db_agent.prompts = []
        db_agent.variables = []
        db.query(model_llm.AgentVariable).filter(model_llm.AgentVariable.agent_id == agent_id).delete(synchronize_session='fetch')
        for i, prompt in enumerate(agent.prompts):
            db_prompt = db.query(model_llm.Prompt).filter(model_llm.Prompt.prompt_id == prompt.prompt_id).first()

            if db_prompt:
                # stmt = insert(model_llm.agent_prompts).values(
                #     agent_id=db_agent.agent_id,
                #     prompt_id=db_prompt.prompt_id
                # )
                # db.execute(stmt)
                db_agent.prompts.append(db_prompt)

                # for db_prompt_message in db_prompt.promptMessage:
                #     message = db_prompt_message.message
                    
                for variable in prompt.variables:
                    db_variable = model_llm.AgentVariable(
                        agent_id=db_agent.agent_id,
                        variable_name=variable.variable_name,
                        variable_value=variable.value
                    )
                    db.add(db_variable)

                    # 이 부분은 어떻게 처리를 하나 ==> conversation 을 만들때 처리하는 것이 좋을 듯. 어떻게? ...
                    # new_message = model_llm.Message(
                    #     conversation_id=db_agent.agent_id,
                    #     message_type=db_prompt_message.message_type,
                    #     message=message,
                    #     input_path='prompt'
                    # )
                    # db.add(new_message)

    # Handle tools
    if 'tools' in update_data:
        db_agent.tools = []
        for tool_id in update_data['tools']:
            db_tool = db.query(model_llm.Tool).filter(model_llm.db_comment_endpoint).filter(model_llm.Tool.tool_id == tool_id).first()
            if db_tool:
                db_agent.tools.append(db_tool)

    # Handle tags
    if 'tags' in update_data:
        db_agent.tags = []
        for tag_id in agent.tags:
            db_tag = db.query(model_llm.Tag).filter(model_llm.db_comment_endpoint).filter(model_llm.Tag.tag_id == tag_id).first()
            if db_tag:
                db_agent.tags.append(db_tag)

    # Handle child agents associations
    if 'sub_agents' in update_data:
        current_child_agents = {child.agent_id for child in db_agent.sub_agents}
        new_child_agents = set(update_data['sub_agents'])

        # Remove relationships that are no longer valid
        agents_to_remove = current_child_agents - new_child_agents
        for child_agent_id in agents_to_remove:
            db_child_agent = db.query(model_llm.Agent).filter(model_llm.Agent.agent_id == child_agent_id).first()
            if db_child_agent:
                db_child_agent.parent_agent_id = None

        # Add new relationships
        agents_to_add = new_child_agents - current_child_agents
        for child_agent_id in agents_to_add:
            db_child_agent = db.query(model_llm.Agent).filter(model_llm.Agent.agent_id == child_agent_id).first()
            if db_child_agent:
                db_child_agent.parent_agent_id = db_agent.agent_id

    if 'datasources' in update_data:
        db_agent.datasources = []
        for datasource_id in agent.datasources:
            db_datasource = db.query(model_llm.DataSource).filter(model_llm.db_comment_endpoint).filter(model_llm.DataSource.datasource_id == datasource_id).first()
            if db_datasource:
                db_agent.datasources.append(db_datasource)
    db.flush()
    db.refresh(db_agent)
    
    return db_agent

@router.delete(
    "/delete_agent/{agent_id}",
    response_model=schema_llm.Agent,
    dependencies=[Depends(cookie)],
    tags=["Agents"],
    description="""<pre>
    Delete a specific agent by ID.
    </pre>
    """
)
def delete_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(cookie)
):
    query = db.query(model_llm.Agent).filter(model_llm.db_comment_endpoint)
    query = query.filter(model_llm.Agent.agent_id == agent_id)
    db_agent = query.first()

    if db_agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    
    deleted_data = schema_llm.Agent.from_orm(db_agent)   
    db_agent.prompts = []
    db_agent.tools = []
    db_agent.tags = []
    db_agent.conversations = [] 
    db.delete(db_agent)
    
    return deleted_data

"""
개인만이 조회하는 경우가 있고 전체를 조회하는 경우가 있음.
공통적으로 사용해야 하는 Function 이 필요함.
"""
def search_agent_basic(db, search_exclude):
    query = db.query(model_llm.Agent).filter(model_llm.db_comment_endpoint)
    if 'search_words' in search_exclude and search_exclude['search_words']:
        search_filter_word = []
        if 'title' in search_exclude['search_range_list']:
            filter_title = (model_llm.Agent.name.like(f'%{search_exclude["search_words"]}%'))
            search_filter_word.append(filter_title)
        if 'description' in search_exclude['search_range_list']:
            filter_description = (model_llm.Agent.description.like(f'%{search_exclude["search_words"]}%'))
            search_filter_word.append(filter_description)
        if len(search_filter_word) > 0 :
            query = query.filter(or_(*search_filter_word))
            
    if 'visibility_list' in search_exclude and search_exclude['visibility_list']:
        query = query.filter(model_llm.Agent.visibility.in_(search_exclude['visibility_list']))

    if 'tag_list' in search_exclude and search_exclude['tag_list']:
        query = query.join(model_llm.agent_tags).filter(model_llm.agent_tags.c.tag_id.in_(search_exclude['tag_list']))
    
    if 'user_list' in search_exclude and len(search_exclude['user_list']) > 0:
        query = query.filter(model_llm.Agent.create_user .in_(search_exclude['user_list']))

    if 'llm_api_list' in search_exclude and search_exclude['llm_api_list']:
        query = query.filter(model_llm.Agent.llm_api_id.in_(search_exclude['llm_api_list']))

    if 'llm_model_list' in search_exclude and search_exclude['llm_model_list']:
        query = query.filter(model_llm.Agent.llm_model.in_(search_exclude['llm_model_list']))
    
    if 'prompt_list' in search_exclude and len(search_exclude['prompt_list']) > 0:
        subquery = (
            select(model_llm.agent_prompts.c.agent_id)
            .join(model_llm.Prompt, model_llm.Prompt.prompt_id == model_llm.agent_prompts.c.prompt_id)
            .filter(model_llm.Prompt.prompt_id.in_(search_exclude['prompt_list']))
        )
        
        query = query.filter(
            exists(subquery.where(model_llm.Agent.agent_id == model_llm.agent_prompts.c.agent_id))
        )
     
    if 'tool_list' in search_exclude and len(search_exclude['tool_list']) > 0:
        subquery = (
            select(model_llm.agent_tools.c.tool_id)
            .join(model_llm.Tool, model_llm.Tool.tool_id == model_llm.agent_tools.c.tool_id)
            .filter(model_llm.Tool.tool_id.in_(search_exclude['tool_list']))
        )
        
        query = query.filter(
            exists(subquery.where(model_llm.Agent.agent_id == model_llm.agent_tools.c.agent_id))
        )
               
    if 'datasource_list' in search_exclude and len(search_exclude['datasource_list']) > 0:
        subquery = (
            select(model_llm.agent_datasource.c.agent_id)
            .join(model_llm.DataSource, model_llm.DataSource.datasource_id == model_llm.agent_datasource.c.datasource_id)
            .filter(model_llm.DataSource.datasource_id.in_(search_exclude['datasource_list']))
        )
        
        query = query.filter(
            exists(subquery.where(model_llm.Agent.agent_id == model_llm.agent_datasource.c.agent_id))
        )
        
    filter_component = []  
    if 'component_list' in search_exclude and len(search_exclude['component_list'])> 0 : 
        for component in search_exclude['component_list']:
            if component['type'] == 'prompt':
                subquery = (
                    select(model_llm.agent_prompts.c.agent_id)
                    .join(model_llm.Prompt, model_llm.Prompt.prompt_id == model_llm.agent_prompts.c.prompt_id)
                    .filter(model_llm.Prompt.prompt_id  == component['id']))
                filter_component.append(
                    exists(subquery.where(model_llm.Agent.agent_id == model_llm.agent_prompts.c.agent_id))
                )
            if component['type'] == 'tool':
                subquery = (
                    select(model_llm.agent_tools.c.agent_id)
                    .join(model_llm.Tool, model_llm.Tool.tool_id == model_llm.agent_tools.c.tool_id)
                    .filter(model_llm.Tool.tool_id  == component['id']))
                filter_component.append(
                    exists(subquery.where(model_llm.Agent.agent_id == model_llm.agent_tools.c.agent_id))
                )
                
            # 하위에이전트
            if component['type'] == 'agent':
                subquery = (
                    select(model_llm.Agent.agent_id)
                    .filter(model_llm.Agent.parent_agent_id == component['id'])
                )
                filter_component.append(
                    exists(subquery.where(model_llm.Agent.agent_id == model_llm.Agent.agent_id))
                )
            
            if component['type'] == 'datasource':
                subquery = (
                    select(model_llm.agent_datasource.c.agent_id)
                    .join(model_llm.DataSource, model_llm.DataSource.datasource_id == model_llm.agent_datasource.c.datasource_id)
                    .filter(model_llm.DataSource.datasource_id == component['id'])
                )
                filter_component.append(
                    exists(subquery.where(model_llm.Agent.agent_id == model_llm.agent_datasource.c.agent_id))
                )
    query = query.filter(or_(*filter_component))    

    return query



@router.post(
    "/search_agents",
    response_model=schema_llm.SearchAgentsResponse,
    dependencies=[Depends(cookie)],
    tags=["Agents"],
    description=
    """
    ***Searches for agents based on various criteria.***<br/><br/>
    **Request Body**<br/>
        - **search_words (Optional[str])**: Words to search for in names and descriptions.<br/>
        - **search_range_list (Optional[List[str]])**: List of fields to search within (e.g., 'name', 'description').<br/>
        - **visibility_list (Optional[List[str]])**: List of visibility types to filter by (e.g., 'public','private').<br/>
        - **component_list (Optional[List[int]])**: List of components to filter by.<br/>
        - **tag_list (Optional[List[int]])**: List of tags to filter by.<br/>
        - **user_list (Optional[List[str]])**: List of users to filter by.<br/>
        - **llm_api_list (Optional[List[int]])**: List of LLM APIs to filter by.<br/>
        - **llm_model_list (Optional[List[str]])**: List of LLM models to filter by.<br/>
        - **prompt_list : Optional[List[int]]**: List of prompt to filter.<br/>
        - **tool_list : Optional[List[int]]**: List of tool to filter.<br/>
        - **datasource_list : Optional[List[str]]**: List of datasource to filter.<br/>
        - **skip (Optional[int])**: The number of records to skip for pagination.<br/>
        - **limit (Optional[int])**: The maximum number of records to return for pagination.
    """
)
def search_agents(
    search: schema_llm.AgentSearch,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    search_exclude = search.dict(exclude_unset=True)
    
    query = search_agent_basic(db, search_exclude)
    
    # 사용자 필터링
    user_filter_basic = [
        model_llm.Agent.create_user == session_data.user_id ,
        model_llm.Agent.visibility == 'public'
    ]
    query = query.filter(or_(*user_filter_basic))
    total_count = query.count()

    query = query.order_by(model_llm.Agent.updated_at.desc(),model_llm.Agent.created_at.desc())
    if 'skip' in search_exclude and 'limit' in search_exclude:
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit'])

    agents = query.all()

    return schema_llm.SearchAgentsResponse(totalCount=total_count, list=agents)


@router.post(
    "/search_agents_all",
    response_model=schema_llm.SearchAgentsResponse,
    dependencies=[Depends(cookie)],
    tags=["Agents"],
    description=
    """
    ***개인권한과 관련이 없는 모든 Agent를 검색한다.***<br/><br/>
    **Request Body**<br/>
        - **search_words (Optional[str])**: Words to search for in names and descriptions.<br/>
        - **search_range_list (Optional[List[str]])**: List of fields to search within (e.g., 'name', 'description').<br/>
        - **visibility_list (Optional[List[str]])**: List of visibility types to filter by (e.g., 'public','private').<br/>
        - **component_list (Optional[List[int]])**: List of components to filter by.<br/>
        - **tag_list (Optional[List[int]])**: List of tags to filter by.<br/>
        - **user_list (Optional[List[str]])**: List of users to filter by.<br/>
        - **llm_api_list (Optional[List[int]])**: List of LLM APIs to filter by.<br/>
        - **llm_model_list (Optional[List[str]])**: List of LLM models to filter by.<br/>
        - **prompt_list : Optional[List[int]]**: List of prompt to filter.<br/>
        - **tool_list : Optional[List[int]]**: List of tool to filter.<br/>
        - **datasource_list : Optional[List[str]]**: List of datasource to filter.<br/>
        - **skip (Optional[int])**: The number of records to skip for pagination.<br/>
        - **limit (Optional[int])**: The maximum number of records to return for pagination.
    """
)
def search_agents_all(
    search: schema_llm.AgentSearch,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    search_exclude = search.dict(exclude_unset=True)
    
    query = search_agent_basic(db, search_exclude)
    
    total_count = query.count()

    query = query.order_by(model_llm.Agent.updated_at.desc(),model_llm.Agent.created_at.desc())
    if 'skip' in search_exclude and 'limit' in search_exclude:
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit'])

    agents = query.all()

    return schema_llm.SearchAgentsResponse(totalCount=total_count, list=agents)


class AgentNameRequest(BaseModel):
    name: str

@router.post(
    "/check_name",
    response_model=schema_llm.SearchAgentsResponse,  # Adjust the response model to the correct one for agents
    dependencies=[Depends(cookie)],  # Adjust this dependency based on your authentication setup
    tags=["Agents"],
    description="""
    지정된 이름을 가진 에이전트가 존재하는지 확인합니다. 
    결과로 일치하는 에이전트의 목록과 총 개수를 반환합니다.
    """
)
def check_agent_name(request: AgentNameRequest, db: Session = Depends(get_db)):
    """
    지정된 이름을 가진 에이전트가 존재하는지 확인하고, 결과를 리스트 형식으로 반환합니다.
    """
    query = db.query(model_llm.Agent).filter(
        model_llm.db_comment_endpoint ,
        model_llm.Agent.name == request.name
    )
    total_count = query.count()
    results = query.all()  # Fetch all matching records

    return schema_llm.SearchAgentsResponse(
        totalCount=total_count,
        list=results
    )

