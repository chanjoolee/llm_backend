from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import insert, or_, text
from sqlalchemy.orm import Session
from app import models, database
from app.utils import utils
from app.endpoint.login import cookie , SessionData , verifier
from app.database import SessionLocal, get_db

router = APIRouter()


@router.post(
    "/create_agent",
    response_model=models.Agent,
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
    agent: models.AgentCreate,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    db_agent = database.Agent(
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
            db_prompt = db.query(database.Prompt).filter(database.Prompt.prompt_id == prompt.prompt_id).first()

            if db_prompt:
                stmt = insert(database.agent_prompts).values(
                    agent_id=db_agent.agent_id,
                    prompt_id=db_prompt.prompt_id
                    # ,sort_order=i+1
                )
                db.execute(stmt)

                for db_prompt_message in db_prompt.promptMessage:
                    message = db_prompt_message.message
                    for variable in prompt.variables:
                        db_variable = database.AgentVariable(
                            agent_id=db_agent.agent_id,
                            variable_name=variable.variable_name,
                            variable_value=variable.value
                        )
                        db.add(db_variable)

                    # 이 부분은 어떻게 처리를 하나 ==> conversation 을 만들때 처리하는 것이 좋을 듯. 어떻게? ...
                    # new_message = database.Message(
                    #     conversation_id=db_agent.agent_id,
                    #     message_type=db_prompt_message.message_type,
                    #     message=message,
                    #     input_path='prompt'
                    # )
                    # db.add(new_message)


    # Handle tools
    if 'tools' in update_data:
        for tool_id in update_data['tools']:
            db_tool = db.query(database.Tool).filter(database.db_comment_endpoint).filter(database.Tool.tool_id == tool_id).first()
            if db_tool:
                db_agent.tools.append(db_tool)

    # Handle tags
    if 'tags' in update_data:
        for tag_id in agent.tags:
            db_tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
            if db_tag:
                db_agent.tags.append(db_tag)

    # Handle child agents associations
    if 'sub_agents' in update_data:
        for child_agent_id in agent.sub_agents:
            db_child_agent = db.query(database.Agent).filter(database.Agent.agent_id == child_agent_id).first()
            if db_child_agent:
                db_child_agent.parent_agent_id = db_agent.agent_id
                db.add(db_child_agent)
    if 'datasources' in update_data:
        for datasource_id in agent.datasources:
            db_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
            if db_datasource:
                db_agent.datasources.append(db_datasource)
    db.flush()
    db.refresh(db_agent)

    return db_agent

@router.get(
    "/get_agent/{agent_id}",
    response_model=models.Agent,
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
    query = db.query(database.Agent).filter(database.db_comment_endpoint)
    query = query.filter(database.Agent.agent_id == agent_id)

    db_agent = query.first()
    if db_agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return db_agent


@router.put(
    "/update_agent/{agent_id}",
    response_model=models.Agent,
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
    agent: models.AgentUpdate,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(database.Agent).filter(database.db_comment_endpoint)
    query = query.filter(database.Agent.agent_id == agent_id)
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
        db.query(database.AgentVariable).filter(database.AgentVariable.agent_id == agent_id).delete(synchronize_session='fetch')
        for i, prompt in enumerate(agent.prompts):
            db_prompt = db.query(database.Prompt).filter(database.Prompt.prompt_id == prompt.prompt_id).first()

            if db_prompt:
                # stmt = insert(database.agent_prompts).values(
                #     agent_id=db_agent.agent_id,
                #     prompt_id=db_prompt.prompt_id
                # )
                # db.execute(stmt)
                db_agent.prompts.append(db_prompt)

                for db_prompt_message in db_prompt.promptMessage:
                    message = db_prompt_message.message
                    for variable in prompt.variables:
                        db_variable = database.AgentVariable(
                            agent_id=db_agent.agent_id,
                            variable_name=variable.variable_name,
                            variable_value=variable.value
                        )
                        db.add(db_variable)

                    # 이 부분은 어떻게 처리를 하나 ==> conversation 을 만들때 처리하는 것이 좋을 듯. 어떻게? ...
                    # new_message = database.Message(
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
            db_tool = db.query(database.Tool).filter(database.db_comment_endpoint).filter(database.Tool.tool_id == tool_id).first()
            if db_tool:
                db_agent.tools.append(db_tool)

    # Handle tags
    if 'tags' in update_data:
        db_agent.tags = []
        for tag_id in agent.tags:
            db_tag = db.query(database.Tag).filter(database.db_comment_endpoint).filter(database.Tag.tag_id == tag_id).first()
            if db_tag:
                db_agent.tags.append(db_tag)

    # Handle child agents associations
    if 'sub_agents' in update_data:
        current_child_agents = {child.agent_id for child in db_agent.sub_agents}
        new_child_agents = set(update_data['sub_agents'])

        # Remove relationships that are no longer valid
        agents_to_remove = current_child_agents - new_child_agents
        for child_agent_id in agents_to_remove:
            db_child_agent = db.query(database.Agent).filter(database.Agent.agent_id == child_agent_id).first()
            if db_child_agent:
                db_child_agent.parent_agent_id = None

        # Add new relationships
        agents_to_add = new_child_agents - current_child_agents
        for child_agent_id in agents_to_add:
            db_child_agent = db.query(database.Agent).filter(database.Agent.agent_id == child_agent_id).first()
            if db_child_agent:
                db_child_agent.parent_agent_id = db_agent.agent_id

    if 'datasources' in update_data:
        db_agent.datasources = []
        for datasource_id in agent.datasources:
            db_datasource = db.query(database.DataSource).filter(database.db_comment_endpoint).filter(database.DataSource.datasource_id == datasource_id).first()
            if db_datasource:
                db_agent.datasources.append(db_datasource)
    db.flush()
    db.refresh(db_agent)
    
    return db_agent

@router.delete(
    "/delete_agent/{agent_id}",
    response_model=models.Agent,
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
    query = db.query(database.Agent).filter(database.db_comment_endpoint)
    query = query.filter(database.Agent.agent_id == agent_id)
    db_agent = query.first()

    if db_agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    
    deleted_data = models.Agent.from_orm(db_agent)   
    db_agent.prompts = []
    db_agent.tools = []
    db_agent.tags = []
    db_agent.conversations = [] 
    db.delete(db_agent)
    
    return deleted_data


@router.post(
    "/search_agents",
    response_model=models.SearchAgentsResponse,
    dependencies=[Depends(cookie)],
    tags=["Agents"],
    description="""<pre>
    Searches for agents based on various criteria.
    <h3>Request Body:</h3>
        - search_words (Optional[str]): Words to search for in names and descriptions.
        - search_range_list (Optional[List[str]]): List of fields to search within (e.g., 'name', 'description').
        - visibility_list (Optional[List[str]]): List of visibility types to filter by (e.g., 'public','private').
        - component_list (Optional[List[int]]): List of components to filter by.
        - tag_list (Optional[List[int]]): List of tags to filter by.
        - user_list (Optional[List[str]]): List of users to filter by.
        - llm_api_list (Optional[List[int]]): List of LLM APIs to filter by.
        - llm_model_list (Optional[List[str]]): List of LLM models to filter by.
        - skip (Optional[int]): The number of records to skip for pagination.
        - limit (Optional[int]): The maximum number of records to return for pagination.
    </pre>
    """
)
def search_agents(
    search: models.AgentSearch,
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    query = db.query(database.Agent).filter(database.db_comment_endpoint)
    
    search_exclude = search.dict(exclude_unset=True)

    if 'search_words' in search_exclude and search_exclude['search_words']:
        if 'name' in search_exclude['search_range_list']:
            query = query.filter(database.Agent.name.like(f'%{search_exclude["search_words"]}%'))
        if 'description' in search_exclude['search_range_list']:
            query = query.filter(database.Agent.description.like(f'%{search_exclude["search_words"]}%'))

    if 'visibility_list' in search_exclude and search_exclude['visibility_list']:
        query = query.filter(database.Agent.visibility.in_(search_exclude['visibility_list']))

    if 'tag_list' in search_exclude and search_exclude['tag_list']:
        query = query.join(database.agent_tags).filter(database.agent_tags.c.tag_id.in_(search_exclude['tag_list']))

    # 사용자 필터링
    user_filter_basic = [
        database.Agent.create_user == session_data.user_id ,
        database.Agent.visibility == 'public'
    ]
    query = query.filter(or_(*user_filter_basic))
    if 'user_list' in search_exclude and len(search_exclude['user_list']) > 0:
        query = query.filter(database.Agent.create_user .in_(search_exclude['user_list']))

    if 'llm_api_list' in search_exclude and search_exclude['llm_api_list']:
        query = query.filter(database.Agent.llm_api_id.in_(search_exclude['llm_api_list']))

    if 'llm_model_list' in search_exclude and search_exclude['llm_model_list']:
        query = query.filter(database.Agent.llm_model.in_(search_exclude['llm_model_list']))

    total_count = query.count()

    if 'skip' in search_exclude and 'limit' in search_exclude:
        query = query.offset(search_exclude['skip']).limit(search_exclude['limit'])

    agents = query.all()

    return models.SearchAgentsResponse(totalCount=total_count, list=agents)



