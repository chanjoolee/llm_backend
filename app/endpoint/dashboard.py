import calendar
from datetime import datetime, timedelta
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app import models, database
from app.database import get_db
# from app.models import Agent, Message, Conversation, conversation_agent  # Adjust import paths as needed
from app.endpoint.login import cookie , SessionData , verifier
from sqlalchemy import func, or_, text


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()
comment = text("/* is_endpoint_query */ 1=1")

def rows_to_dict_list(rows):
    """
    Converts a SQLAlchemy result (list of Row objects) into a list of dictionaries.
    """
    return [row._asdict() for row in rows]

def get_agent_message_counts(db: Session, current_user: str):
    """
    Queries the database to count messages for each agent linked to the current user.
    """
    
    
    query = db.query(
        database.Agent.name.label("agent_name"), 
        func.count(database.Message.message_id).label('message_count')
    )
    
    query = query.join(database.conversation_agent, database.Agent.agent_id == database.conversation_agent.c.agent_id)
    query = query.join(database.Conversation, database.Conversation.conversation_id == database.conversation_agent.c.conversation_id)
    query = query.join(database.Message, database.Message.conversation_id == database.Conversation.conversation_id)
    query = query.filter(database.Agent.create_user == current_user)  # Filter by logged-in user
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)  
    query = query.group_by(database.Agent.name)        
    db_result = query.all() 
    
    
    # SQLAlchemy returns a list of tuples, convert to list of dictionaries
    # return [{"agent_name": row.agent_name, "message_count": row.message_count} for row in result]
    return rows_to_dict_list(db_result)

def get_agent_message_counts_sql(db: Session, current_user: str):
    """
    Queries the database to count messages for each agent linked to the current user.
    """
    
    sql_query = """
        SELECT a.name AS agent_name, COUNT(m.message_id) AS message_count
        FROM agents AS a
        JOIN conversation_agent AS ca ON a.agent_id = ca.agent_id
        JOIN conversations AS c ON ca.conversation_id = c.conversation_id
        JOIN messages AS m ON c.conversation_id = m.conversation_id
        WHERE a.create_user = :current_user
        GROUP BY a.name
    """
    result = db.execute(sql_query, {"current_user": current_user}).fetchall()
    return rows_to_dict_list(result)




@router.get(
    "/agent-message-count-private",
    dependencies=[Depends(cookie)],
    description="""<pre>
    <h3>에이전트별 메세지 실행 현황. 내가만든것만</h3>
    </pre>
    """
    
)
def get_agent_message_count_private(
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)  # Fetch session data for current user
):
    """
    Endpoint to fetch message counts per agent for the current user.
    """
    current_user = session_data.user_id  # Access username from session_data
    query = db.query(
        database.Agent.name.label("agent_name"), 
        func.count(database.Message.message_id).label('message_count')
    )
    
    query = query.join(database.conversation_agent, database.Agent.agent_id == database.conversation_agent.c.agent_id)
    query = query.join(database.Conversation, database.Conversation.conversation_id == database.conversation_agent.c.conversation_id)
    query = query.join(database.Message, database.Message.conversation_id == database.Conversation.conversation_id)
    query = query.filter(database.Agent.create_user == current_user)  # Filter by logged-in user
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)  
    query = query.group_by(database.Agent.name)        
    db_result = query.all()
    
    return rows_to_dict_list(db_result)

@router.get(
    "/agent-message-count",
    dependencies=[Depends(cookie)],
    description="""<pre>
    <h3>에이전트별 메세지 실행 현황. 내것과 공개</h3>
    </pre>
    """
)
def get_agent_message_count(
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    """
    Endpoint to get message counts grouped by nickname and agent name.
    """
    comment = text("/* is_endpoint_query */ 1=1")
    user_filter_basic = [
        database.Agent.create_user == session_data.user_id ,
        database.Agent.visibility == 'public'
    ]
    query = (
        db.query(
            database.Agent.name.label("agent_name"),
            # database.User.nickname.label("nickname"),
            func.count(database.Message.message_id).label("message_count")
        )
        .join(database.User, database.Agent.create_user == database.User.user_id)
        .join(database.conversation_agent, database.Agent.agent_id == database.conversation_agent.c.agent_id)
        .join(database.Conversation, database.Conversation.conversation_id == database.conversation_agent.c.conversation_id)
        .join(database.Message, database.Message.conversation_id == database.Conversation.conversation_id)
        .filter(comment)   
        .filter(or_(*user_filter_basic))
        .group_by(
            database.Agent.name, 
            # database.User.nickname
        )
    )
    result = query.all()
    return rows_to_dict_list(result)



@router.post(
    "/message-token",
    dependencies=[Depends(cookie)],
    description="""  
    <pre>
    
    This endpoint returns token count usage, grouped by the following parameters:
    - **llm_api_name**: The name of the LLM API used (e.g., "OpenAI", "Anthropic").
    - **llm_model**: The model of the LLM (e.g., "gpt-3", "gpt-4"). The model is filtered by name.
    - **Agent**: The agent name that processed the message.
    - **visibility**: Whether the message is considered "public" or "private".
    - **send_at**: The date when the message was sent, formatted as YYYY-MM-DD.
    
    The endpoint allows filtering by the following parameters:
    - **llm_api_list**: A list of LLM API IDs (optional).
    - **llm_model_list**: A list of LLM model names (optional).
    - **start_date**: The start date for the filter (optional, default: the first day of the current month).
    - **end_date**: The end date for the filter (optional, default: 30 days from the start date).
    
    The response includes aggregated token usage:
    - **total_input_token**: Total number of input tokens.
    - **total_output_token**: Total number of output tokens.
    - **total_token**: The total number of tokens (input + output). 
    
    """
)
def get_grouped_token(
    search: models.GroupedTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Endpoint Token Counts By:
    Group: llm_api_name ,llm_model , Agent, visibility, send_at.
    """
    
    """
    Queries and calculates token usage grouped by llm_api_name ,llm_model , Agent, visibility, send_at.
    """
    
    search_exclude = search.dict(exclude_unset=True)
    
    query = (
        db.query(
            database.LlmApi.llm_api_name.label("llm_api_name"),
            database.Conversation.llm_model.label("llm_model"),
            database.Agent.name.label("agent_name"),
            database.Agent.visibility.label("visibility"),
            func.DATE_FORMAT(database.Message.sent_at, "%Y-%m-%d").label("sent_at"),
            func.sum(database.Message.input_token).label("total_input_token"),
            func.sum(database.Message.output_token).label("total_output_token"),
            func.sum(database.Message.total_token).label("total_token")
        )
        .join(database.conversation_agent, database.Agent.agent_id == database.conversation_agent.c.agent_id)
        .join(database.Conversation, database.Conversation.conversation_id == database.conversation_agent.c.conversation_id)
        .join(database.Message, database.Message.conversation_id == database.Conversation.conversation_id)
        .join(database.LlmApi, database.Agent.llm_api_id == database.LlmApi.llm_api_id)
    )
    comment = text("/* is_endpoint_query */ 1=1")
    query = query.filter(comment)   
    
    # Apply filters
    if 'llm_api_list' in search_exclude and len(search_exclude['llm_api_list']) > 0:
        query = query.filter(database.LlmApi.llm_api_id.in_(search_exclude['llm_api_list']))
    if 'llm_model_list' in search_exclude and len(search_exclude['llm_model_list']) > 0:
        # query = query.filter(database.Conversation.llm_model.like(f"%{search_exclude['llm_model']}%"))  # LIKE filter
        query = query.filter(database.Conversation.llm_model.in_(search_exclude['llm_model_list']))
    
    # Default date handling
    if not 'start_date' in search_exclude:
        now = datetime.now()
        start_date = datetime(now.year, now.month, 1).strftime("%Y-%m-%d")
    else:
        start_date = parse_date(search_exclude['start_date'])

    if not 'end_date' in  search_exclude:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = (start_date_dt + timedelta(days=30)).strftime("%Y-%m-%d")
    else:
        end_date = parse_date(search_exclude['end_date'])
        
    query = query.filter(database.Message.sent_at >= start_date)
    query = query.filter(database.Message.sent_at <= end_date)
    
    
    # Grouping and executing the query
    query = query.group_by(
        database.LlmApi.llm_api_name,
        database.Conversation.llm_model,
        database.Agent.name,
        database.Agent.visibility,
        func.DATE_FORMAT(database.Message.sent_at, "%Y-%m-%d")
    )
    
    result = query.all()
    return rows_to_dict_list(result)



def parse_date(date_str: str) -> str:
    """
    Parse a date string and return it in the standard format YYYY-MM-DD.
    Supports formats: YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD.
    """
    for date_format in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise HTTPException(status_code=400, detail=f"Invalid date format: {date_str}. Use YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD.")

def get_last_day_of_month(date: datetime) -> datetime:
    """
    Returns the last day of the month for a given date.
    """
    last_day = calendar.monthrange(date.year, date.month)[1]  # Get the last day of the month
    return datetime(date.year, date.month, last_day)

@router.get(
    "/conversation-daily-count",
    dependencies=[Depends(cookie)],
    description="""
    일별 시스템경고현황
    <pre>Daily Conversation Counts Grouped by Date and Nickname</pre>
    """
)
def get_daily_conversation_count(
    start_date: str = Query(None, description="Start date in YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD format"),
    end_date: str = Query(None, description="End date in YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD format"),
    db: Session = Depends(get_db),
):
    """
    Endpoint to get the daily count of conversations grouped by date and system (nickname).
    """
    # If no start_date is provided, default to 30 days ago
    if not start_date:
        now = datetime.now()
        start_date = datetime(now.year, now.month, 1).strftime("%Y-%m-%d")
    else:
        start_date = parse_date(start_date)

    if not end_date:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = get_last_day_of_month(start_date_dt).strftime("%Y-%m-%d")
    else:
        end_date = parse_date(end_date)

    # Query the database
    comment = text("/* is_endpoint_query */ 1=1")
    query = (
        db.query(
            func.DATE_FORMAT(database.Conversation.started_at, "%Y-%m-%d").label("date"),  # Group by formatted date
            database.User.nickname.label("nickname"),
            func.count(database.Conversation.conversation_id).label("count")
        )
        .join(database.User, database.Conversation.user_id == database.User.user_id)  # Join users to conversations
        .filter(database.User.user_roll == 'ALERT')
        .filter(database.Conversation.started_at >= start_date)  # Filter by start_date
        .filter(database.Conversation.started_at <= end_date)    # Filter by end_date
        .filter(comment)   
        .group_by(
            func.DATE_FORMAT(database.Conversation.started_at, "%Y-%m-%d"),
            database.User.nickname
        )
        .order_by(func.DATE_FORMAT(database.Conversation.started_at, "%Y-%m-%d"))  # Sort by date
    )
    result = query.all()

    # if not result:
    #     raise HTTPException(status_code=404, detail="No conversations found for the specified date range.")

    # Format the response
    return rows_to_dict_list(result)




@router.get(
    "/conversation-summary",
    dependencies=[Depends(cookie)],
    description="""
    공개/비공개된 대화/에이전트/프롬프트/도구/데이터소스 수
    ORYX-169
    대화가 공개인지 비공개인지에 따라 해당 콤포넌트 숫자를 센다
    """
)
def get_conversation_summary(
    start_date: str = Query(None, description="Start date in YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD format"),
    end_date: str = Query(None, description="End date in YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD format"),
    db: Session = Depends(get_db),
    # session_data: SessionData = Depends(verifier)
):
    """
    Fetch summary data of counts of agents, prompts, tools, and datasources grouped by conversation_type.
    """
    # Default date handling
    if not start_date:
        now = datetime.now()
        start_date = datetime(now.year, now.month, 1).strftime("%Y-%m-%d")
    else:
        start_date = parse_date(start_date)

    if not end_date:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = (start_date_dt + timedelta(days=30)).strftime("%Y-%m-%d")
    else:
        end_date = parse_date(end_date)

    # user_filter_agent = [
    #     database.Agent.create_user == session_data.user_id ,
    #     database.Agent.visibility == 'public'
    # ]
    # user_filter_prompt = [
    #     database.Prompt.create_user == session_data.user_id ,
    #     database.Prompt.open_type == 'public'
    # ]
    # user_filter_tool = [
    #     database.Tool.create_user == session_data.user_id ,
    #     database.Tool.visibility == 'public'
    # ]
    # user_filter_datasource = [
    #     database.DataSource.create_user == session_data.user_id ,
    #     database.DataSource.visibility == 'public'
    # ]
    # user_filter_conversation = [
    #     database.Conversation.user_id == session_data.user_id,
    #     database.Conversation.conversation_type == 'public'
    # ]
    
    # SQLAlchemy Query
    query = (
        db.query(
            database.Conversation.conversation_type.label("conversation_type"),
            func.count(func.distinct(database.conversation_agent.c.agent_id)).label("agent_count"),
            func.count(func.distinct(database.conversation_prompt.c.prompt_id)).label("prompt_count"),
            func.count(func.distinct(database.conversation_tools.c.tool_id)).label("tool_count"),
            func.count(func.distinct(database.conversation_datasource.c.datasource_id)).label("datasource_count"),
            func.count(database.Conversation.conversation_id).label("conversation_count")
        )
        .outerjoin(database.conversation_agent, database.Conversation.conversation_id == database.conversation_agent.c.conversation_id)
        .outerjoin(database.conversation_prompt, database.Conversation.conversation_id == database.conversation_prompt.c.conversation_id)
        .outerjoin(database.conversation_tools, database.Conversation.conversation_id == database.conversation_tools.c.conversation_id)
        .outerjoin(database.conversation_datasource, database.Conversation.conversation_id == database.conversation_datasource.c.conversation_id)
        .filter(comment) 
        .filter(database.Conversation.started_at >= start_date)  # Filter by start_date
        .filter(database.Conversation.started_at <= end_date)  # Filter by end_date
        # .filter(or_(*user_filter_agent))
        # .filter(or_(*user_filter_prompt))
        # .filter(or_(*user_filter_tools))
        # .filter(or_(*user_filter_datasource))
        # .filter(or_(*user_filter_conversation))
        .group_by(database.Conversation.conversation_type)  # Group by conversation_type
        .order_by(database.Conversation.conversation_type)  # Optional: order by conversation_type
    )

    result = query.all()
    return rows_to_dict_list(result)



@router.get(
    "/visibility-summary",
    dependencies=[Depends(cookie)],
    description="""
    대시보드 하단의 콤포넌트현황
    
    공개/비공개된 대화/에이전트/프롬프트/도구/데이터소스 수
    ORYX-169
    공개인지 비공개인지에 따른 해당 콤포넌트 숫자를 센다
    """
)
def get_visibility_summary(
    start_date: str = Query(None, description="Start date in YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD format"),
    end_date: str = Query(None, description="End date in YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD format"),
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    """
    Fetch summary data of counts of agents, prompts, tools, datasources, and conversations grouped by visibility.
    """
    # Default date handling
    if not start_date:
        now = datetime.now()
        start_date = datetime(now.year, now.month, 1).strftime("%Y-%m-%d")
    else:
        start_date = parse_date(start_date)

    if not end_date:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = (start_date_dt + timedelta(days=30)).strftime("%Y-%m-%d")
    else:
        end_date = parse_date(end_date)

    user_filter_agent = [
        database.Agent.create_user == session_data.user_id ,
        database.Agent.visibility == 'public'
    ]
    user_filter_prompt = [
        database.Prompt.create_user == session_data.user_id ,
        database.Prompt.open_type == 'public'
    ]
    user_filter_tool = [
        database.Tool.create_user == session_data.user_id ,
        database.Tool.visibility == 'public'
    ]
    user_filter_datasource = [
        database.DataSource.create_user == session_data.user_id ,
        database.DataSource.visibility == 'public'
    ]
    user_filter_conversation = [
        database.Conversation.user_id == session_data.user_id,
        database.Conversation.conversation_type == 'public'
    ]
    
    # SQLAlchemy Query grouped by visibility
    conversations_query = (
        db.query(
            database.Conversation.conversation_type.label("visibility"),
            func.count(database.Conversation.conversation_id).label("conversation_count")
        )
        .filter(database.Conversation.started_at >= start_date)
        .filter(database.Conversation.started_at <= end_date)
        .filter(comment) 
        .filter(or_(*user_filter_conversation))
        .group_by(database.Conversation.conversation_type)
    )

    agents_query = (
        db.query(
            database.Agent.visibility.label("visibility"),
            func.count(database.Agent.agent_id).label("agent_count")
        )
        .filter(comment) 
        .filter(or_(*user_filter_agent))
        .group_by(database.Agent.visibility)
    )

    prompts_query = (
        db.query(
            database.Prompt.open_type.label("visibility"),
            func.count(database.Prompt.prompt_id).label("prompt_count")
        )
        .filter(comment) 
        .filter(or_(*user_filter_prompt))
        .group_by(database.Prompt.open_type)
    )

    tools_query = (
        db.query(
            database.Tool.visibility.label("visibility"),
            func.count(database.Tool.tool_id).label("tool_count")
        )
        .filter(comment) 
        .filter(or_(*user_filter_tool))
        .group_by(database.Tool.visibility)
    )

    datasources_query = (
        db.query(
            database.DataSource.visibility.label("visibility"),
            func.count(database.DataSource.datasource_id).label("datasource_count")
        )
        .filter(comment) 
        .filter(or_(*user_filter_datasource))
        .group_by(database.DataSource.visibility)
    )

    # Execute all queries
    conversations_result = conversations_query.all()
    agents_result = agents_query.all()
    prompts_result = prompts_query.all()
    tools_result = tools_query.all()
    datasources_result = datasources_query.all()

    # Combine results by visibility
    visibility_summary = {}
    
    for row in conversations_result:
        visibility_summary[row.visibility] = {
            "conversation_count": row.conversation_count,
            "agent_count": 0,
            "prompt_count": 0,
            "tool_count": 0,
            "datasource_count": 0
        }

    for row in agents_result:
        if row.visibility not in visibility_summary:
            visibility_summary[row.visibility] = {
                "conversation_count": 0,
                "agent_count": row.agent_count,
                "prompt_count": 0,
                "tool_count": 0,
                "datasource_count": 0
            }
        else:
            visibility_summary[row.visibility]["agent_count"] = row.agent_count

    for row in prompts_result:
        if row.visibility not in visibility_summary:
            visibility_summary[row.visibility] = {
                "conversation_count": 0,
                "agent_count": 0,
                "prompt_count": row.prompt_count,
                "tool_count": 0,
                "datasource_count": 0
            }
        else:
            visibility_summary[row.visibility]["prompt_count"] = row.prompt_count

    for row in tools_result:
        if row.visibility not in visibility_summary:
            visibility_summary[row.visibility] = {
                "conversation_count": 0,
                "agent_count": 0,
                "prompt_count": 0,
                "tool_count": row.tool_count,
                "datasource_count": 0
            }
        else:
            visibility_summary[row.visibility]["tool_count"] = row.tool_count

    for row in datasources_result:
        if row.visibility not in visibility_summary:
            visibility_summary[row.visibility] = {
                "conversation_count": 0,
                "agent_count": 0,
                "prompt_count": 0,
                "tool_count": 0,
                "datasource_count": row.datasource_count
            }
        else:
            visibility_summary[row.visibility]["datasource_count"] = row.datasource_count

    # Convert to list for JSON response
    return [
        {
            "visibility": visibility,
            **counts
        }
        for visibility, counts in visibility_summary.items()
    ]
