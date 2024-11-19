from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app import models, database
from app.database import get_db
# from app.models import Agent, Message, Conversation, conversation_agent  # Adjust import paths as needed
from app.endpoint.login import cookie , SessionData , verifier
from sqlalchemy import func

router = APIRouter()

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

def get_message_counts_grouped(db: Session):
    """
    Queries the database to count messages grouped by nickname and agent_name.
    """
    query = (
        db.query(
            database.Agent.name.label("agent_name"),
            database.User.nickname.label("nickname"),
            func.count(database.Message.message_id).label("message_count")
        )
        .join(database.User, database.Agent.create_user == database.User.user_id)
        .join(database.conversation_agent, database.Agent.agent_id == database.conversation_agent.c.agent_id)
        .join(database.Conversation, database.Conversation.conversation_id == database.conversation_agent.c.conversation_id)
        .join(database.Message, database.Message.conversation_id == database.Conversation.conversation_id)
        .group_by(database.Agent.name, database.User.nickname)
    )
    result = query.all()
    return rows_to_dict_list(result)



@router.get(
    "/agent-message-count",
    dependencies=[Depends(cookie)],
    description="""<pre>
    <h3>에이전트별 메세지 실행 현황</h3>
    </pre>
    """
    
)
def get_agents_message_count(
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)  # Fetch session data for current user
):
    """
    Endpoint to fetch message counts per agent for the current user.
    """
    current_user = session_data.user_id  # Access username from session_data
    data = get_agent_message_counts(db, current_user)
    return data

@router.get(
    "/message-counts-grouped",
    dependencies=[Depends(cookie)],
    description="""<pre>
    <h3>Message Counts Grouped by Nickname and Agent Name</h3>
    </pre>
    """
)
def get_message_counts_grouped_endpoint(
    db: Session = Depends(get_db),
    session_data: SessionData = Depends(verifier)
):
    """
    Endpoint to get message counts grouped by nickname and agent name.
    """
    data = get_message_counts_grouped(db)
    return data




def get_token_usage_by_group(db: Session):
    """
    Queries and calculates token usage grouped by llm_api_name ,llm_model , Agent, visibility, send_at.
    """
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
        .group_by(
            database.LlmApi.llm_api_name,
            database.Conversation.llm_model,
            database.Agent.name,            
            database.Agent.visibility,
            func.DATE_FORMAT(database.Message.sent_at, "%Y-%m-%d")
        )
    )
    result = query.all()
    return rows_to_dict_list(result)


@router.get(
    "/message-token",
    dependencies=[Depends(cookie)],
    description="""<pre>Token Count Usage grouped by llm_api_name ,llm_model , Agent, visibility, send_at.  </pre>"""
)
def get_grouped_token(db: Session = Depends(get_db)):
    """
    Endpoint Token Counts By:
    Group: llm_api_name ,llm_model , Agent, visibility, send_at.
    """
    return get_token_usage_by_group(db)