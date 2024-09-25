import enum
from sqlalchemy import Boolean, Float, UniqueConstraint, create_engine ,Table, Column, Integer, String, Text, Enum, ForeignKey, TIMESTAMP , event, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker , backref
from urllib.parse import quote
from datetime import datetime
from enum import Enum as PyEnum
import os
import app.config
from typing import ForwardRef
from ai_core.checkpoint.mysql_saver import MySQLSaver
from ai_core.data_source.base import DataSourceType
import pytz

KST = pytz.timezone('Asia/Seoul')
# PromptMessages = ForwardRef('PromptMessages')

username = os.getenv('db_connection_username')
password = os.getenv('history_connection_password')
host = os.getenv('db_connection_host')
port = os.getenv('db_connection_port')
database = os.getenv('db_connection_database')
encoded_password = quote(password)

# SQLALCHEMY_DATABASE_URL = "mysql+pymysql://user:password@localhost/dbname"
# SQLALCHEMY_DATABASE_URL = "mysql+pymysql://grechan:gre!2lee3@grechan.cafe24.com:3306/grechan"
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
db_comment_endpoint = text("/* is_endpoint_query */ 1=1")

# Check environment variable for charset setting
use_mysql_charset = os.getenv('USE_MYSQL_CHARSET', 'False') == 'True'
mysql_charset = os.getenv('MYSQL_CHARSET', 'utf8mb3')
mysql_collate = os.getenv('MYSQL_COLLATE', 'utf8mb3_general_ci')

table_args = {}
if use_mysql_charset:
    table_args = {
        'mysql_charset': mysql_charset,
        'mysql_collate': mysql_collate
    }


sync_conn_pool = None
async_conn_pool = None

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e :
        db.rollback()
        raise e
    finally:
        db.close()

class CustomSession(Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_endpoint_query = False

     
def create_sync_connection_pool():
    global sync_conn_pool
    if not sync_conn_pool:
        username = os.getenv('history_connection_username')
        password = os.getenv('history_connection_password')
        host = os.getenv('history_connection_host')
        port = os.getenv('history_connection_port')
        database_name = os.getenv('history_connection_database')

        encoded_password = quote(password)
        connection_url = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database_name}"

        sync_conn_pool = MySQLSaver.create_sync_connection_pool(
            host=host,
            user=username,
            password=password,
            db=database_name,
            port=int(port),
            autocommit=True
        )
        MySQLSaver.create_tables(sync_conn_pool)
    return sync_conn_pool

async def create_async_connection_pool():
    global async_conn_pool
    if not async_conn_pool:
        username = os.getenv('history_connection_username')
        password = os.getenv('history_connection_password')
        host = os.getenv('history_connection_host')
        port = os.getenv('history_connection_port')
        database_name = os.getenv('history_connection_database')
        encoded_password = quote(password)

        async_conn_pool = await MySQLSaver.create_async_connection_pool(
            host=host,
            user=username,
            password=password,
            db=database_name,
            port=int(port),
            autocommit=True
        )
    return async_conn_pool

async def get_async_connection_pool():
    return await create_async_connection_pool()

def get_sync_connection_pool():
    return create_sync_connection_pool()

async def reconnect_async_pool():
    global async_conn_pool
    if async_conn_pool:
        async_conn_pool.close()
        await async_conn_pool.wait_closed()
    await create_async_connection_pool()

def reconnect_sync_pool():
    global sync_conn_pool
    if sync_conn_pool:
        sync_conn_pool.close()
    create_sync_connection_pool()


class UserRoll(PyEnum):
    GUEST = "GUEST"
    ADMIN = "ADMIN"
    DEVELOPER = "DEVELOPER"

class CommonCodeGroup(Base):
    __tablename__ = 'common_code_group'
    __table_args__ = {
        'comment': '공통코드그룹',
        # 'mysql_charset': 'utf8mb3',
        # 'mysql_collate': 'utf8mb3_general_ci'
    }
    if use_mysql_charset:
        __table_args__.update({
            'mysql_charset': mysql_charset,
            'mysql_collate': mysql_collate
        })
    code_group = Column(String(20), primary_key=True, index=True ,comment='공통코드그룹')
    group_desc = Column(String(255) ,comment='설명')    

    create_user = Column(String(255) ,comment='생성자')
    update_user = Column(String(255) ,comment='수정자')
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(KST) , comment='생성일')
    updated_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), onupdate=lambda: datetime.now(KST), comment='수정일')

    # commonCode = relationship("CommonCodeDetail",  back_populates="codeGroup")

class CommonCodeDetail(Base):
    __tablename__ = 'common_code_detail'
    __table_args__ = {
        'comment': '공통코드상세',
        # 'mysql_charset': 'utf8mb3',
        # 'mysql_collate': 'utf8mb3_general_ci'
    }    
    if use_mysql_charset:
        __table_args__.update({
            'mysql_charset': mysql_charset,
            'mysql_collate': mysql_collate
        })

    code_group = Column(String(20), ForeignKey('common_code_group.code_group',ondelete='NO ACTION'),primary_key=True, index=True ,comment='공통코드그룹')
    code_value = Column(String(20), primary_key=True, index=True ,comment='공통코드그룹')
    code_desc = Column(String(255), comment='설명')
    sort_order = Column(Integer, comment='정렬순서')
    category1 = Column(String(255), comment='필터')

    create_user = Column(String(255) ,comment='생성자')
    update_user = Column(String(255) ,comment='수정자')
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(KST) ,comment='생성일')
    updated_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), onupdate=lambda: datetime.now(KST), comment='수정일')

    codeGroup = relationship("CommonCodeGroup", back_populates="commonCode")


class User(Base):
    __tablename__ = 'users'
    __table_args__ = {
        'comment': '사용자정보',
        # 'mysql_charset': 'utf8mb3',
        # 'mysql_collate': 'utf8mb3_general_ci'
    }    
    if use_mysql_charset:
        __table_args__.update({
            'mysql_charset': mysql_charset,
            'mysql_collate': mysql_collate
        })
    user_id = Column(String(255), primary_key=True, index=True ,comment='사용자아이디 이메일')
    nickname = Column(String(255), unique=True ,comment='닉네임')
    password = Column(String(255), nullable=False , comment='비밀번호')
    user_roll = Column(String(20), nullable=False , default='GUEST' ,  comment='역할')
    token_gitlab = Column(String(255), comment='Token GitLab')
    token_confluence = Column(String(255), comment='Token Confluence')
    token_jira = Column(String(255), comment='Token Jira')
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(KST) ,comment='생성일')
    updated_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), onupdate=lambda: datetime.now(KST), comment='수정일')

    # Relation
    llm_api = relationship("LlmApi",back_populates="create_user_info")
    prompts = relationship("Prompt",back_populates="create_user_info")
    tools = relationship("Tool",back_populates="create_user_info")
    agents = relationship("Agent", back_populates="create_user_info")
    datasources = relationship("DataSource", back_populates="create_user_info")
# 태그

# Many-to-many association table between Conversation and Agent
conversation_agent = Table(
    'conversation_agent',
    Base.metadata,
    Column('conversation_id', ForeignKey('conversations.conversation_id'), primary_key=True),
    Column('agent_id', ForeignKey('agents.agent_id'), primary_key=True)
)

# Association table for the many-to-many relationship between DataSource and Conversation
conversation_datasource = Table(
    'datasource_conversation', 
    Base.metadata,
    Column('datasource_id', String(255), ForeignKey('datasources.datasource_id'), primary_key=True),
    Column('conversation_id', String(255), ForeignKey('conversations.conversation_id'), primary_key=True)
)

class ConversationVariable(Base):
    __tablename__ = 'conversation_variables'
    variable_id = Column(Integer, primary_key=True, index=True, comment='변수ID')
    conversation_id = Column(String(255), ForeignKey('conversations.conversation_id'), nullable=False, comment='대화ID')
    variable_name = Column(String(255), nullable=False, comment='변수명')
    variable_value = Column(Text, nullable=False, comment='변수값')

    conversation = relationship("Conversation", back_populates="variables")

class Conversation(Base):
    __tablename__ = 'conversations'
    __table_args__ = {
        'comment': '대화정보',
        # 'mysql_charset': 'utf8mb3',
        # 'mysql_collate': 'utf8mb3_general_ci'
    }
    if use_mysql_charset:
        __table_args__.update({
            'mysql_charset': mysql_charset,
            'mysql_collate': mysql_collate
        })

    conversation_id = Column(String(255), primary_key=True, index=True,comment='대화ID')
    conversation_title = Column(String(255), nullable=False , comment='대화제목') 
    user_id = Column(String(255), ForeignKey('users.user_id', ondelete='NO ACTION'), nullable=False,  comment='사용자아이디 이메일')
    # user_id = Column(String(255), nullable=False)
    conversation_type = Column(String(10),nullable=False, default='private', comment='대화종류 private public')

    llm_api_id = Column(Integer , ForeignKey('llm_api.llm_api_id',ondelete='NO ACTION') , nullable=True, comment='LLM API ID')    
    llm_model =  Column(String(255), nullable=False , comment='LLM Model. LLM API에서 입력된 값중 하나만 입력')

    temperature = Column(Float, nullable=False , comment='')
    max_tokens = Column(Integer, nullable=False , comment='최대토큰')
    used_tokens = Column(Integer, nullable=True , comment='토큰사용량')
    component_configuration = Column(String(255), nullable=False, default='none', comment='Component Configuration')  # New field

    last_conversation_time = Column(TIMESTAMP, comment='마지막 대화시간')
    last_message_id = Column(Integer,comment='메세지ID')
    started_at = Column(TIMESTAMP, default=lambda: datetime.now(KST),comment='생성일')
    

    # Relation
    user_info = relationship("User", back_populates="conversation")
    llm_api = relationship("LlmApi", back_populates="conversation")
    variables = relationship("ConversationVariable", back_populates="conversation" , cascade="all, delete-orphan")
    agents = relationship("Agent", secondary=conversation_agent, back_populates="conversations")
    
    # Many-to-many relationship with DataSource
    datasources = relationship(
        "DataSource",
        secondary=conversation_datasource,
        back_populates="conversations"
    )


class SessionStore(Base):
    __tablename__ = 'session_store'
    id = Column(Integer , primary_key=True, index=True)
    session_id = Column(String(100))
    message = Column(Text)

# class Checkpoints(Base):
#     __tablename__ = 'checkpoints'
#     thread_id = Column(String(255) , primary_key=True, index=True)
#     thread_ts = Column(String(255) , primary_key=True, index=True)
    


class Message(Base):
    __tablename__ = 'messages'
    __table_args__ = (
        Index('ix_messages_message', 'message', mysql_prefix='FULLTEXT',mysql_with_parser='ngram'),
        {
            'comment': '메세지',
            # 'mysql_charset': 'utf8mb3',
            # 'mysql_collate': 'utf8mb3_general_ci'
        }
    )
    if use_mysql_charset:
        __table_args__ = (
            Index('ix_messages_message', 'message', mysql_prefix='FULLTEXT',mysql_with_parser='ngram'),
            {
                'comment': '메세지',
                'mysql_charset': 'utf8mb3',
                'mysql_collate': 'utf8mb3_general_ci'
            }
        )
    
    # __table_args__ = {
    #     'comment': '메세지',
    #     'mysql_charset': 'utf8mb3',
    #     'mysql_collate': 'utf8mb3_general_ci'
    # }

    message_id = Column(Integer, primary_key=True, index=True, comment='메세지ID')
    conversation_id = Column(String(255), ForeignKey('conversations.conversation_id',ondelete='NO ACTION'), comment='대화ID')
    message_type = Column(String(20), nullable=False, comment='메세지타입 ai,human,system')
    message = Column(Text, nullable=False, comment='메세지내용')
    input_path = Column(String(20), nullable=False, comment='입력경로 prompt,conversation')
    sent_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), comment='보낸시간')

    conversation = relationship("Conversation", back_populates="messages")

class LlmApi(Base):
    __tablename__ = 'llm_api'
    __table_args__ = {
        'comment': 'LLM API',
        # 'mysql_charset': 'utf8mb3',
        # 'mysql_collate': 'utf8mb3_general_ci'
    }
    if use_mysql_charset:
        __table_args__.update({
            'mysql_charset': mysql_charset,
            'mysql_collate': mysql_collate
        })

    llm_api_id = Column(Integer, primary_key=True, index=True, comment='LLM API ID')
    llm_api_name = Column(String(255), nullable=False , comment='LLM API 이름') 
    llm_api_type = Column(String(10),nullable=False, default='private', comment='오픈종료 private,public')
    llm_api_provider = Column(String(255) ,comment='LLM 타입(API)')
    llm_api_url = Column(String(255), nullable=False , comment='LLM Url')    
    llm_api_key = Column(String(255), nullable=False , comment='LLM Api Key')
    llm_model =  Column(String(255), nullable=False , comment='LLM Model. 쉼표로 구분하여 입력')
    embedding_model = Column(String(255), nullable=True, comment='Embedding Model 쉼표로 구분하여 입력 : sentence-transformers/all-MiniLM-L6-v2')


    create_user = Column(String(255) ,ForeignKey('users.user_id', ondelete='NO ACTION'), comment='생성자')
    update_user = Column(String(255) ,comment='수정자')
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(KST) , comment='생성일')
    updated_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), onupdate=lambda: datetime.now(KST), comment='수정일')\
    
    # Relation
    conversation = relationship("Conversation", back_populates="llm_api")
    create_user_info = relationship("User", back_populates="llm_api")
    agents = relationship("Agent", back_populates="llm_api")
    embeddings = relationship("Embedding", back_populates="llm_api")


# Association table for many-to-many relationship between Conversation and Prompt
conversation_prompt = Table(
    'conversation_prompt', Base.metadata,
    Column('conversation_id', String(255), ForeignKey('conversations.conversation_id'), primary_key=True),
    Column('prompt_id', Integer, ForeignKey('prompts.prompt_id'), primary_key=True),
    Column('sort_order', Integer),
    **table_args  # Unpack table_args if it's not empty
)

conversation_tools = Table('conversation_tools', Base.metadata,
    Column('conversation_id', String(255), ForeignKey('conversations.conversation_id'), primary_key=True),
    Column('tool_id', Integer, ForeignKey('tools.tool_id'), primary_key=True),
    **table_args
)

# 태그
# Association table for the many-to-many relationship
prompt_tags = Table('prompt_tags', Base.metadata,
    Column('prompt_id', Integer, ForeignKey('prompts.prompt_id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.tag_id'), primary_key=True)
)


# Association table for the many-to-many relationship
tool_tags = Table('tool_tags', Base.metadata,
    Column('tool_id', Integer, ForeignKey('tools.tool_id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.tag_id'), primary_key=True)
)

datasource_tags = Table('datasource_tags', Base.metadata,
    Column('datasource_id', String(255), ForeignKey('datasources.datasource_id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.tag_id'), primary_key=True)
)





class Prompt(Base):
    __tablename__ = 'prompts'
    # __table_args__ = (
    #     # Index('ix_prompts_tags', 'tags', mysql_prefix='FULLTEXT') ,
    #     {
    #         'comment': '프롬프트',
    #         'mysql_charset': 'utf8mb3',
    #         'mysql_collate': 'utf8mb3_general_ci'
    #     }        
    # )
    __table_args__ = {
        'comment': '프롬프트',
        # 'mysql_charset': 'utf8mb3',
        # 'mysql_collate': 'utf8mb3_general _ci'
    }        
    if use_mysql_charset:
        __table_args__.update({
            'mysql_charset': mysql_charset,
            'mysql_collate': mysql_collate
        })

    prompt_id = Column(Integer, primary_key=True, index=True,comment='프롬프트ID')
    prompt_title =  Column(String(255), nullable=False , comment='프롬프트제목') 
    prompt_desc = Column(String(400), nullable=False , comment='프롬프트설명') 
    open_type = Column(String(20), default='private', comment='private,public')
    # tags = Column(Text, nullable=False, comment='태그')

    create_user = Column(String(255) ,ForeignKey('users.user_id', ondelete='NO ACTION') ,comment='생성자')
    update_user = Column(String(255) ,comment='수정자')
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(KST) , comment='생성일')
    updated_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), onupdate=lambda: datetime.now(KST), comment='수정일')

    conversations = relationship(
        "Conversation", 
        secondary=conversation_prompt, 
        back_populates="prompts"
    )

    promptMessage = relationship(
        "PromptMessages", 
        order_by='PromptMessages.message_id',
        cascade="all, delete-orphan",
        back_populates="prompts"
        
    )

    tags = relationship('Tag', secondary=prompt_tags , back_populates='prompts')

    variables = relationship(
        'PromptVariable', 
        cascade="all, delete-orphan",
        back_populates='prompts'
        
    )
    create_user_info = relationship("User", back_populates="prompts")
    agents = relationship("Agent", secondary="agent_prompts", back_populates="prompts")

class PromptMessages(Base):
    __tablename__ = 'prompt_messages'
    __table_args__ = (
        Index('ix_prompts_messages_message', 'message', mysql_prefix='FULLTEXT', mysql_with_parser='ngram'),
        {
            'comment': '프롬프트 메세지',
            # 'mysql_charset': 'utf8mb3',
            # 'mysql_collate': 'utf8mb3_general_ci'
        }
    )
    if use_mysql_charset:
        __table_args__ = (
            Index('ix_prompts_messages_message', 'message', mysql_prefix='FULLTEXT', mysql_with_parser='ngram'),
            {
                'comment': '프롬프트 메세지',
                'mysql_charset': 'utf8mb3',
                'mysql_collate': 'utf8mb3_general_ci'
            }
        )

    # __table_args__ = {
    #     'comment': '프롬프트 상세',
    #     'mysql_charset': 'utf8mb3',
    #     'mysql_collate': 'utf8mb3_general_ci'
    # }
    message_id = Column(Integer, primary_key=True, index=True, comment='메세지ID')  
    prompt_id = Column(Integer, ForeignKey('prompts.prompt_id',ondelete='NO ACTION'),comment='프롬프트ID')
      
    message_type = Column(String(20), comment='system, human, ai')
    message = Column(Text, nullable=False, comment='프롬프트 내용')

    create_user = Column(String(255) ,comment='생성자')
    update_user = Column(String(255) ,comment='수정자')
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(KST) , comment='생성일')
    updated_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), onupdate=lambda: datetime.now(KST), comment='수정일')

    prompts = relationship("Prompt", back_populates="promptMessage")

class PromptVariable(Base):
    __tablename__ = 'prompt_variables'
    __table_args__ = (
        UniqueConstraint('prompt_id', 'variable_name', name='uix_prompt_variable'),
        {
            'comment': '프롬프트 변수',
        }
    )
    if use_mysql_charset:
        __table_args__ = (
            UniqueConstraint('prompt_id', 'variable_name', name='uix_prompt_variable'),
            {
                'comment': '프롬프트 변수',
                'mysql_charset': 'utf8mb3',
                'mysql_collate': 'utf8mb3_general_ci'
            }
        )
    variable_id = Column(Integer, primary_key=True, index=True, comment='변수ID')
    prompt_id = Column(Integer, ForeignKey('prompts.prompt_id'), nullable=False, comment='프롬프트ID')
    variable_name = Column(String(255), nullable=False, comment='변수명')

    prompts = relationship(
        "Prompt", 
        back_populates="variables"
    )

class Tag(Base):
    __tablename__ = 'tags'
    __table_args__ = {
        'comment': '태그',
        # 'mysql_charset': 'utf8mb3',
        # 'mysql_collate': 'utf8mb3_general_ci'
    }        
    if use_mysql_charset:
        __table_args__.update({
            'mysql_charset': mysql_charset,
            'mysql_collate': mysql_collate
        })
    tag_id = Column(Integer, primary_key=True, index=True, comment='태그ID')
    name = Column(String(100), unique=True, nullable=False, comment='태그명')
    background_color = Column(String(100) , comment='배경색 #009988')
    
    create_user = Column(String(255) ,comment='생성자')
    update_user = Column(String(255) ,comment='수정자')
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(KST) , comment='생성일')
    updated_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), onupdate=lambda: datetime.now(KST), comment='수정일')

    prompts = relationship('Prompt', secondary=prompt_tags, back_populates='tags')
    tools = relationship("Tool", secondary=tool_tags, back_populates="tags")
    agents = relationship("Agent", secondary='agent_tags', back_populates="tags")
    datasources = relationship("DataSource", secondary=datasource_tags, back_populates="tags")



class Tool(Base):
    __tablename__ = 'tools'
    tool_id = Column(Integer, primary_key=True, index=True, comment='Tool ID')
    name = Column(String(255), nullable=False, comment='Tool Name')
    description = Column(Text, comment='Tool Description')
    visibility = Column(String(20), nullable=False, default='private', comment='Visibility (private/public)')
    tool_configuration = Column(String(255), nullable=False, default='code', comment='Configuration (code/git)')  # New field
    code = Column(Text, comment='Tool Code')
    git_url = Column(String(255), comment='Git URL')
    git_branch = Column(String(255), comment='Git Branch')
    git_path = Column(String(500), comment='Git Path')

    create_user = Column(String(255) , ForeignKey('users.user_id', ondelete='NO ACTION'),comment='생성자')
    update_user = Column(String(255) ,comment='수정자')
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(KST) , comment='생성일')
    updated_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), onupdate=lambda: datetime.now(KST), comment='수정일')
    

    # Relations
    tags = relationship("Tag", secondary=tool_tags, back_populates="tools")
    conversations = relationship("Conversation", secondary=conversation_tools, back_populates="tools")
    create_user_info = relationship("User", back_populates="tools")
    agents = relationship("Agent", secondary="agent_tools", back_populates="tools")

# Association table for the many-to-many relationship with tags
agent_tags = Table('agent_tags', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agents.agent_id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.tag_id'), primary_key=True)
)

# Association table for the many-to-many relationship with prompts
agent_prompts = Table('agent_prompts', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agents.agent_id'), primary_key=True),
    Column('prompt_id', Integer, ForeignKey('prompts.prompt_id'), primary_key=True)
)

# Association table for the many-to-many relationship with tools
agent_tools = Table('agent_tools', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agents.agent_id'), primary_key=True),
    Column('tool_id', Integer, ForeignKey('tools.tool_id'), primary_key=True)
)

class AgentVariable(Base):
    __tablename__ = 'agent_variables'
    variable_id = Column(Integer, primary_key=True, index=True, comment='Variable ID')
    agent_id = Column(Integer, ForeignKey('agents.agent_id'), nullable=False, comment='Agent ID')
    variable_name = Column(String(255), nullable=False, comment='Variable Name')
    variable_value = Column(Text, nullable=False, comment='Variable Value')

    agent = relationship("Agent", back_populates="variables")

class Agent(Base):
    __tablename__ = "agents"
    agent_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    description = Column(String(255), nullable=True)
    visibility = Column(String(255), nullable=False, default='private')  # private or public
    llm_api_id = Column(Integer, ForeignKey('llm_api.llm_api_id'), nullable=True)
    llm_model = Column(String(255), nullable=False)
    create_user = Column(String(255), ForeignKey('users.user_id'), nullable=False)
    update_user = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(KST))
    updated_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), onupdate=lambda: datetime.now(KST))
    parent_agent_id = Column(Integer, ForeignKey('agents.agent_id'), nullable=True)

    sub_agents = relationship(
        "Agent",
        backref=backref('parent_agent', remote_side=[agent_id]),
        cascade="all, delete-orphan"
    )

    # Relationships to other tables
    llm_api = relationship("LlmApi", back_populates="agents")
    create_user_info = relationship("User", back_populates="agents")
    prompts = relationship("Prompt", secondary=agent_prompts, back_populates="agents")
    tools = relationship("Tool", secondary=agent_tools, back_populates="agents")
    tags = relationship("Tag", secondary=agent_tags, back_populates="agents")
    variables = relationship("AgentVariable", cascade="all, delete-orphan", back_populates="agent")
    conversations = relationship("Conversation", secondary=conversation_agent, back_populates="agents")

# Define possible data source types as an Enum


class DataSource(Base):
    """
    SQLAlchemy model for a data source. This represents a table in the database where each row is a data source.
    """
    __tablename__ = 'datasources'
    __table_args__ = (
        Index('ix_datasources_rawtext', 'raw_text', mysql_prefix='FULLTEXT',mysql_with_parser='ngram'),
        {
            'comment': '데이타소스',  # Table comment
        }
    )

    # Conditionally update the table args based on the environment variable
    if use_mysql_charset:
        __table_args__ = (
            Index('ix_datasources_rawtext', 'raw_text', mysql_prefix='FULLTEXT',mysql_with_parser='ngram'),
            {
            'mysql_charset': mysql_charset,
            'mysql_collate': mysql_collate
        })

    # Unique identifier for each data source, serves as the primary key
    datasource_id = Column(String(255), primary_key=True, index=True, comment='Unique identifier for the data source')
    # Name of the data source, must be unique and indexed for fast look-up
    name = Column(String(255), unique=True, index=True, comment='Name of the data source')
    # Description of the data source, allowing for longer text
    description = Column(Text, comment='Description of the data source')
    # Visibility status of the data source: 'private' or 'public'
    visibility = Column(String(255), nullable=False, default='private', comment='Visibility of the data source: private or public')


    # Type of the data source (e.g., TEXT, GITLAB, etc.)
    datasource_type = Column(String(100), default=DataSourceType.TEXT ,comment='Type of the data source text, pdf_file, confluence, gitlab, url, doc_file, jira')
    
    # Fields specific to GitLab data sources
    namespace = Column(String(255), nullable=True, comment='Namespace for GitLab projects')
    project_name = Column(String(255), nullable=True, comment='Project name in GitLab')
    branch = Column(String(255), nullable=True, comment='Branch name in GitLab')

    # Fields specific to JIRA data sources
    project_key = Column(String(255), nullable=True, comment='Project key for JIRA')
    start = Column(Integer, nullable=True, comment='Start index for JIRA issues')
    limit = Column(Integer, nullable=True, comment='Limit for JIRA issues')
    
    # Fields specific to Confluence data sources
    space_key = Column(String(255), nullable=True, comment='Space key for Confluence')

    # Field specific to document and PDF data sources
    file_path = Column(String(255), nullable=True, comment='File path for document or PDF')

    # Field specific to text data sources
    raw_text = Column(Text, nullable=True, comment='Raw text data')

    # Fields specific to URL data sources
    url = Column(String(255), nullable=True, comment='URL for URL data source')
    base_url = Column(String(255), nullable=True, comment='Base URL for URL data source')
    max_depth = Column(Integer, nullable=True, comment='Maximum depth for URL crawling')



    # Timestamp when the data source was created, automatically set to the current time
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), comment='Timestamp when the data source was created')
    # Timestamp for the last update to the data source, automatically set to the current time
    updated_at = Column(TIMESTAMP, default=lambda: datetime.now(KST), comment='Timestamp for the last update to the data source')
    uploaded_at = Column(TIMESTAMP, nullable=True, comment='Timestamp when the data was uploaded')  # New column added
    # User who created the data source
    create_user = Column(String(255), ForeignKey('users.user_id', ondelete='NO ACTION'), comment='User ID of the creator of the data source')
    # User who last updated the data source
    update_user = Column(String(255), comment='User ID of the last person to update the data source')

    
    # Many-to-many relationship with Conversation
    tags = relationship("Tag", secondary=datasource_tags, back_populates="datasources")
    conversations = relationship(
        "Conversation",
        secondary=conversation_datasource,
        back_populates="datasources"
    )
    # Many-to-one relationship with Embedding
    embeddings = relationship("Embedding", back_populates="datasource" , cascade="all, delete-orphan")
    create_user_info = relationship("User", back_populates="datasources")

class EmbeddingStatus(Enum):
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'

class Embedding(Base):
    """
    SQLAlchemy model for an embedding. Represents an embedding operation related to a data source.
    """
    __tablename__ = 'embeddings'
    __table_args__ = {
        'comment': 'Embeddings related to data sources'
    }
        # Conditionally update the table args based on the environment variable
    if use_mysql_charset:
        __table_args__.update({
            'mysql_charset': mysql_charset,
            'mysql_collate': mysql_collate
        })

    embedding_id = Column(String(255), primary_key=True, index=True, comment='Unique identifier for the embedding')
    datasource_id = Column(String(255), ForeignKey('datasources.datasource_id'), primary_key=True, nullable=False, comment='Foreign key to the data source')
    llm_api_id = Column(Integer, ForeignKey('llm_api.llm_api_id',ondelete='NO ACTION'), nullable=False, comment='LLM API used for embedding')
    embedding_model = Column(String(255), nullable=False, comment='Embedding model used')
    splitter = Column(String(255), nullable=True, comment='Splitter used for preprocessing. RecursiveCharacterTextSplitter,')

    # Fields related to RecursiveCharacterTextSplitter
    language = Column(String(255), nullable=True, default='PLAIN_TEXT', comment='Language for RecursiveCharacterTextSplitter. PLAIN_TEXT(default), CPP, GO, JAVA, KOTLIN, JS, TS, PYTHON, SCALA, MARKDOWN, HTML')
    
    # Fields related to RecursiveCharacterTextSplitter , CharacterTextSplitter
    chunk_size = Column(Integer, nullable=True, comment='Chunk size for RecursiveCharacterTextSplitter, CharacterTextSplitter')
    chunk_overlap = Column(Integer, nullable=True, default=1 ,comment='Chunk overlap for RecursiveCharacterTextSplitter, CharacterTextSplitter')
    
    """
    Fields related to HTMLHeaderTextSplitter, HTMLSectionSplitter, and MarkdownHeaderTextSplitter
    HTMLHeaderTextSplitter, HTMLSectionSplitter : h1(default) , h2, h3
    MarkdownHeaderTextSplitter : #(default),##,###
    """
    tag = Column(String(255), nullable=True, comment='Tag for HTMLHeaderTextSplitter, HTMLSectionSplitter, MarkdownHeaderTextSplitter')

    # Fields related to CharacterTextSplitter
    separator = Column(String(255), nullable=True, comment='Separator for CharacterTextSplitter')
    is_separator_regex = Column(Boolean, nullable=True, comment='Whether to use regex for separator in CharacterTextSplitter')

    # Field related to RecursiveJsonSplitter
    max_chunk_size = Column(Integer, nullable=True, comment='Max chunk size for RecursiveJsonSplitter')


    status = Column(String(255), nullable=False, default=EmbeddingStatus.PENDING, comment='Status of the embedding process  pending,in_progress,completed,failed')
    data_size = Column(Integer, nullable=True, comment="Size of the data in kilobytes")
    started_at = Column(TIMESTAMP, nullable=True, comment='Timestamp when the embedding started')
    completed_at = Column(TIMESTAMP, nullable=True, comment='Timestamp when the embedding completed')
    success_at = Column(TIMESTAMP, nullable=True, comment='Timestamp when the embedding succeeded')
    
    last_update_time = Column(TIMESTAMP, default=lambda: datetime.now(KST), comment='Last time this embedding was updated')

    # Relationship back to DataSource
    datasource = relationship("DataSource", back_populates="embeddings")    
    llm_api = relationship("LlmApi", back_populates="embeddings")
    


# Create tables without foreign keys in the database
# # Create tables
Base.metadata.create_all(bind=engine)

# Relation with orderby
User.conversation = relationship("Conversation", order_by=Conversation.conversation_id, back_populates="user_info")
Conversation.messages = relationship("Message", order_by=Message.message_id, back_populates="conversation" , cascade="all, delete-orphan")
# Conversation.states = relationship("ChatbotState", order_by=ChatbotState.state_id, back_populates="conversation")
CommonCodeGroup.commonCode = relationship("CommonCodeDetail", order_by=CommonCodeDetail.sort_order, back_populates="codeGroup")
Conversation.prompts = relationship(
    "Prompt", 
    secondary=conversation_prompt, 
    back_populates="conversations"
    , order_by=conversation_prompt.c.sort_order
)

Conversation.tools = relationship(
    "Tool", 
    secondary=conversation_tools, 
    back_populates="conversations"
    # , order_by=conversation_prompt.c.sort_order
)

# Conversation.llm_api = relationship("LlmApi", back_populates="conversation")