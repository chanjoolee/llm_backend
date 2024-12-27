from fastapi import File, UploadFile
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, TIMESTAMP
from enum import Enum
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

from ai_core.data_source.base import DataSourceType
from ai_core.data_source.splitter.base import SplitterType
from .database import Base , UserRoll , KST
from pydantic import BaseModel , SecretStr , EmailStr, Field, computed_field, field_validator, root_validator, validator, model_validator
from typing import Annotated, Optional ,List,ForwardRef

# For lang chain  refer:  https://chatgpt.com/c/a1b04731-1218-4336-b466-8e67743f0ed1

Message = ForwardRef('Message')
Prompt = ForwardRef('Prompt')
PromptCreate = ForwardRef('PromptCreate')
LlmApi = ForwardRef('LlmApi')
Tag = ForwardRef('Tag')
PromptVariableValue = ForwardRef('PromptVariableValue')
ConversatonPromptCreate = ForwardRef('ConversatonPromptCreate')
# ConversationVariable = ForwardRef('ConversationVariable')
# Embedding = ForwardRef('Embedding')

class Visibility(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"
class MediaType(str, Enum):
    TEXT = "text/plain"
    JSON = "application/json"
class CommonCodeGroupBase(BaseModel):
    code_group: str
    group_desc: Optional[str] = None

class CommonCodeGroupCreate(CommonCodeGroupBase):
    pass

class CommonCodeGroup(CommonCodeGroupBase):
    create_user: Optional[str] = None
    update_user: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes: True

class CommonCodeDetailBase(BaseModel):
    code_group: str
    code_value: str
    code_desc: Optional[str] = None
    sort_order: Optional[int] = None
    category1: Optional[str] = None

class CommonCodeDetailCreate(CommonCodeDetailBase):
    pass

class CommonCodeDetail(CommonCodeDetailBase):
    create_user: Optional[str] = None
    update_user: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes: True

class UserBase(BaseModel):
    user_id: str
    nickname: str
    password: SecretStr
class UserCreate(UserBase):
    pass


class User(UserBase):
    user_roll : Optional[str]
    created_at: datetime
    updated_at: datetime
    token_gitlab: Optional[str]
    token_confluence: Optional[str] 
    token_jira: Optional[str] 
    llm_api_id : Optional[int] = Field(None,description="LLM API ID " )
    llm_model : Optional[str] =  Field(None, description="LLM Model. LLM API에서 입력된 값중 하나만 입력")

    class Config:
        from_attributes = True

class UserRoleUpdateBase(BaseModel):
    user_id: str
    user_roll: str

class UserRoleUpdates(BaseModel):
    user_list: Optional[List[UserRoleUpdateBase]]

class UserUpdate(BaseModel):
    nickname: Optional[str] = Field(None, description="The user's nickname") 
    password: Optional[SecretStr] = Field(None, description="The user's password")
    user_roll: Optional[str] = Field(None, description="The role of the user")

class UserLogin(BaseModel):
    user_id: str
    password: SecretStr

    class Config:
        from_attributes = True
class UserSearch(BaseModel):
    user_search_field: Optional[str] =''
    search_words: Optional[str] =''
    user_roll: Optional[str] = ''
    skip: Optional[int] = 0
    limit: Optional[int] = 10

class UserUpdateToken(BaseModel):
    token_gitlab: str = Field(None, description="Gitlab Access Token")
    token_confluence: str = Field(None, description="Confluence Access Token")
    token_jira: str = Field(None, description="Jira Access Token")
class UserAlertCreate(BaseModel):
    user_id: str
    llm_api_id : Optional[int] = Field(None,description="LLM API ID " )
    llm_model : Optional[str] =  Field('gpt-4o', description="LLM Model. LLM API에서 입력된 값중 하나만 입력")

class UserAlertUpdate(BaseModel):
    llm_api_id : Optional[int] = Field(None,description="LLM API ID " )
    llm_model : Optional[str] =  Field('gpt-4o', description="LLM Model. LLM API에서 입력된 값중 하나만 입력")
    
# 대화검색
class ConversationSearch(BaseModel):
    search_range_list: Optional[List[str]] = ["message"] # multi-select 검색범위 제목 title, 메세지 message
    search_words: Optional[str] = ""
    started_at: Optional[datetime] = None
    last_conversation_time: Optional[datetime] = None
    input_path_list : Optional[List[str]] = Field([],description="메세지 입력경로: prompt,conversation,system")
    user_roll_list : Optional[List[str]] = Field([],description="생성자롤: MANAGER,ALERT,GUEST,OWNER,SYSTEM")
    conversation_type_list : Optional[List[str]] = [] # multi-select for 공개 여부 ['private','public']
    component_list: Optional[List[object]] = [] # multi-select for 컴포넌트
    user_list: Optional[List[str]] = [] # multi-select for 사용자
    llm_api_list : Optional[List[int]] = []
    llm_model_list: Optional[List[str]] = []# multi-select for 텍스트 생성 모델 gpt-4o
    datasource_list : Optional[List[str]] = []
    skip: Optional[int] = 0
    limit: Optional[int] = 10


def default_conversation_title():
    return lambda: datetime.now(KST).strftime("Conversation_%Y%m%d_%H%M%S")

class ConversationBase(BaseModel):
    conversation_title: str = "Conversation_"
    conversation_type : str ='private' # public
    llm_api_id : Optional[int] = Field(None,description="LLM API ID " )
    llm_model : str =  Field('gpt-4o', description="LLM Model. LLM API에서 입력된 값중 하나만 입력")
    temperature: float = 0.5
    max_tokens : int = 1024
    component_configuration: str = Field('none', description="Component Configuration none,component,agent,all")  # New field
    


class ConversationCreate(ConversationBase):
    prompts: Optional[List[ConversatonPromptCreate]] = None
    tools: Optional[List[int]] = None  # List of tool IDs
    agents: Optional[List[int]] = None  # List of agent IDs
    datasources: Optional[List[str]] = Field(None, description="List of datasources IDs")
    

class ConversatonPromptCreate(BaseModel):
    prompt_id: int
    variables: Optional[List[PromptVariableValue]] = Field(default_factory=list)

class ConversationUpdate(BaseModel):
    user_id: Optional[str] = Field("chanjoo", description="사용자아이디")
    conversation_title: Optional[str] = Field("conversation", description="대화제목")
    conversation_type: Optional[str] = Field("private", description="대화종류 private public")
    llm_api_id: Optional[int] = Field(None, description="대화종류 private public")
    llm_model : Optional[str] =  Field('gpt-4o', description="LLM Model. LLM API에서 입력된 값중 하나만 입력")
    temperature: Optional[float] = Field(0.5, description="")
    max_tokens: Optional[int] = Field(0, description="최대토큰")
    input_token : Optional[int] = Field(0, description="input토큰")
    output_token : Optional[int] = Field(0, description="output토큰")
    used_tokens: Optional[int] = Field(0, description="토큰사용량")
    last_conversation_time: Optional[datetime] = Field(None, description="마지막 대화시간")
    last_message_id: Optional[int] = Field(None, description="메세지ID")  # Changed from Integer to int
    started_at: Optional[datetime] = Field(None, description="생성일")
    component_configuration: Optional[str] = Field('none', description="Component Configuration none,component,agent,all")  # New field
    prompts: Optional[List[ConversatonPromptCreate]] = None
    tools: Optional[List[int]] = Field(None, description="List of tools IDs")
    agents: Optional[List[int]] = Field(None, description="List of agent IDs")
    datasources: Optional[List[str]] = Field(None, description="List of datasources IDs")


class ConversationCopy(BaseModel):
    conversation_id_origin : str 

class ConversationGenerateTitle(BaseModel):
    conversation_id : str 
class ConversationVariable(BaseModel):
    variable_id : int 
    conversation_id : str
    variable_name : str
    variable_value : str

    class Config:
        from_attributes = True
class Conversation(ConversationBase):
    conversation_id: str
    user_id: str 
    input_token : Optional[int] = Field(0, description="input토큰")
    output_token : Optional[int] = Field(0, description="output토큰")
    used_tokens : Optional[int] = Field(0, description="총사용토큰")
    last_conversation_time : Optional[datetime]
    last_message_id: Optional[int]
    started_at: Optional[datetime]

    #Relations
    messages: List[Message] = []
    prompts: List["Prompt"] = []
    tools: List["Tool"] = []
    llm_api : Optional[LlmApi] = []
    user_info : Optional[User] = None
    variables: Optional[List[ConversationVariable]] = []
    agents : Optional[List["Agent"]] = []
    datasources : Optional[List["Datasource"]] = []

    class Config:
        from_attributes = True



class SearchConversationsResponse(BaseModel):
    totalCount: int
    list: List[Conversation]
    
class SessionStoreBase(BaseModel):
    id: int
    

class SessionStoreCreate(SessionStoreBase):
    pass

class SessionStore(SessionStoreBase):
    session_id :str 
    message: str 

class MessageBase(BaseModel):
    # message_id: int
    conversation_id: str
    message : str
    input_path : str = Field("conversation", description="입력경로 prompt,conversation,system")

class MessageCreate(MessageBase):
    pass

class SystemMessageCreate(BaseModel):
    message: str

class Message(MessageBase):
    message_id: int
    message_type: str = "human"
    sent_at: Optional[datetime]
    input_token: Optional[int] = Field(0, description="input 토큰사용량")
    output_token: Optional[int] = Field(0, description="output 토큰사용량")
    total_token: Optional[int] = Field(0, description="토탈토큰사용량")
    # conversation : Optional[Conversation]

    class Config:
        from_attributes = True


class LlmApiBase(BaseModel):
    
    llm_api_name : str = 'sk-'
    llm_api_type : str = 'private'
    llm_api_provider : str = 'openai'
    llm_api_url : str = 'https://aihub-api.sktelecom.com/aihub/v1/sandbox'
    llm_api_key : str = 'ba3954fe-9cbb-4599-966b-20b04b5d3441'
    llm_model : str = 'gpt-4o'
    embedding_model : str = ''
    

class LlmApiCreate(LlmApiBase):
    pass

class LlmApiSearch(BaseModel):
    search_field: Optional[str] = Field('')
    search_words: Optional[str] = Field('')
    llm_api_type : Optional[str] = Field('private',description="private,public")

    skip: Optional[int] = 0
    limit: Optional[int] = 10

class LlmApiDelete(BaseModel):
    llm_api_ids : List[int]

    skip: Optional[int] = 0
    limit: Optional[int] = 10

class LlmApiUpdate(BaseModel):

    llm_api_name : Optional[str] = 'sk-'
    llm_api_type : Optional[str] = 'private'    
    llm_api_provider : Optional[str] = 'openai'
    llm_api_url : Optional[str] = 'https://aihub-api.sktelecom.com/aihub/v1/sandbox'
    llm_model : Optional[str] = 'gpt-4o'
    embedding_model : Optional[str] =""
    llm_api_key : Optional[str] = 'ba3954fe-9cbb-4599-966b-20b04b5d3441'

class LlmApi(LlmApiBase):
    llm_api_id : Optional[int] 
    create_user: Optional[str] 
    update_user: Optional[str] 
    created_at: Optional[datetime] 
    updated_at: Optional[datetime]
    create_user_info : Optional[User]
    
    class Config:
        from_attributes = True

class SearchLlmApiResponse(BaseModel):
    totalCount: int
    list: List[LlmApi]
    
class PromptMessageBase(BaseModel):
    message_type: str = Field(default="system,ai,human" ,description="메세지타입 system,ai,human")
    message: str

class PromptMessageCreate(PromptMessageBase):
    pass

class PromptMessageUpdate(PromptMessageBase):
    # message_id: Optional[int] = None
    pass

class PromptMessage(PromptMessageBase):
    message_id: int
    prompt_id: int
    create_user: Optional[str] = None
    update_user: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
class PromptVariableValue(BaseModel):
    variable_name: str
    value: Optional[str]

class PromptVariableCrate(BaseModel):
    variable_name: str

class PromptVariable(PromptVariableCrate):
    variable_id: int
    class Config:
        from_attributes = True
class PromptBase(BaseModel):
    prompt_title: str
    prompt_desc: str
    open_type: str = 'private'

class PromptCreate(PromptBase):
    tag_ids: List[int] 
    promptMessages: Optional[List[PromptMessageCreate]] 

class PromptUpdate(BaseModel):
    prompt_title: Optional[str] = None
    prompt_desc: Optional[str] = None
    open_type: Optional[str] = 'private'
    tag_ids: Optional[List[int]] = []
    promptMessages: Optional[List[PromptMessageUpdate]] = None

class Prompt(PromptBase):
    prompt_id: int
    create_user: Optional[str] = None
    create_user_info : Optional[User]
    update_user: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    
    tags: Optional[List[Tag]]
    promptMessage: Optional[List[PromptMessage]]
    variables: Optional[List[PromptVariable]]  # Add this line
    # conversations : Optional[List[Conversation]]
    # conversations: Optional[List["Conversation"]] = []  # Use forward reference

    class Config:
        from_attributes = True
 
class SearchPromptResponse(BaseModel):
    totalCount: int
    list: List[Prompt]

class PromptSearch(BaseModel):
    search_words: Optional[str] = ""
    search_scope: Optional[List[str]] = ['message']  # e.g., ["name", "desc", "message"]
    open_type: Optional[List[str]] = []  # e.g., "public", "private"
    tag_ids: Optional[List[int]] = [1]    
    user: Optional[List[str]] = []
    skip: Optional[int] = 0
    limit: Optional[int] = 10


class TagBase(BaseModel):
    name : str
    background_color : Optional[str] = ''

class TagCreate(TagBase):
    pass

class TagUpdate(TagBase):
    pass 

class Tag(TagBase):
    tag_id : int 
    create_user: Optional[str] = None
    update_user: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True    

class TagSearch(BaseModel):
    search_words: Optional[str] = ''

    skip: Optional[int] = 0
    limit: Optional[int] = 10

class TagDelete(BaseModel):
    tag_ids : List[int]

    skip: Optional[int] = 0
    limit: Optional[int] = 10

class SearchTagResponse(BaseModel):
    totalCount: int
    list: List[Tag]

class ToolBase(BaseModel):
    name: str
    description: Optional[str] = None
    visibility: str = 'private'
    tool_configuration: str = 'code'
    code: Optional[str] = ""
    git_url: Optional[str] = ""
    git_branch: Optional[str] = ""
    git_path: Optional[str] = ""
    

class ToolCreate(ToolBase):
    tag_ids : Optional[List[int]] = []

class ToolUpdate(ToolBase):
    tag_ids : Optional[List[int]] = []

class Tool(ToolBase):
    tool_id: int
    create_user: str
    update_user: str
    created_at: datetime
    updated_at: datetime

    tags: List[Tag] = []
    # create_user_info 추가함.
    create_user_info : Optional[User] = None

    class Config:
        from_attributes = True

class ConversationToolLink(BaseModel):
    conversation_id: str
    tool_id: int

    class Config:
        from_attributes = True

class ToolSearch(BaseModel):
    search_words: Optional[str] = ""
    search_scope: Optional[List[str]] = ["name", "description"]
    visibility: Optional[List[str]] = []
    tag_ids: Optional[List[int]] = []
    user_list: Optional[List[str]] = []
    skip: Optional[int] = 0
    limit: Optional[int] = 10

class SearchToolsResponse(BaseModel):
    totalCount: int
    list: List[Tool]



# 에이젼트
class AgentBase(BaseModel):
    name: str
    description: Optional[str] = None
    visibility: str = 'private'  # private or public
    llm_api_id: int
    llm_model: str
    prompts: Optional[List[ConversatonPromptCreate]] = None
    
    
class AgentCreate(AgentBase):
    tools: Optional[List[int]] = None  # List of tool IDs
    tags: Optional[List[int]] = None  # List of tag IDs
    sub_agents: Optional[List[int]] = None
    datasources : Optional[List[str]] = Field([],description="List of datasources id")

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    visibility: Optional[str] = 'private'  # private or public
    llm_api_id: Optional[int] = None
    llm_model: Optional[str] = None
    prompts: Optional[List[ConversatonPromptCreate]] = None
    tools: Optional[List[int]] = None  # List of tool IDs
    tags: Optional[List[int]] = None  # List of tag IDs
    sub_agents: Optional[List[int]] = None
    datasources : Optional[List[str]] = Field([],description="List of datasources id")

class AgentVariable(BaseModel):
    variable_id: int
    agent_id: int
    variable_name: str
    variable_value: str

    class Config:
        from_attributes = True
class Agent(AgentBase):
    agent_id: int
    create_user: str
    update_user: Optional[str]
    created_at: datetime
    updated_at: datetime

    create_user_info : Optional[User]
    llm_api : Optional[LlmApi] = []
    prompts: Optional[List["Prompt"]] = []
    tools: Optional[List["Tool"]] = []
    tags: Optional[List[Tag]] = []
    variables: Optional[List[AgentVariable]] = None
    sub_agents: Optional[List["Agent"]] = None
    datasources : Optional[List["Datasource"]] = Field([],description="List of datasources")

    class Config:
        from_attributes = True



class AgentSearch(BaseModel):
    search_words: Optional[str] = Field(None,description="검색어")
    search_range_list: Optional[List[str]] = None  # e.g., ['name', 'description']
    visibility_list: Optional[List[str]] = None  # e.g., ['public', 'private']
    # component_list: Optional[List[int]] = None  # e.g., list of component IDs
    tag_list: Optional[List[int]] = None  # e.g., list of tag IDs
    user_list: Optional[List[str]] = None  # e.g., list of user IDs
    llm_api_list: Optional[List[int]] = None  # e.g., list of LLM API IDs
    
    llm_model_list: Optional[List[str]] = None  # e.g., list of LLM models
    prompt_list : Optional[List[int]] = []
    tool_list : Optional[List[int]] = []
    datasource_list : Optional[List[str]] = []
    component_list: Optional[List[object]] = [] # multi-select for 컴포넌트
    
    skip: Optional[int] = 0
    limit: Optional[int] = 10

class SearchAgentsResponse(BaseModel):
    totalCount: int
    list: List[Agent]


class DatasourceBase(BaseModel):
    name: str = Field(..., description="Name of the data source")
    description: str = Field(..., description="Description of the data source")
    visibility: str = Field("private", description="Visibility of the data source: private or public")
    datasource_type: Optional[str] = Field("text", description="Type of the data source (e.g., text, pdf_file, confluence, gitlab, url, doc_file, jira)")
    namespace: Optional[str] = Field(None, description="Namespace for GitLab projects")
    project_name: Optional[str] = Field(None, description="Project name in GitLab")
    branch: Optional[str] = Field(None, description="Branch name in GitLab")
    project_key: Optional[str] = Field(None, description="Project key for JIRA")
    start: Optional[int] = Field(None, description="Start index for JIRA issues")
    limit: Optional[int] = Field(None, description="Limit for JIRA issues")
    token: Optional[str] = Field(None, description="API Token for JIRA")
    
    space_key: Optional[str] = Field(None, description="Space key for Confluence")
    
    raw_text: Optional[str] = Field(None, description="Raw text data")
    url: Optional[str] = Field(None, description="URL for URL data source")
    base_url: Optional[str] = Field(None, description="Base URL for URL data source")
    max_depth: Optional[int] = Field(None, description="Maximum depth for URL crawling")


class DatasourceCreate(DatasourceBase):
    tag_ids : Optional[List[int]] = []
    # file: Annotated[UploadFile | None, File(description="File path for document or PDF")] = None

class Datasource(DatasourceBase):
    datasource_id: str = Field(..., description="Unique identifier for the data source")
    file_path: Optional[str] = Field(None, description="File path for document or PDF")
    
    status: Optional[str] = Field(None, description="Status of the datasource download process  downloaded,downloading,error,cancelled")
    downloaded_at: Optional[datetime] = Field(None, description="Timestamp when the datasource data downloaded")
    
    created_at: datetime = Field(..., description="Timestamp when the data source was created")
    updated_at: datetime = Field(..., description="Timestamp for the last update to the data source")
    uploaded_at: Optional[datetime] = Field(None, description="Timestamp when the data was uploaded")
    create_user: Optional[str] = Field(None, description="User ID of the creator of the data source")
    update_user: Optional[str] = Field(None, description="User ID of the last person to update the data source")

    tags: List[Tag] = []
    # embeddings
    embeddings : List["Embedding"] = []
    create_user_info : Optional[User]
    class Config:
        from_attributes = True


class DataSourceSearch(BaseModel):
    search_words: Optional[str] = Field("",description="Keywords to search")  # Keywords to search
    search_scope: Optional[List[str]] = ["name", "description"]  # Search scope (name, description)
    visibility: Optional[List[Visibility]] = Field([], description="Visibility of the data source: private or public")
    tag_ids: Optional[List[int]] = []  # Tag IDs for filtering
    user_list: Optional[List[str]] = []  # Users associated with the data source
    datasource_types: Optional[List[DataSourceType]] = Field([], 
        description="""
            Type of the data source 
            (e.g., text, pdf_file, confluence, gitlab, url, doc_file, jira)
        """
        )
    skip: Optional[int] = 0  # Pagination skip
    limit: Optional[int] = 10  # Pagination limit

class DataSourceSearchResponse(BaseModel):
    total_count: int  # Total number of results
    list: List[Datasource]  # List of matching data sources


class EmbeddingBase(BaseModel):
    datasource_id: str = Field(..., description="Data source ID for which embedding is created")
    llm_api_id: Optional[int] = Field(None, description="LLM API ID")
    embedding_model: Optional[str] = Field('text-embedding-3-small', description="Embedding model used")
    splitter: Optional[SplitterType] = Field(None, description="Splitter type used for preprocessing")

    # Fields related to chunking
    chunk_size: Optional[int] = Field(None, description="Chunk size for text splitting")
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap for text splitting")

    # Language specific for RecursiveCharacterTextSplitter
    language: Optional[str] = Field("PLAIN_TEXT", 
        description="""
        Language for RecursiveCharacterTextSplitter.
        PLAIN_TEXT(default)

        class Language(str, Enum):
            CPP = "cpp"
            GO = "go"
            JAVA = "java"
            KOTLIN = "kotlin"
            JS = "js"
            TS = "ts"
            PHP = "php"
            PROTO = "proto"
            PYTHON = "python"
            RST = "rst"
            RUBY = "ruby"
            RUST = "rust"
            SCALA = "scala"
            SWIFT = "swift"
            MARKDOWN = "markdown"
            LATEX = "latex"
            HTML = "html"
            SOL = "sol"
            CSHARP = "csharp"
            COBOL = "cobol"
            C = "c"
            LUA = "lua"
            PERL = "perl"
            HASKELL = "haskell"

        """
    )
    
    # Fields related to header-based splitting
    tag: Optional[str] = Field(
        None, 
        description="""
        Tag for HTMLHeaderTextSplitter, HTMLSectionSplitter, MarkdownHeaderTextSplitter
        HTMLHeaderTextSplitter  : h1(default), h2, h3
        HTMLSectionSplitter     : h1(default), h2, h3
        MarkdownHeaderTextSplitter : 값 : #(default), ##, ###
        """
    )

    # Fields related to CharacterTextSplitter
    separator: Optional[str] = Field(None, description="Separator for CharacterTextSplitter")
    is_separator_regex: Optional[bool] = Field(None, description="If True, treat separator as a regex pattern for CharacterTextSplitter")

    # Field related to RecursiveJsonSplitter
    max_chunk_size: Optional[int] = Field(None, description="Max chunk size for RecursiveJsonSplitter")


    @field_validator("llm_api_id", mode='before')
    def validate_llm_api_id(cls, value):
        if value is not None and value <= 0:
            raise ValueError("LLM API ID must be a positive integer")
        return value

    # class Config:
    #     from_attributes = True

# Pydantic model for creating a new embedding
class EmbeddingCreate(EmbeddingBase):

    # Validator for chunk_size and chunk_overlap
    @field_validator("chunk_size", "chunk_overlap", mode='before')
    def validate_chunk_fields(cls, value, values):
        splitter = values.data.get('splitter')

        if splitter in [SplitterType.RecursiveCharacterTextSplitter, SplitterType.CharacterTextSplitter]:
            if cls.__name__ == "chunk_size" and (value is None or value <= 0):
                raise ValueError("Chunk size must be provided and greater than 0 for RecursiveCharacterTextSplitter, CharacterTextSplitter")
            if cls.__name__ == "chunk_overlap" and (value is None or value < 0):
                raise ValueError("Chunk overlap must be provided and must be 0 or greater for RecursiveCharacterTextSplitter, CharacterTextSplitter")
        
        return value

    # Validator for separator and separator_regex
    @field_validator("separator", "is_separator_regex", mode='before')
    def validate_separator_fields(cls, value, values):
        splitter = values.data.get('splitter')

        if splitter == SplitterType.CharacterTextSplitter:
            if cls.__name__ == "separator" and not value:
                raise ValueError("Separator must be provided for CharacterTextSplitter.")
            if cls.__name__ == "is_separator_regex" and value is None:
                raise ValueError("separator_regex must be provided for CharacterTextSplitter.")

        return value

    # Validator for tag
    @field_validator("tag", mode='before')
    def validate_tag_field(cls, value, values):
        splitter = values.data.get('splitter')

        if splitter in [SplitterType.HTMLHeaderTextSplitter, SplitterType.HTMLSectionSplitter, SplitterType.MarkdownHeaderTextSplitter]:
            if not value:
                raise ValueError("Tag must be provided for HTMLHeaderTextSplitter,HTMLSectionSplitter,MarkdownHeaderTextSplitter")
        
        return value
    
    # Validator for language
    @field_validator("language", mode='before')
    def validate_language_field(cls, value, values):
        splitter = values.data.get('splitter')

        if splitter in [SplitterType.RecursiveCharacterTextSplitter]:
            if not value:
                raise ValueError("language must be provided for RecursiveCharacterTextSplitter")
        
        return value

    # Validator for max_chunk_size
    @field_validator("max_chunk_size", mode='before')
    def validate_max_chunk_size_field(cls, value, values):
        splitter = values.data.get('splitter')

        if splitter == SplitterType.RecursiveJsonSplitter:
            if value is None or value <= 0:
                raise ValueError("Max chunk size must be provided and greater than 0 for RecursiveJsonSplitter.")
        
        return value
    
# Pydantic model for updating an embedding
class EmbeddingUpdate(EmbeddingCreate):
    datasource_id: Optional[str] = Field(None, description="Data source ID for which embedding is created")
    status_message: Optional[str] = Field(None, description="message when there is error")
    

class Embedding(EmbeddingBase):
    embedding_id: str = Field(..., description="Unique identifier for the embedding")
    # datasource_id: str = Field(..., description="Data source ID for which embedding is created")
    status: str = Field(..., description="Status of the embedding process (e.g., )")
    status_message: Optional[str] = Field(None, description="message when there is error")
    data_size: Optional[int] = Field(0, description="Size of the data in kilobytes")
    started_at: Optional[datetime] = Field(None, description="Timestamp when the embedding started")
    completed_at: Optional[datetime] = Field(None, description="Timestamp when the embedding was completed")
    success_at: Optional[datetime] = Field(None, description="Time when embedding succeeded")
    last_update_time: Optional[datetime] = Field(None, description="Timestamp when the embedding was last updated")

    llm_api : Optional[LlmApi] = None
    
    @computed_field
    @property
    def data_size_readable(self) -> str:
        """Converts data_size in bytes to human-readable format."""
        size = self.data_size or 0
        if size < 1024:
            return f"{size} Bytes"
        elif size < 1024 ** 2:
            return f"{size / 1024:.2f} KB"
        elif size < 1024 ** 3:
            return f"{size / 1024 ** 2:.2f} MB"
        elif size < 1024 ** 4:
            return f"{size / 1024 ** 3:.2f} GB"
        elif size < 1024 ** 5:
            return f"{size / 1024 ** 4:.2f} TB"
        else:
            return f"{size / 1024 ** 5:.2f} PB"
        
    class Config:
        from_attributes = True



class EmbeddingSearch(BaseModel):
    datasource_id: str = Field(..., description="Data source ID for which embedding is created")
    skip: int = 0  # Pagination skip
    limit: int = 10  # Pagination limit

class EmbeddingSearchResponse(BaseModel):
    total_count: int  # Total number of results
    list: List[Embedding]  # List of matching embeddings
    
    class Config:
        from_attributes = True
    
class EmbeddingPreview(EmbeddingCreate):
    splitter: SplitterType = Field(..., description="Splitter type used for preprocessing")
    
class EmbeddingSimilaritySearch(BaseModel):
    embedding_id: str = Field(..., description="Unique identifier for the embedding")
    query: str = Field(...,description="query for collection")
    
    
    
    
# Define a Pydantic model for the request body
class GroupedTokenRequest(BaseModel):
    llm_api_list: Optional[List[int]] = []
    llm_model_list: Optional[List[str]] = []
    start_date: Optional[str] = None
    end_date: Optional[str] = None    
    
class EmailRequest(BaseModel):
    sender_email: EmailStr = Field(default="no-reply.daisy@sktelecom.com", description="")
    receiver_email: EmailStr = Field(default="", description="")
    subject: str  = Field(default="Test Email", description="")
    content: str  = Field(default="This is a test email using SKT's internal SMTP service.", description="")