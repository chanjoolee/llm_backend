import re
import traceback
from fastapi import FastAPI ,Request, HTTPException, status
# from .sample import router as sample_router
# from .sample_langchain import router as sample_langchain_router
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pymysql
# from .endpoint.docs import router as docs_router
from .endpoint.utils import router as utils_router
from .endpoint.commonCode import router as cmmcode_router
from .endpoint.users import router as user_router
from .endpoint.usersAlert import router as userAlert_router
from .endpoint.messages import router as message_router
from .endpoint.messages_system import router as message_system_router
from .endpoint.conversations import router as conversation_router
from .endpoint.agent import router as agent_router
# from .endpoint.chatbotConfig import router as config_router
# from .endpoint.chatbotState import router as state_router
from .endpoint.login import router as login_router , CheckSessionMiddleware, session_backend , cookie
from .endpoint.llmApi import router as llm_api_router
from .endpoint.tags import router as tags_router
from .endpoint.prompt import router as prompt_router
from .endpoint.tools import router as tool_router
from .endpoint.datasource import router as datasource_router
from .endpoint.dashboard import router as dashbord_router
from .endpoint.sendMail import router as sendmail_route
from .endpoint import realtime_updates
from .endpoint.hotel.hotel import router as hotel_router
# from .endpoint.conversationPrompt import router as conversationPrompt_router
from app import database
from app.model import model_llm
from app.schema import schema_llm
# from .utils.session_utils import get_session 
from uuid import UUID, uuid4
from starlette.middleware.base import BaseHTTPMiddleware 
from typing import List
from fastapi_sessions.backends.implementations import InMemoryBackend
# from .endpoint import langchainAgent as agent
# from langserve import add_routes
# from dotenv import load_dotenv
import os
import app.config
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import logging
import sqlparse
from fastapi.staticfiles import StaticFiles
from ai_core.checkpoint.mysql_saver import MySQLSaver
from sqlalchemy import Engine, event
from app.logging_config import root_logger

from app.model import model_hotel




logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI(docs_url=None)
# app = FastAPI()

# Serve static files from the 'static' directory within the 'app' folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# database.Base.metadata.create_all(engine)
# model_llm.Base.metadata.create_all(database.engine)
model_hotel.Base.metadata.create_all(database.engine)


logger_sql = None

# Function to format SQL statements
def format_sql_statement(statement):
    formatted_statement = sqlparse.format(statement, reindent=True, keyword_case='upper')
    return formatted_statement

class FilteredHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if msg:  # Only log if msg is not empty
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)

class SQLFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.statement_before = ""

    def format(self, record):
        if record.msg.startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE')):
            record.msg = format_sql_statement(record.msg)
            self.statement_before = record.msg
            return None

        # return super().format(record)
        else: 
            return super().format(record)
            # if "/* is_endpoint_query */" in self.statement_before  : 
            #     return super().format(record)
            # else :
            #     return None
        
    

def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    if isinstance(parameters, list):
        for param_set in parameters:
            statement = bind_params(statement, param_set)
    else:
        statement = bind_params(statement, parameters)

    formatted_sql = format_sql_statement(statement)
    if formatted_sql.lower().startswith("select") :
        if "/* is_endpoint_query */" in formatted_sql  : 
            logger.info(f"Executing SQL:\n{formatted_sql}")
    else :
        logger.info(f"Executing SQL:\n{formatted_sql}")
    
    
    
    return statement, parameters

def bind_params(statement, parameters):
    def replace(match):
        key = match.group(1)
        return repr(parameters[key]) if key in parameters else match.group(0)
    
    return re.sub(r'%\((\w+)\)s', replace, statement)


# Custom event listener to log result counts
def after_execute(conn, clauseelement, multiparams, params, result):
    if clauseelement.__class__.__name__ == 'Select':
        count = result.rowcount
        if "/* is_endpoint_query */" in result.cursor._executed  : 
            logger.info(f"Result count: {count}\n")


# Custom logging formatter for SQL statements


@app.on_event("startup")
async def startup_event():
    global  logger_sql
    
    logger_sql = logging.getLogger('sqlalchemy.engine')
    """
    에러가 난 경우에만 로그를 찍는다. 
    sql log 는 before_cursor_execute,after_execute 에서 logging 을 이용한다.
    개발의 편의성을 위해 파라메터에서 바인딩된 후의 로그를 찍는다.
    예) users.user_id = 'chanjoo'  
    """
    logger_sql.setLevel(logging.WARNING)

    # Apply the custom SQLFormatter
    # handler = logging.StreamHandler()
    handler = FilteredHandler()
    handler.setFormatter(SQLFormatter())
    # below  logged twice
    logger_sql.addHandler(handler)

    # Attach the event listener to the engine
    from app.database import engine  # Assuming your engine is imported from database module
    event.listen(engine, "after_execute", after_execute)
    event.listen(engine, "before_cursor_execute", before_cursor_execute)


    database.create_sync_connection_pool()
    await database.create_async_connection_pool()
    

@app.on_event("shutdown")
async def shutdown_event():
    if database.sync_conn_pool:
        database.sync_conn_pool.close()
    if database.async_conn_pool:
        database.async_conn_pool.close()


# # Create tables
# Base.metadata.create_all(
#     bind=engine, 
#     tables=[
#         User.__table__, 
#         Conversation.__table__,
#         Message.__table__,
#         ChatbotConfig.__table__,
#         ChatbotState.__table__
#     ]
# )

# Add CORS middleware if needed
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8081",
    "http://localhost:3000",
    "http://myapp.tde.sktelecom.com",
    "http://myapp.tde.sktelecom.com:8080",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8081",
    "http://150.6.13.90",
    "http://150.6.13.90:8080",
    "http://150.6.13.90:3000",
    "http://your-frontend-domain.com",
    "http://tat-02-10",
    "http://tat-02-10:8080",
    "http://tat-02-10:3000",    
    # product1 : daisy backend, daisy frontend, mysql, opensearch, opensearch-dashboards
    "http://dtl-xu-01",
    "http://dtl-xu-01:8080",
    "http://dtl-xu-01:8081",
    "http://dtl-xu-01:3000",
    "http://150.6.15.81",
    "http://150.6.15.81:8080",
    "http://150.6.15.81:8081",
    "http://150.6.15.81:3000",
    # product2 : mysql, opensearch
    "http://dtl-xu-02",
    "http://dtl-xu-02:8080",
    "http://dtl-xu-02:8081",
    "http://dtl-xu-02:3000"     
    "http://150.6.15.105",
    "http://150.6.15.105:8080",
    "http://150.6.15.105:8081",
    "http://150.6.15.105:3000",   
    
    # Add more origins as needed
    # 김건우
    "http://172.23.76.156:8080"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

exempt_paths = [
    "/api/create_session", 
    "/api/delete_session", 
    "/api/whoami" , 
    "/docs", 
    "/openapi.json",
    "/favicon.ico"
]
# 뭔가 잘못된듯.
# 당분간 endpoint별로 적용한다.
# app.add_middleware(CheckSessionMiddleware, backend=session_backend, cookie=cookie, exempt_paths=exempt_paths)


# Middleware to add session to request state

# Secret key used for signing cookies
SECRET_KEY = "DONOTUSE"
serializer = URLSafeTimedSerializer(SECRET_KEY,"cookie_daisy")

def check_sync_connection_pool(pool):
    try:
        conn = pool.connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Sync connection pool check failed: {e}")
        return False
    
async def check_async_connection_pool(pool):
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT 1")
                await cursor.fetchall()
        return True
    except Exception as e:
        logging.error(f"Async connection pool check failed: {e}")
        return False

@app.middleware("http")
async def add_session_to_request(request: Request, call_next):
    session_cookie = request.cookies.get("cookie_daisy")
    if session_cookie:
        try:
            # session_id = UUID(session_cookie)
            session_id_str = serializer.loads(session_cookie)
            session_id = UUID(session_id_str)
            session_data = await session_backend.read(session_id)
            request.state.session = session_data
        except SignatureExpired as e:
            print("SignatureExpired error:", e)
            request.state.session = None
        except BadSignature as e:
            print("BadSignature error:", e)
            request.state.session = None
        except Exception as e:
            print("General error:", e)
            request.state.session = None
            
    else:
        request.state.session = None
    
    try:
        return await call_next(request)
    except Exception as e:
        
        trace_str = traceback.format_exc()
        response_normal = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": f"Unhandled error: {str(e)}",
                "traceback": trace_str.split("\n")
            }
        )

        # async 로 call 하는 경우 이런식으로 return 한다.
        response_async =  JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "detail": {
                    "error_code" : status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": f"Unhandled error: {str(e)}"
                },
                "traceback": trace_str.split("\n")
            }
        )
        
        response_message = response_normal
        if request.url.path in ['/api/messages','/api/messages_stream'] and request.method in ['POST','post']:
            response_message = response_normal
        else:
            response_message = response_normal

        response_message.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", origins)
        response_message.headers["Access-Control-Allow-Credentials"] = "true"
        response_message.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response_message.headers["Access-Control-Allow-Headers"] = "Content-Type"

        logger.error(f"Unhandled error: {str(e)}")
        logger.error(trace_str)
        return response_message

# app.include_router(sample_router, prefix="/api/sample")
# app.include_router(sample_langchain_router, prefix="/api")
app.include_router(utils_router, prefix="/api", tags=["Common Utils"])
app.include_router(cmmcode_router, prefix="/api")
app.include_router(user_router, prefix="/api",tags=["Users"])   
app.include_router(userAlert_router, prefix="/api/user/alert",tags=["Users Alert 시스템요청"])
app.include_router(message_router, prefix="/api")
app.include_router(message_system_router, prefix="/api")
app.include_router(conversation_router, prefix="/api")
app.include_router(agent_router, prefix="/api/agent")
# app.include_router(config_router, prefix="/api")
# app.include_router(state_router, prefix="/api")
app.include_router(login_router, prefix="/api")
app.include_router(llm_api_router, prefix="/api/llm_api", tags=["LLM APIs"])
app.include_router(tags_router, prefix="/api/tags", tags=["Tags"])
app.include_router(prompt_router, prefix="/api" ,tags=["Prompts"])
app.include_router(tool_router, prefix="/api" ,tags=["Tools"])
app.include_router(datasource_router, prefix="/api/datasource" ,tags=["Data Source"])
app.include_router(dashbord_router, prefix="/api/dashboard" ,tags=["Dashboard"])
app.include_router(sendmail_route, prefix="/api/sendmail" ,tags=["Send Mail"])
app.include_router(realtime_updates.router)
app.include_router(hotel_router)

# app.include_router(docs_router,tags=["Docs"])
# app.include_router(conversationPrompt_router, prefix="/api")
# add_routes(
#     app,
#     agent.agent_executor.with_types(input_type=agent.Input, output_type=agent.Output),
#     path="/api/langchain/agent/test",
    
# )

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    logger.info("Custom Swagger UI endpoint hit")
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )
@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(title=app.title, version=app.version, routes=app.routes)


# Optional: Add root endpoint for basic API check
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/test")
async def test_endpoint():
    logger.info("Test endpoint hit")
    print("Test endpoint hit")
    return {"message": "Test endpoint is working"}