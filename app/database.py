import asyncio
import enum
from uuid import UUID
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from sqlalchemy import Boolean, Float, UniqueConstraint, create_engine ,Table, Column, Integer, String, Text, Enum, ForeignKey, TIMESTAMP , event, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker , backref
from urllib.parse import quote
from datetime import datetime, timedelta
from enum import Enum as PyEnum
import os
import app.config
from typing import Dict, ForwardRef, Optional
from ai_core.checkpoint.mysql_saver import MySQLSaver
from ai_core.data_source.base import DataSourceType
import pytz
from sqlalchemy.dialects.mysql import JSON

import app.endpoint

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
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600  # 1시간마다 연결을 재활용, MySQL 기본 타임아웃 8시간
    # , echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
db_comment_endpoint = text("/* is_endpoint_query */ 1=1")

# Check environment variable for charset setting
use_mysql_charset = os.getenv('USE_MYSQL_CHARSET', 'False') == 'True'
mysql_charset = os.getenv('MYSQL_CHARSET', 'utf8mb4')
mysql_collate = os.getenv('MYSQL_COLLATE', 'utf8mb4_general_ci')

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
        
def get_db_async():
    db = SessionLocal()
    try:
        yield db
    finally:
        pass
        # db.commit()
        # db.close()
        
class CustomSession(Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_endpoint_query = False

def checkpointer_setup(connection_url):
    with PyMySQLSaver.from_conn_string(connection_url) as checkpointer: # type: PyMySQLSaver
        checkpointer.setup()
     
def create_sync_connection_pool():
    global sync_conn_pool
    if not sync_conn_pool:
        username = os.getenv('history_connection_username')
        password = os.getenv('history_connection_password')
        host = os.getenv('history_connection_host')
        port = os.getenv('history_connection_port')
        database_name = os.getenv('history_connection_database')
        environment = os.getenv('ENVIRONMENT')
        
        encoded_password = quote(password)
        connection_url = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database_name}"
        connection_url_checkpoint = f"mysql://{username}:{encoded_password}@{host}:{port}/{database_name}"
        
        # if environment == 'development':
        #     connection_url_checkpoint = "mysql://root:Sktelecom1!@localhost:3306/daisy"
        #     checkpointer_setup(connection_url_checkpoint)
        # elif environment == 'product':
        #     connection_url_checkpoint = "mysql://daisy:Skdaisy3!@150.6.13.242:3306/daisy"
        #     checkpointer_setup(connection_url_checkpoint)
        # else:
        #     connection_url_checkpoint = "mysql://root:dataservice123!@#@localhost:3306/daisy"
        #     # connection_url_checkpoint = f"mysql://{username}:{encoded_password}@{host}:{port}/{database_name}"
        #     connection_url_checkpoint = f"mysql://{username}:{encoded_password}@{host}:{port}/{database_name}"
        #     checkpointer_setup(connection_url_checkpoint)
        
        checkpointer_setup(connection_url_checkpoint)  
        
        sync_conn_pool = MySQLSaver.create_sync_connection_pool(
            host=host,
            user=username,
            password=password,
            db=database_name,
            port=int(port),
            autocommit=True
        )
        # MySQLSaver.create_tables(sync_conn_pool)
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
        
        # Test a query
        # async with async_conn_pool.acquire() as aconnection:
        #     async with aconnection.cursor() as a_cursor:
        #         await a_cursor.execute("SELECT VERSION()")
        #         version = await a_cursor.fetchone()
        #         print("Database version:", version[0])
                
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


