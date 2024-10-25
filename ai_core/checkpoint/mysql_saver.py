import aiomysql
import pymysql
from dbutils.pooled_db import PooledDB
from aiomysql import Pool


class MySQLSaver:
    @staticmethod
    def create_sync_connection_pool(host, port, user, password, db, autocommit, maxconnections=10) -> PooledDB:
        return PooledDB(
            creator=pymysql,
            host=host,
            port=port,
            user=user,
            password=password,
            db=db,
            autocommit=autocommit,
            maxconnections=maxconnections,
        )

    @staticmethod
    async def create_async_connection_pool(host, port, user, password, db, autocommit, maxsize=10) -> Pool:
        return await aiomysql.pool._create_pool(
            host=host,
            port=port,
            user=user,
            password=password,
            db=db,
            autocommit=autocommit,
            maxsize=maxsize,
        )
