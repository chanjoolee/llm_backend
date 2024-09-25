"""Implementation of a langgraph checkpoint saver using MySQL."""
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Optional,
    Union,
    Tuple,
    List,
    Sequence,
)

import aiomysql
import pymysql
from dbutils.pooled_db import PooledDB
from aiomysql import Pool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple


class JsonAndBinarySerializer(JsonPlusSerializer):
    def _default(self, obj):
        if isinstance(obj, (bytes, bytearray)):
            return self._encode_constructor_args(
                obj.__class__, method="fromhex", args=[obj.hex()]
            )
        return super()._default(obj)

    def dumps(self, obj: Any) -> tuple[str, bytes]:
        if isinstance(obj, bytes):
            return "bytes", obj
        elif isinstance(obj, bytearray):
            return "bytearray", obj

        return "json", super().dumps(obj)

    def loads(self, s: tuple[str, bytes]) -> Any:
        if s[0] == "bytes":
            return s[1]
        elif s[0] == "bytearray":
            return bytearray(s[1])
        elif s[0] == "json":
            return super().loads(s[1])
        else:
            raise NotImplementedError(f"Unknown serialization type: {s[0]}")


@contextmanager
def _get_sync_connection(
        connection_pool: PooledDB,
) -> Generator[pymysql.Connection, None, None]:
    """Get the connection to the MySQL database."""
    with connection_pool.connection() as connection:
        yield connection


@asynccontextmanager
async def _get_async_connection(
        connection_pool: Union[Pool, None],
) -> AsyncGenerator[aiomysql.Connection, None]:
    """Get the connection to the MySQL database."""
    async with connection_pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 42;")
            (r,) = await cur.fetchone()
            assert r == 42
            yield conn
    # if isinstance(connection, aiomysql.Connection):
    #     yield connection
    # else:
    #     raise ValueError(
    #         "Invalid async connection object. Please initialize the check pointer "
    #         f"with an appropriate async connection object. "
    #         f"Got {type(connection)}."
    #     )


class MySQLSaver(BaseCheckpointSaver):
    sync_connection_pool: PooledDB
    """The synchronous connection to the MySQL database.
    
    Please ensure that the connection is open
    and remember to close the connection when done.
    """
    async_connection_pool: Pool
    """The asynchronous connection to the MySQL database.
    
    Please ensure that the connection is open
    and remember to close the connection when done.
    """

    def __init__(
            self,
            sync_connection_pool: PooledDB,
            async_connection_pool: Pool,
    ):
        super().__init__(serde=JsonPlusSerializer())
        self.sync_connection_pool = sync_connection_pool
        self.async_connection_pool = async_connection_pool

    @contextmanager
    def _get_sync_connection(self) -> Generator[pymysql.Connection, None, None]:
        """Get the connection to the MySQL database."""
        with _get_sync_connection(self.sync_connection_pool) as connection:
            yield connection

    @asynccontextmanager
    async def _get_async_connection(
            self,
    ) -> AsyncGenerator[aiomysql.Connection, None]:
        """Get the connection to the MySQL database."""
        async with _get_async_connection(self.async_connection_pool) as connection:
            yield connection

    CREATE_TABLES_QUERY = """
    CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id VARCHAR(255) NOT NULL,
        thread_ts VARCHAR(255) NOT NULL,
        parent_ts VARCHAR(255),
        checkpoint LONGBLOB NOT NULL,
        metadata LONGBLOB NOT NULL,
        PRIMARY KEY (thread_id, thread_ts)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    
    CREATE TABLE IF NOT EXISTS writes (
        thread_id VARCHAR(255) NOT NULL,
        thread_ts VARCHAR(255) NOT NULL,
        task_id VARCHAR(255) NOT NULL,
        idx INT NOT NULL,
        channel VARCHAR(255) NOT NULL,
        value LONGBLOB,
        PRIMARY KEY (thread_id, thread_ts, task_id, idx)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

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

    @staticmethod
    def create_tables(sync_connection_pool: PooledDB, /) -> None:
        """Create the schema for the checkpoint saver."""
        with _get_sync_connection(sync_connection_pool) as conn:
            with conn.cursor() as cur:
                for query in MySQLSaver.CREATE_TABLES_QUERY.split(';'):
                    if query.strip():
                        cur.execute(query)
            conn.commit()

    @staticmethod
    def drop_tables(sync_connection_pool: PooledDB, /) -> None:
        """Drop the table for the checkpoint saver."""
        with _get_sync_connection(sync_connection_pool) as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS checkpoints, writes;")
            conn.commit()

    UPSERT_CHECKPOINT_QUERY = """
    INSERT INTO checkpoints 
        (thread_id, thread_ts, parent_ts, checkpoint, metadata)
    VALUES 
        (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        checkpoint = VALUES(checkpoint),
        metadata = VALUES(metadata);
    """

    def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Put the checkpoint for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
            checkpoint: The checkpoint to persist.
        Returns:
            The RunnableConfig that describes the checkpoint that was just created.
            It'll contain the `thread_id` and `thread_ts` of the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        parent_ts = config["configurable"].get("thread_ts")
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    self.UPSERT_CHECKPOINT_QUERY,
                    (
                        thread_id,
                        checkpoint["id"],
                        parent_ts if parent_ts else None,
                        self.serde.dumps(checkpoint),
                        self.serde.dumps(metadata),
                    ),
                )
            # conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": checkpoint["id"],
            },
        }

    async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Put the checkpoint for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
            checkpoint: The checkpoint to persist.
        Returns:
            The RunnableConfig that describes the checkpoint that was just created.
            It'll contain the `thread_id` and `thread_ts` of the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        parent_ts = config["configurable"].get("thread_ts")
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    self.UPSERT_CHECKPOINT_QUERY,
                    (
                        thread_id,
                        checkpoint["id"],
                        parent_ts if parent_ts else None,
                        self.serde.dumps(checkpoint),
                        self.serde.dumps(metadata),
                    ),
                )
            # await conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": checkpoint["id"],
            },
        }

    UPSERT_WRITES_QUERY = """
    INSERT INTO writes
        (thread_id, thread_ts, task_id, idx, channel, value)
    VALUES
        (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        value = VALUES(value);
    """

    def put_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[Tuple[str, Any]],
            task_id: str,
    ) -> None:
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    self.UPSERT_WRITES_QUERY,
                    [
                        (
                            str(config["configurable"]["thread_id"]),
                            str(config["configurable"]["thread_ts"]),
                            task_id,
                            idx,
                            channel,
                            self.serde.dumps(value),
                        )
                        for idx, (channel, value) in enumerate(writes)
                    ],
                )
            # conn.commit()

    async def aput_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[Tuple[str, Any]],
            task_id: str,
    ) -> None:
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    self.UPSERT_WRITES_QUERY,
                    [
                        (
                            str(config["configurable"]["thread_id"]),
                            str(config["configurable"]["thread_ts"]),
                            task_id,
                            idx,
                            channel,
                            self.serde.dumps(value),
                        )
                        for idx, (channel, value) in enumerate(writes)
                    ],
                )
            # await conn.commit()

    LIST_CHECKPOINTS_QUERY_STR = """
    SELECT checkpoint, metadata, thread_ts, parent_ts
    FROM checkpoints
    {where}
    ORDER BY thread_ts DESC
    """

    def list(
            self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
    ) -> Generator[CheckpointTuple, None, None]:
        """Get all the checkpoints for the given configuration."""
        where, args = self._search_where(config, filter, before)
        query = self.LIST_CHECKPOINTS_QUERY_STR.format(where=where)
        if limit:
            query += f" LIMIT {limit}"
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                thread_id = config["configurable"]["thread_id"]
                cur.execute(query, tuple(args))
                for value in cur:
                    checkpoint, metadata, thread_ts, parent_ts = value
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        },
                        checkpoint=self.serde.loads(checkpoint),
                        metadata=self.serde.loads(metadata),
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        }
                        if parent_ts
                        else None,
                    )

    async def alist(
            self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Get all the checkpoints for the given configuration."""
        where, args = self._search_where(config, filter, before)
        query = self.LIST_CHECKPOINTS_QUERY_STR.format(where=where)
        if limit:
            query += f" LIMIT {limit}"
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                thread_id = config["configurable"]["thread_id"]
                await cur.execute(query, tuple(args))
                async for value in cur:
                    checkpoint, metadata, thread_ts, parent_ts = value
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        },
                        checkpoint=self.serde.loads(checkpoint),
                        metadata=self.serde.loads(metadata),
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        }
                        if parent_ts
                        else None,
                    )

    GET_CHECKPOINT_BY_TS_QUERY = """
    SELECT checkpoint, metadata, thread_ts, parent_ts
    FROM checkpoints
    WHERE thread_id = %s AND thread_ts = %s
    """

    GET_CHECKPOINT_QUERY = """
    SELECT checkpoint, metadata, thread_ts, parent_ts
    FROM checkpoints
    WHERE thread_id = %s
    ORDER BY thread_ts DESC LIMIT 1
    """

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get the checkpoint tuple for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
        Returns:
            The checkpoint tuple for the given configuration if it exists,
            otherwise None.
            If thread_ts is None, the latest checkpoint is returned if it exists.
        """
        thread_id = config["configurable"]["thread_id"]
        thread_ts = config["configurable"].get("thread_ts")
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                # find the latest checkpoint for the thread_id
                if thread_ts:
                    cur.execute(
                        self.GET_CHECKPOINT_BY_TS_QUERY,
                        (thread_id, thread_ts),
                    )
                else:
                    cur.execute(
                        self.GET_CHECKPOINT_QUERY,
                        (thread_id,),
                    )

                # if a checkpoint is found, return it
                if value := cur.fetchone():
                    checkpoint, metadata, thread_ts, parent_ts = value
                    if not config["configurable"].get("thread_ts"):
                        config = {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        }

                    # find any pending writes
                    cur.execute(
                        "SELECT task_id, channel, value FROM writes WHERE thread_id = %s AND thread_ts = %s",
                        (thread_id, thread_ts),
                    )
                    # deserialize the checkpoint and metadata
                    return CheckpointTuple(
                        config=config,
                        checkpoint=self.serde.loads(checkpoint),
                        metadata=self.serde.loads(metadata),
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": parent_ts,
                            }
                        }
                        if parent_ts
                        else None,
                        pending_writes=[
                            (task_id, channel, self.serde.loads(value))
                            for task_id, channel, value in cur
                        ],
                    )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get the checkpoint tuple for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
        Returns:
            The checkpoint tuple for the given configuration if it exists,
            otherwise None.
            If thread_ts is None, the latest checkpoint is returned if it exists.
        """
        thread_id = config["configurable"]["thread_id"]
        thread_ts = config["configurable"].get("thread_ts")
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                # find the latest checkpoint for the thread_id
                if thread_ts:
                    await cur.execute(
                        self.GET_CHECKPOINT_BY_TS_QUERY,
                        (thread_id, thread_ts),
                    )
                else:
                    await cur.execute(
                        self.GET_CHECKPOINT_QUERY,
                        (thread_id,),
                    )
                # if a checkpoint is found, return it
                if value := await cur.fetchone():
                    checkpoint, metadata, thread_ts, parent_ts = value
                    if not config["configurable"].get("thread_ts"):
                        config = {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": thread_ts,
                            }
                        }

                    # find any pending writes
                    await cur.execute(
                        "SELECT task_id, channel, value FROM writes WHERE thread_id = %s AND thread_ts = %s",
                        (thread_id, thread_ts),
                    )
                    # deserialize the checkpoint and metadata
                    return CheckpointTuple(
                        config=config,
                        checkpoint=self.serde.loads(checkpoint),
                        metadata=self.serde.loads(metadata),
                        parent_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": parent_ts,
                            }
                        }
                        if parent_ts
                        else None,
                        pending_writes=[
                            (task_id, channel, self.serde.loads(value))
                            async for task_id, channel, value in cur
                        ],
                    )

    def _search_where(
            self,
            config: Optional[RunnableConfig],
            filter: Optional[dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
    ) -> Tuple[str, List[Any]]:
        """Return WHERE clause predicates for given config, filter, and before parameters.
        Args:
            config (Optional[RunnableConfig]): The config to use for filtering.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria.
            before (Optional[RunnableConfig]): A config to limit results before a certain timestamp.
        Returns:
            Tuple[str, Sequence[Any]]: A tuple containing the WHERE clause and parameter values.
        """
        wheres = []
        param_values = []

        # Add predicate for config
        if config is not None:
            wheres.append("thread_id = %s")
            param_values.append(config["configurable"]["thread_id"])

        if filter:
            raise NotImplementedError()

        # Add predicate for limiting results before a certain timestamp
        if before is not None:
            wheres.append("thread_ts < %s")
            param_values.append(before["configurable"]["thread_ts"])

        where_clause = "WHERE " + " AND ".join(wheres) if wheres else ""
        return where_clause, param_values

    DELETE_CHECKPOINTS_QUERY = "DELETE FROM checkpoints WHERE thread_id = %s"
    DELETE_WRITES_QUERY = "DELETE FROM writes WHERE thread_id = %s"

    def delete_thread(self, thread_id: str) -> None:
        """Delete all data associated with a specific thread_id."""
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(self.DELETE_CHECKPOINTS_QUERY, (thread_id,))
                cur.execute(self.DELETE_WRITES_QUERY, (thread_id,))
            conn.commit()

    async def adelete_thread(self, thread_id: str) -> None:
        """Asynchronously delete all data associated with a specific thread_id."""
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(self.DELETE_CHECKPOINTS_QUERY, (thread_id,))
                await cur.execute(self.DELETE_WRITES_QUERY, (thread_id,))
            await conn.commit()

    CLONE_CHECKPOINTS_QUERY = """
    INSERT INTO checkpoints (thread_id, thread_ts, parent_ts, checkpoint, metadata)
    SELECT %s, thread_ts, parent_ts, checkpoint, metadata
    FROM checkpoints
    WHERE thread_id = %s
    """

    CLONE_WRITES_QUERY = """
    INSERT INTO writes (thread_id, thread_ts, task_id, idx, channel, value)
    SELECT %s, thread_ts, task_id, idx, channel, value
    FROM writes
    WHERE thread_id = %s
    """

    def clone_thread(self, source_thread_id: str, new_thread_id: str) -> None:
        """Clone all data associated with a specific thread_id to a new thread_id."""
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(self.CLONE_CHECKPOINTS_QUERY, (new_thread_id, source_thread_id))
                cur.execute(self.CLONE_WRITES_QUERY, (new_thread_id, source_thread_id))
            conn.commit()

    async def aclone_thread(self, source_thread_id: str, new_thread_id: str) -> None:
        """Asynchronously clone all data associated with a specific thread_id to a new thread_id."""
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(self.CLONE_CHECKPOINTS_QUERY, (new_thread_id, source_thread_id))
                await cur.execute(self.CLONE_WRITES_QUERY, (new_thread_id, source_thread_id))
            await conn.commit()
