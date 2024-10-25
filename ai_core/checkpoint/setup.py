from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

DB_URI = "mysql://root:wkdwjdgh7!@localhost:3306/daisy"

with PyMySQLSaver.from_conn_string(DB_URI) as checkpointer: # type: PyMySQLSaver
    checkpointer.setup()
