from app.llm.schema_agent import get_sqlalchemy_chain

async def process_request_with_llm(user_input: str):
    chain = get_sqlalchemy_chain()
    return chain(user_input)