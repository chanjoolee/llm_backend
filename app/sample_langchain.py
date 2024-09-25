from typing import Dict , List , Any
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
# from langchain.tools import SentenceSplitter, WordTokenizer
from fastapi import APIRouter


# Initialize FastAPI app
# app = FastAPI()
router = APIRouter()

# Setup LangChain with OpenAI's LLM
llm = ChatOpenAI(openai_api_key='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')
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])
chain = prompt | llm 

@router.get("/langchain_simple/", response_model=Dict , tags=["Sample Lang Chain"])
async def langchain_simple(query: str):
    # response = await chain.run(query)
    # response = await llm.complete(prompt=query, max_tokens=50)
    
    # return {"query": query, "answer": response}

    response = await chain.invoke({"input": query})
    return {"query": query, "answer": response}

# # This will serve the API and allow you to make requests to it.
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
