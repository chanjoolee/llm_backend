from fastapi import APIRouter, Request
from app.llm.process_request import process_request_with_llm

router = APIRouter(prefix="/api/hotel", tags=["Hotel"])

@router.post("/query")
async def generic_ai_query(request: Request):
    payload = await request.json()
    user_input = payload.get("content")
    if not user_input:
        return {"error": "Missing 'content' in request body."}
    response = await process_request_with_llm(user_input)
    return {"response": response}

