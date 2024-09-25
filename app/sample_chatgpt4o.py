from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import List

# Define a Pydantic model for the input data
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# Initialize the FastAPI app
app = FastAPI()

# Initialize the LangChain components
openai_api_key = "your_openai_api_key_here"  # Replace with your OpenAI API key
llm = ChatOpenAI(api_key=openai_api_key)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Define the endpoint to handle chat requests
@app.post("/chat")
async def chat(chat_request: ChatRequest):
    try:
        # Extract the messages from the request
        messages = chat_request.messages

        # Add messages to LangChain memory
        for message in messages:
            if message.role == "user":
                memory.add_user_message(message.content)
            elif message.role == "ai":
                memory.add_ai_message(message.content)

        # Get the AI's response
        response = conversation.predict(input=memory.get_conversation())

        # Return the response
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
