from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from typing import Set
import asyncio
import json
from json import JSONEncoder
from datetime import datetime
from pydantic import SecretStr

router = APIRouter()
connected_clients: Set[WebSocket] = set()



@router.websocket("/ws/data-updates")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while websocket.application_state == WebSocketState.CONNECTED:
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

class DateTimeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # or str(obj) if you prefer a simpler format
        return super().default(obj)
class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO format string
        elif isinstance(obj, SecretStr):
            # return obj.get_secret_value()  # Retrieve the underlying string value of SecretStr
            return "****"
        return super().default(obj)
    
async def broadcast_update(message: dict):
    for client in connected_clients:
        if client.application_state == WebSocketState.CONNECTED:
            await client.send_text(json.dumps(message,cls=CustomJSONEncoder))
