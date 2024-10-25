from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from typing import Set
import asyncio
import json

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

async def broadcast_update(message: dict):
    for client in connected_clients:
        if client.application_state == WebSocketState.CONNECTED:
            await client.send_text(json.dumps(message))
