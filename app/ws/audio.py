import base64

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.common.factory import ServiceFactory

from .connection_manager import ConnectionManager
from loguru import logger

manager = ConnectionManager()
amplifier = ServiceFactory.get_service("amplifier")


async def amplify_endpoint(websocket: WebSocket):
    if not await manager.connect(websocket):
        return

    sample_rate = int(websocket.query_params.get("sample_rate", "8000"))

    amplifier.set_sample_rate(sample_rate)
    amplifier.set_ranges({
        (250, 500): 10,
        (700, 1000): 6
    })
    amplifier.to_linear_gain()

    logger.debug(f"Openning a websocket connection to amplify audio at sample rate {sample_rate}")

    try:
        while True:
            data = await websocket.receive_json()

            audio_buffer = base64.b64decode(data.get("audio"))

            audio = amplifier.apply_gain_from_bytes(audio_buffer).tobytes()
            audio_base64 = base64.b64encode(audio).decode('utf-8')

            await websocket.send_json({"ear": data.get("ear"), "audio": audio_base64})
    except WebSocketDisconnect:
        manager.disconnect(websocket)