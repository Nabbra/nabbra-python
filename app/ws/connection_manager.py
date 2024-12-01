import aiohttp

from app.common.endpoints import Endpoints

from typing import Dict
from collections import defaultdict
from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    def __init__(self):
        self._connections = defaultdict()

    async def connect(self, websocket: WebSocket) -> bool:
        await websocket.accept()

        token = self._get_token(websocket)

        if not await self._is_valid_token(token):
            await self._close(websocket, reason="Invalid token")
            return False

        connection = self._get_connection_by_token(token)

        if connection is not None:
            await self._close(websocket, reason=f"Connection for token '{token}' already exists for user {token}")
            return False

        self._add_connection(websocket, token)

        return True

    async def _close(self, websocket: WebSocket, reason: str = None) -> None:
        self._remove_connection(self._get_token(websocket))
        await websocket.close(code=1008, reason=reason)

    def disconnect(self, websocket: WebSocket):
        self._remove_connection(self._get_token(websocket))

    async def _is_valid_token(self, token: str | None) -> bool:
        if token is None:
            return False

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {token}"}
            try:
                async with session.post(url=Endpoints.TOKEN_VALIDATOR, headers=headers) as response:
                    if response.status_code == 200:
                        return True
            except:
                return False
        return False

    def _get_existing_connections(self, token: str) -> Dict[str, WebSocket]:
        return self._connections[token]

    def _add_connection(self, websocket: WebSocket, token: str) -> None:
        logger.debug(f"Adding a connection for token {token}")
        self._connections[token] = websocket

    def _remove_connection(self, token: str) -> None:
        if token in self._connections:
            del self._connections[token]

    def _get_token(self, websocket: WebSocket):
        return websocket.query_params.get("token", "")

    def _get_connection_by_token(self, token: str) -> WebSocket:
        return self._connections.get(token)
