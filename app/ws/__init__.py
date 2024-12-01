from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .audio import amplify_endpoint

ws = FastAPI()

ws.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ws.add_websocket_route(path="/audio/amplify", route=amplify_endpoint)