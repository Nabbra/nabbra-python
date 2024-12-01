import os
import sys

from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .ws import ws

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "DEBUG"))


app = FastAPI(
    title=os.getenv("APP_NAME", "Nabbra"),
    openapi_tags=[{"name": "Authentication", "description": "API protected by Bearer tokens"}]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/stream", ws)

app.include_router(api_router, prefix="/api")

@app.get("/", response_class=JSONResponse)
def home():
    raise HTTPException(status_code=404, detail="Page not found")
