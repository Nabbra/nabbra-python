from fastapi import APIRouter
from .audiogram import router as audiogram_router

router = APIRouter()

router.include_router(audiogram_router, tags=["Audiogram Reader"])