import os
import numpy as np

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from app.common.factory import ServiceFactory

router = APIRouter(prefix="/audiogram")


@router.post("/read", response_class=JSONResponse)
async def read(request: Request):
    form = await request.form()
    audiogram_image = await form["audiogram"].read()
    token = request.query_params.get("token")

    if not token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

    if os.getenv("MODAL_ENV"):
        logger.debug("Spawning AI model on Modal")

        try:
            modal_app = __import__("modal_app", fromlist=["launch_yolo_modal"])
        except ImportError:
            logger.error("Failed to import launch_yolo_modal from modal_app")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to import launch_yolo_modal from modal_app"
            )

        call = modal_app.launch_yolo_model.spawn(audiogram_image, request.query_params.get("token"))

        return JSONResponse({
            "success": True,
            "call_id": call.object_id,
        })

    reader = ServiceFactory.get_service(
        "audiogram_reader",
        {
            "box_model_path": "./models/yolov8_box.pt",
            "symbol_model_path": "./models/yolov8_symbol.pt"
        }
    )

    right_ear_result, left_ear_result = reader.feature_extraction(np.frombuffer(audiogram_image, dtype=np.float32))

    return JSONResponse({
        "status": True,
        "right_ear": right_ear_result.to_dict(),
        "left_ear": left_ear_result.to_dict()
    })
