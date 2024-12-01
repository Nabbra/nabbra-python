import os
import modal
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

        call = modal_app.launch_yolo_model.spawn(audiogram_image)

        return JSONResponse({
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

@router.get("/reader/result/{call_id}", response_class=JSONResponse)
async def audiogram_results(call_id: str):
    function_call = modal.functions.FunctionCall.from_id(call_id)
    try:
        right_ear_result, left_ear_result = function_call.get(timeout=0)
    except TimeoutError:
        return JSONResponse(content="", status_code=202)

    return JSONResponse({
        "success": True,
        "right_ear": right_ear_result.to_dict(),
        "left_ear": left_ear_result.to_dict(),
    })