import modal
import aiohttp

from app.common.endpoints import AUDIOGRAM_SAVE

from dotenv import load_dotenv
from app.server import app as webapp
from loguru import logger

app = modal.App(name="nabbra-ai")

env_vars = load_dotenv()

image = (
    modal.Image.debian_slim(python_version="3.12")
        .apt_install("python3-opencv")
        .pip_install_from_requirements("app/requirements.txt")
)

@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv(), modal.Secret.from_dict({"MODAL_ENV": "1"})],
)
@modal.asgi_app()
def api():
    return webapp


@app.function(
    cpu=1.0,
    image=image,
    mounts=[modal.Mount.from_local_dir("./models", remote_path="/root/models")],
    keep_warm=1,
    enable_memory_snapshot=True,
    max_inputs=1,
    retries=0,
)
async def launch_yolo_model(image: bytes, token: str):
    import io
    import numpy as np

    from PIL import Image
    from app.common.factory import ServiceFactory

    reader = ServiceFactory.get_service(
        "audiogram_reader",
        {
            "box_model_path": "./models/yolov8_box.pt",
            "symbol_model_path": "./models/yolov8_symbol.pt"
        }
    )

    input_image = Image.open(io.BytesIO(image))
    input_image = np.array(input_image)

    air_right_ear_audiogram, air_left_ear_audiogram = reader.feature_extraction(input_image)

    payload = {
        "right_ear": air_right_ear_audiogram.to_dict(),
        "left_ear": air_left_ear_audiogram.to_dict(),
    }

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url=AUDIOGRAM_SAVE, json=payload, headers=headers) as response:
                if response.status != 200:
                    logger.debug(f"Failed to send data. Status code: {response.status}")
                logger.debug("Data sent successfully:", await response.json())
        except aiohttp.ClientError as e:
            print(f"Request failed: {e}")

    return air_right_ear_audiogram, air_left_ear_audiogram
