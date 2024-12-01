import base64
import json
import websockets
import asyncio
import sounddevice as sd
import numpy as np

SAMPLE_SIZE = 16000
BLOCK_SIZE = 2048
CHANNELS = 1

AUTH_TOKEN = "dd"
WS_SERVER_URI = f"wss://abdallamohammed--nabbra-ai-api-dev.modal.run/stream/audio/amplify?token={AUTH_TOKEN}"


async def stream_audio():
    loop = asyncio.get_running_loop()

    async with websockets.connect(WS_SERVER_URI) as websocket:
        def callback(indata, outdata, frames, time, status):
            if status:
                print(f"Status: {status}")
            try:
                audio_base64 = base64.b64encode(indata.tobytes()).decode("utf-8")

                message = json.dumps({
                    "ear": "left",  # Or "right", depending on your setup
                    "audio": audio_base64
                })

                asyncio.run_coroutine_threadsafe(websocket.send(message), loop)

                response = asyncio.run_coroutine_threadsafe(websocket.recv(), loop).result()

                response_json = json.loads(response)

                amplified_audio_base64 = response_json["audio"]
                amplified_audio = base64.b64decode(amplified_audio_base64)

                outdata[:] = np.frombuffer(amplified_audio, dtype=np.float32).reshape(-1, 1)
            except Exception as e:
                print(f"Error in callback: {e}")

        with sd.Stream(
            samplerate=SAMPLE_SIZE,
            channels=CHANNELS,
            dtype="float32",
            callback=callback
        ):
            print("Streaming audio. Press Ctrl+C to stop.")
            await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(stream_audio())