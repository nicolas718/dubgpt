from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import requests
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

@app.get("/")
def read_root():
    return {"message": "DubGPT backend is live."}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # Save uploaded video to temp directory
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Extract audio as .mp3
    audio_path = file_location.rsplit(".", 1)[0] + ".mp3"
    try:
        video = VideoFileClip(file_location)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Audio extraction failed: {str(e)}"})

    # Send audio to ElevenLabs
    try:
        with open(audio_path, "rb") as audio_file:
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
            }
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers=headers,
                files={"audio": audio_file},
                data={"model_id": "whisper-large-v3"}
            )
            transcript = response.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Transcription failed: {str(e)}"})

    return {
        "message": "Transcription successful",
        "transcript": transcript
    }
