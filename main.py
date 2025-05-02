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
    # Save uploaded video
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Extract audio to .mp3
    audio_path = file_location.rsplit(".", 1)[0] + ".mp3"
    try:
        video = VideoFileClip(file_location)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Audio extraction failed: {str(e)}"})

    # Step 1: Transcribe audio with ElevenLabs Scribe
    try:
        with open(audio_path, "rb") as audio_file:
            headers = { "xi-api-key": ELEVENLABS_API_KEY }
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers=headers,
                data={"model_id": "scribe_v1"},
                files={"file": audio_file}
            )
            transcript_data = response.json()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Transcription failed: {str(e)}"})

    # Step 2: Clone the speaker's voice
    try:
        with open(audio_path, "rb") as audio_file:
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY
            }
            files = {
                "files": audio_file
            }
            data = {
                "name": f"voice_clone_{file.filename}",
                "labels": "{}"
            }
            voice_response = requests.post(
                "https://api.elevenlabs.io/v1/voices/add",
                headers=headers,
                data=data,
                files=files
            )
            voice_data = voice_response.json()
            voice_id = voice_data.get("voice_id", None)
            if not voice_id:
                return JSONResponse(status_code=500, content={"error": "Voice cloning failed", "details": voice_data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Voice cloning failed: {str(e)}"})

    return {
        "message": "Transcription and voice cloning successful",
        "transcript": transcript_data,
        "voice_id": voice_id
    }
