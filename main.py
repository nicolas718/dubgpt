from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import requests
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")

@app.get("/")
def read_root():
    return {"message": "DubGPT backend is live."}

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    # Save uploaded file
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Extract audio
    audio_path = file_location.rsplit(".", 1)[0] + ".mp3"
    try:
        video = VideoFileClip(file_location)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Audio extraction failed: {str(e)}"})

    # Transcription & voice cloning
    try:
        with open(audio_path, "rb") as audio_file:
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY
            }
            files = {
                "audio": audio_file
            }
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text/scribe-v1",
                headers=headers,
                files=files
            )
            result = response.json()
            transcript = result.get("text", "")
            voice_id = result.get("voice_id", "unknown")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Transcription failed: {str(e)}"})

    # Translate transcript
    try:
        translate_response = requests.post(
            f"https://translation.googleapis.com/language/translate/v2?key={GOOGLE_TRANSLATE_API_KEY}",
            json={
                "q": transcript,
                "target": target_language,
                "format": "text"
            }
        )
        translate_data = translate_response.json()
        translated_text = translate_data["data"]["translations"][0]["translatedText"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Translation failed: {str(e)}"})

    return {
        "message": "Transcription, voice cloning, and translation complete",
        "original_transcript": transcript,
        "translated_transcript": translated_text,
        "voice_id": voice_id
    }

