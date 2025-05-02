from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import requests
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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

    # Extract audio as .mp3
    audio_path = file_location.rsplit(".", 1)[0] + ".mp3"
    try:
        video = VideoFileClip(file_location)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Audio extraction failed: {str(e)}"})

    # Transcribe and clone voice
    try:
        with open(audio_path, "rb") as audio_file:
            headers = { "xi-api-key": ELEVENLABS_API_KEY }
            files = { "audio": audio_file }
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-speech",
                headers=headers,
                files=files
            )
            transcript_data = response.json()
            transcript_text = transcript_data.get("text", "")
            voice_id = transcript_data.get("voice_id", None)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Transcription or voice cloning failed: {str(e)}"})

    # Translate transcript
    try:
        translate_url = f"https://translation.googleapis.com/language/translate/v2"
        params = {
            "q": transcript_text,
            "target": target_language,
            "key": GOOGLE_API_KEY
        }
        response = requests.post(translate_url, params=params)
        translated_data = response.json()
        translated_text = translated_data["data"]["translations"][0]["translatedText"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Translation failed: {str(e)}"})

    return {
        "message": "Transcription, voice cloning, and translation complete",
        "original_transcript": transcript_text,
        "translated_transcript": translated_text,
        "voice_id": voice_id
    }

