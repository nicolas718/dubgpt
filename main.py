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

    # Transcribe audio
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
            transcript_text = transcript_data.get("text", "")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Transcription failed: {str(e)}"})

    # Clone voice
    try:
        with open(audio_path, "rb") as audio_file:
            headers = { "xi-api-key": ELEVENLABS_API_KEY }
            files = { "files": audio_file }
            data = { "name": f"voice_clone_{file.filename}", "labels": "{}" }
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

    # Translate transcript
    try:
        translate_url = "https://translation.googleapis.com/language/translate/v2"
        translate_params = {
            "q": transcript_text,
            "target": target_language,
            "format": "text",
            "key": GOOGLE_TRANSLATE_API_KEY
        }
        translate_response = requests.post(translate_url, data=translate_params)
        translated_text = translate_response.json()["data"]["translations"][0]["translatedText"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Translation failed: {str(e)}"})

    return {
        "message": "Transcription, voice cloning, and translation complete",
        "original_transcript": transcript_text,
        "translated_transcript": translated_text,
        "voice_id": voice_id
    }
