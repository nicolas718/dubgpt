from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from mimetypes import add_type
import os
import requests
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
import shutil
import uuid
import subprocess

load_dotenv()

app = FastAPI()

# Mount the /static folder to serve files with correct MIME type
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
add_type("video/mp4", ".mp4", strict=True)
add_type("audio/mpeg", ".mp3", strict=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
SYNC_API_KEY = os.getenv("SYNC_API_KEY")
SYNC_API_URL = "https://api.sync.so/v2/generate"

@app.get("/")
def read_root():
    return {"message": "DubGPT backend is live."}

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    session_id = str(uuid.uuid4())[:8]
    file_basename = file.filename.rsplit(".", 1)[0]

    # Save uploaded video to temp
    original_path = f"/tmp/{session_id}_{file.filename}"
    with open(original_path, "wb") as buffer:
        buffer.write(await file.read())

    # Convert to .mp4 if needed
    if not original_path.endswith(".mp4"):
        video_path = original_path.rsplit(".", 1)[0] + ".mp4"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", original_path,
                "-vcodec", "libx264", "-acodec", "aac",
                "-f", "mp4", video_path
            ], check=True)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Video conversion failed: {str(e)}"})
    else:
        video_path = original_path

    # Extract audio
    audio_path = video_path.rsplit(".", 1)[0] + ".mp3"
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Audio extraction failed: {str(e)}"})

    # Transcription
    try:
        with open(audio_path, "rb") as audio_file:
            headers = {"xi-api-key": ELEVENLABS_API_KEY}
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

    # Voice cloning
    try:
        with open(audio_path, "rb") as audio_file:
            headers = {"xi-api-key": ELEVENLABS_API_KEY}
            files = {"files": audio_file}
            data = {"name": f"voice_clone_{file.filename}", "labels": "{}"}
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

    # Generate dubbed audio
    try:
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": translated_text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        tts_response = requests.post(tts_url, headers=headers, json=payload)
        dubbed_audio_filename = f"{session_id}_{file_basename}_{target_language}.mp3"
        dubbed_audio_path = os.path.join(STATIC_DIR, dubbed_audio_filename)
        with open(dubbed_audio_path, "wb") as out_file:
            out_file.write(tts_response.content)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS failed: {str(e)}"})

    # Copy video to static folder
    final_video_filename = f"{session_id}_{file_basename}.mp4"
    static_video_path = os.path.join(STATIC_DIR, final_video_filename)
    shutil.copy(video_path, static_video_path)

    # Log URLs for debugging
    print("Serving video at:", f"https://dubgpt-backend.up.railway.app/static/{final_video_filename}")
    print("Serving audio at:", f"https://dubgpt-backend.up.railway.app/static/{dubbed_audio_filename}")

    # Call Sync Labs
    try:
        sync_headers = {
            "x-api-key": SYNC_API_KEY,
            "Content-Type": "application/json"
        }
        sync_payload = {
            "model": "lipsync-1.8.0",
            "input": [
                {"type": "video", "url": f"https://dubgpt-backend.up.railway.app/static/{final_video_filename}"},
                {"type": "audio", "url": f"https://dubgpt-backend.up.railway.app/static/{dubbed_audio_filename}"}
            ]
        }
        sync_response = requests.post(SYNC_API_URL, headers=sync_headers, json=sync_payload)
        sync_data = sync_response.json()

        if sync_response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Sync Labs failed", "details": sync_data})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Sync Labs request failed: {str(e)}"})

    return {
        "message": "Video dubbing and lip-sync complete.",
        "translated_text": translated_text,
        "voice_id": voice_id,
        "dubbed_audio_url": f"https://dubgpt-backend.up.railway.app/static/{dubbed_audio_filename}",
        "original_video_url": f"https://dubgpt-backend.up.railway.app/static/{final_video_filename}",
        "sync_labs_url": sync_data.get("url")
    }


