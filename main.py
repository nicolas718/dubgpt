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
UPLOAD_IO_SECRET_KEY = os.getenv("UPLOAD_IO_SECRET_KEY")
SYNC_API_KEY = os.getenv("SYNC_API_KEY")
SYNC_API_URL = "https://api.sync.so/v2/generate"
UPLOAD_IO_URL = "https://api.upload.io/v1/files/form_data"

@app.get("/")
def read_root():
    return {"message": "DubGPT backend is live."}

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    try:
        file_location = f"/tmp/{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        audio_path = file_location.rsplit(".", 1)[0] + ".mp3"
        try:
            video = VideoFileClip(file_location)
            video.audio.write_audiofile(audio_path)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Audio extraction failed: {str(e)}"})

        with open(file_location, "rb") as f:
            upload_io_response = requests.post(
                UPLOAD_IO_URL,
                headers={"Authorization": f"Bearer {UPLOAD_IO_SECRET_KEY}"},
                files={"file": (file.filename, f, file.content_type)}
            )

        if upload_io_response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Upload.io failed", "details": upload_io_response.text})

        try:
            uploaded_file_info = upload_io_response.json()
        except Exception:
            return JSONResponse(status_code=500, content={"error": "Upload.io returned invalid JSON", "raw": upload_io_response.text})

        try:
            video_url = uploaded_file_info["files"][0]["fileUrl"]
        except Exception:
            return JSONResponse(status_code=500, content={"error": "Upload.io did not return a file URL.", "full_response": uploaded_file_info})

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
                voice_id = voice_data.get("voice_id")
                if not voice_id:
                    return JSONResponse(status_code=500, content={"error": "Voice cloning failed", "details": voice_data})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Voice cloning failed: {str(e)}"})

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
            dubbed_audio_path = file_location.rsplit(".", 1)[0] + f"_{target_language}.mp3"
            with open(dubbed_audio_path, "wb") as out_file:
                out_file.write(tts_response.content)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"TTS failed: {str(e)}"})

        with open(dubbed_audio_path, "rb") as a:
            audio_upload_response = requests.post(
                UPLOAD_IO_URL,
                headers={"Authorization": f"Bearer {UPLOAD_IO_SECRET_KEY}"},
                files={"file": (os.path.basename(dubbed_audio_path), a, "audio/mpeg")}
            )

        audio_info = audio_upload_response.json()
        audio_url = audio_info["files"][0]["fileUrl"]

        sync_headers = {
            "x-api-key": SYNC_API_KEY,
            "Content-Type": "application/json"
        }
        sync_payload = {
            "model": "lipsync-1.8.0",
            "input": [
                {"type": "video", "url": video_url},
                {"type": "audio", "url": audio_url}
            ]
        }
        sync_response = requests.post(SYNC_API_URL, headers=sync_headers, json=sync_payload)
        sync_data = sync_response.json()

        sync_id = sync_data.get("id")

        return {
            "message": "Dubbing and lip-sync request sent.",
            "translated_text": translated_text,
            "dubbed_audio_url": audio_url,
            "video_url": video_url,
            "voice_id": voice_id,
            "sync_id": sync_id
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})

@app.get("/status/{sync_id}")
def check_sync_status(sync_id: str):
    try:
        status_url = f"https://api.sync.so/v2/generate/{sync_id}"
        headers = {"x-api-key": SYNC_API_KEY}
        response = requests.get(status_url, headers=headers)
        return response.json()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to fetch status: {str(e)}"})

