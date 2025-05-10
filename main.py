from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import requests
import httpx
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
import uuid

load_dotenv()

app = FastAPI()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
WAV2LIP_VM_URL = "http://35.226.116.89:8000/lipsync"  # Your VM endpoint

@app.get("/")
def read_root():
    return {"message": "DubGPT Railway backend is live."}

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    session_id = str(uuid.uuid4())[:8]
    file_location = f"/tmp/{session_id}_{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Extract audio
    audio_path = file_location.rsplit(".", 1)[0] + ".mp3"
    try:
        video = VideoFileClip(file_location)
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

    # Generate dubbed audio using TTS
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

    # Send video + dubbed audio to Wav2Lip VM (async, correct)
    try:
        async with httpx.AsyncClient() as client:
            with open(file_location, "rb") as video_file, open(dubbed_audio_path, "rb") as audio_file:
                files = {
                    "video": (file.filename, video_file, "video/mp4"),
                    "audio": (f"dub_{target_language}.mp3", audio_file, "audio/mpeg")
                }
                response = await client.post(WAV2LIP_VM_URL, files=files)

        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Wav2Lip VM processing failed", "details": response.text})

        result = response.json()
        return {
            "message": "All processing complete, including Wav2Lip.",
            "original_transcript": transcript_text,
            "translated_transcript": translated_text,
            "voice_id": voice_id,
            "download_url": result["download_url"]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Sending to Wav2Lip VM failed: {str(e)}"})


