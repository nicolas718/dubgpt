from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import requests
import os
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@app.get("/")
def read_root():
    return {"message": "DubGPT backend is live."}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), target_language: str = Form(...)):
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    audio_path = file_location.rsplit(".", 1)[0] + ".mp3"

    try:
        video = VideoFileClip(file_location)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Audio extraction failed.", "detail": str(e)})

    # Transcribe and clone voice
    try:
        with open(audio_path, "rb") as audio_file:
            headers = { "xi-api-key": ELEVENLABS_API_KEY }
            files = { "audio": audio_file }
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text", 
                headers=headers, 
                files=files
            )
            transcript_response = response.json()
            transcript_text = transcript_response["transcript"]["text"]
            voice_id = transcript_response["voice_id"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Transcription or voice cloning failed.", "detail": str(e)})

    # Translate the transcript
    try:
        translate_url = f"https://translation.googleapis.com/language/translate/v2?key={GOOGLE_API_KEY}"
        payload = {
            "q": transcript_text,
            "source": "en",
            "target": target_language,
            "format": "text"
        }
        translate_response = requests.post(translate_url, json=payload)
        translated_text = translate_response.json()["data"]["translations"][0]["translatedText"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Translation failed.", "detail": str(e)})

    # Generate translated audio
    try:
        tts_headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        tts_payload = {
            "model_id": "eleven_multilingual_v2",
            "text": translated_text,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        tts_response = requests.post(tts_url, headers=tts_headers, json=tts_payload)

        dubbed_audio_path = file_location.rsplit(".", 1)[0] + f"_{target_language}.mp3"
        with open(dubbed_audio_path, "wb") as out:
            out.write(tts_response.content)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Voice dubbing failed.", "detail": str(e)})

    return {
        "message": "Translation and dubbing complete.",
        "original_transcript": transcript_text,
        "translated_transcript": translated_text,
        "dubbed_audio_path": dubbed_audio_path
    }
