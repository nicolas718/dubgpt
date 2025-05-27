from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import boto3
import requests
import time
from moviepy.editor import VideoFileClip, AudioFileClip
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
# Load environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
# S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)
def upload_to_s3(local_path, s3_key):
    s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
    url = f"https://{S3_BUCKET_NAME}.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{s3_key}"
    return url
@app.get("/")
def read_root():
    return {"message": "Polydub backend is live."}
@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    # ... (Transcription and translation steps remain the same)
    # 7. Generate dubbed audio with Replicate
    try:
        replicate_url = "https://api.replicate.com/v1/predictions"
        replicate_headers = {
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "version": "f9f24e8c643ee44d0a51b9a0a7a17d9a652b0992acf7dac1fc39bcd19afa06a1",
            "input": {
                "audio": audio_s3_url,
                "transcript": translated_text
            }
        }
        replicate_response = requests.post(
            replicate_url,
            headers=replicate_headers,
            json=payload
        )
        replicate_data = replicate_response.json()
        if replicate_response.status_code != 201:
            return JSONResponse(status_code=500, content={"error": "Replicate TTS request failed", "details": replicate_data})
        tts_audio_url = None
        while not tts_audio_url:
            time.sleep(1)
            prediction_url = replicate_data["urls"]["get"]
            prediction_response = requests.get(prediction_url, headers=replicate_headers)
            prediction_data = prediction_response.json()
            if prediction_data["status"] == "succeeded":
                tts_audio_url = prediction_data["output"]
            elif prediction_data["status"] == "failed":
                return JSONResponse(status_code=500, content={"error": "Replicate TTS failed", "details": prediction_data})
        # Download the generated audio
        dubbed_audio_path = file_location.rsplit(".", 1)[0] + f"_{target_language}_dub.mp3"
        with requests.get(tts_audio_url, stream=True) as r:
            with open(dubbed_audio_path, "wb") as out_file:
                for chunk in r.iter_content(chunk_size=8192):
                    out_file.write(chunk)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS failed: {str(e)}"})
    # ... (Remaining steps for uploading dubbed audio, merging video, and uploading final video remain the same)


