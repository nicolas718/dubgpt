import sys
import os
import time
import boto3
import requests
import replicate
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip, AudioFileClip
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

def upload_to_s3(local_path, s3_key):
    s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
    return f"https://{S3_BUCKET_NAME}.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{s3_key}"

def generate_presigned_url(bucket, key, expiration=3600):
    return s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=expiration
    )

@app.get("/")
def read_root():
    return {"message": "Polydub backend is live."}

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    audio_path = file_location.rsplit(".", 1)[0] + ".mp3"
    try:
        video = VideoFileClip(file_location)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Audio extraction failed: {str(e)}"})

    audio_s3_key = f"uploads/{os.path.basename(audio_path)}"
    upload_to_s3(audio_path, audio_s3_key)
    audio_s3_url = generate_presigned_url(S3_BUCKET_NAME, audio_s3_key)

    transcribe = boto3.client("transcribe",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION)

    job_name = f"polydub-job-{int(time.time())}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": audio_s3_url},
        MediaFormat="mp3",
        LanguageCode="en-US"
    )

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
            break
        time.sleep(5)

    if status["TranscriptionJob"]["TranscriptionJobStatus"] == "FAILED":
        return JSONResponse(status_code=500, content={"error": "Transcription failed"})

    transcript_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    transcript_data = requests.get(transcript_url).json()
    transcript_text = transcript_data["results"]["transcripts"][0]["transcript"]

    try:
        translate_response = requests.post(
            "https://translation.googleapis.com/language/translate/v2",
            data={
                "q": transcript_text,
                "target": target_language,
                "format": "text",
                "key": GOOGLE_TRANSLATE_API_KEY
            }
        )
        translated_text = translate_response.json()["data"]["translations"][0]["translatedText"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Translation failed: {str(e)}"})

    print("[DEBUG] Translated text:", translated_text)
    print("[DEBUG] Presigned S3 URL:", audio_s3_url)

    try:
        input = {
            "text": translated_text,
            "speaker": audio_s3_url,
            "language": target_language
        }
        output = replicate.run(
            "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e",
            input=input
        )

        if isinstance(output, list) and len(output) > 0:
            audio_url = output[0]
        else:
            raise ValueError("Invalid output format from replicate.run")

        dubbed_audio_path = file_location.rsplit(".", 1)[0] + f"_{target_language}_dub.wav"
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(dubbed_audio_path, "wb") as out_file:
                for chunk in r.iter_content(chunk_size=8192):
                    out_file.write(chunk)

    except Exception as e:
        tb_str = traceback.format_exc()
        print("[EXCEPTION TYPE]", type(e))
        print("[EXCEPTION]", str(e))
        print("[TRACEBACK]\n", tb_str)
        return JSONResponse(status_code=500, content={
            "error": "Replicate TTS failed",
            "exception": str(e),
            "traceback": tb_str
        })

    dubbed_key = f"dubbed/{os.path.basename(dubbed_audio_path)}"
    dubbed_url = upload_to_s3(dubbed_audio_path, dubbed_key)

    try:
        output_video_path = file_location.rsplit(".", 1)[0] + f"_final_{target_language}.mp4"
        original_video = VideoFileClip(file_location)
        dubbed_audio = AudioFileClip(dubbed_audio_path)
        final_video = original_video.set_audio(dubbed_audio)
        final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Final video merge failed: {str(e)}"})

    final_key = f"final/{os.path.basename(output_video_path)}"
    final_url = upload_to_s3(output_video_path, final_key)

    return {
        "message": "Pipeline complete: Transcription, translation, TTS, merge done.",
        "original_transcript": transcript_text,
        "translated_transcript": translated_text,
        "dubbed_audio_s3_url": dubbed_url,
        "final_video_s3_url": final_url
    }




