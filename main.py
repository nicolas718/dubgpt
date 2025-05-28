from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import boto3
import requests
import time
from moviepy.editor import VideoFileClip, AudioFileClip
from dotenv import load_dotenv
import replicate

load_dotenv()
app = FastAPI()

# Load environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") 
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Set the Replicate API token
replicate.api_token = REPLICATE_API_TOKEN

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
    # 1. Save uploaded video to /tmp
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
    
    # 2. Extract audio as mp3
    audio_path = file_location.rsplit(".", 1)[0] + ".mp3"
    try:
        video = VideoFileClip(file_location)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Audio extraction failed: {str(e)}"})
    
    # 3. Upload audio to S3  
    audio_s3_key = f"uploads/{os.path.basename(audio_path)}"
    audio_s3_url = upload_to_s3(audio_path, audio_s3_key)
    
    # 4. Start AWS Transcribe job
    transcribe = boto3.client(
        "transcribe", 
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )
    job_name = f"polydub-job-{int(time.time())}"
    media_format = "mp3"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": audio_s3_url},
        MediaFormat=media_format,
        LanguageCode="en-US"  # Change this if needed
    )
    
    # 5. Poll for job completion  
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
            break
        time.sleep(5)
    
    if status["TranscriptionJob"]["TranscriptionJobStatus"] == "FAILED":
        return JSONResponse(status_code=500, content={"error": "Transcription failed"})
    
    transcript_file_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    transcript_data = requests.get(transcript_file_url).json()
    transcript_text = transcript_data["results"]["transcripts"][0]["transcript"]
    
    # 6. Translate text
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
    
    # 7. Generate dubbed audio with Replicate
    try:
        model_name = "lucastaco/xtts-v2"
        input_data = {
            "text": translated_text,
            "speaker": audio_s3_url,
            "language": target_language
        }
        output = replicate.run(
            model_name,
            input=input_data
        )
        
        # Download the generated audio
        dubbed_audio_path = file_location.rsplit(".", 1)[0] + f"_{target_language}_dub.wav"
        with open(dubbed_audio_path, "wb") as out_file:
            out_file.write(requests.get(output).content)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS failed: {str(e)}"})
    
    # 8. Upload dubbed audio to S3
    dubbed_audio_s3_key = f"dubbed/{os.path.basename(dubbed_audio_path)}"  
    dubbed_audio_s3_url = upload_to_s3(dubbed_audio_path, dubbed_audio_s3_key)
    
    # 9. Merge dubbed audio and original video
    try:
        output_video_path = file_location.rsplit(".", 1)[0] + f"_final_{target_language}.mp4"
        original_video = VideoFileClip(file_location)
        dubbed_audio = AudioFileClip(dubbed_audio_path)
        final_video = original_video.set_audio(dubbed_audio)
        final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac") 
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Final video merge failed: {str(e)}"})
    
    # 10. Upload final video to S3
    final_video_s3_key = f"final/{os.path.basename(output_video_path)}"
    final_video_s3_url = upload_to_s3(output_video_path, final_video_s3_key)
    
    return {
        "message": "Pipeline complete: Transcription, translation, TTS, merge done.",
        "original_transcript": transcript_text,
        "translated_transcript": translated_text, 
        "dubbed_audio_s3_url": dubbed_audio_s3_url,
        "final_video_s3_url": final_video_s3_url
    }

