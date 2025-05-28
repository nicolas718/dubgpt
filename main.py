import os
import uuid
import shutil
import tempfile
import traceback
from typing import Dict, Optional
from contextlib import asynccontextmanager

import aiofiles
import requests
import replicate
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.fx.all import speedx
from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# Configuration
class Settings(BaseSettings):
    google_translate_api_key: str
    replicate_api_token: str
    whisper_model: str = "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"
    xtts_model: str = "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e"
    max_file_size_mb: int = 500
    audio_tolerance_seconds: float = 0.1
    
    class Config:
        env_file = ".env"

settings = Settings()

# Constants
SUPPORTED_LANGUAGES = ["es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "en", "ar", "hi"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

# Job tracking
job_status: Dict[str, Dict] = {}

# Custom Exceptions
class DubbingError(Exception):
    def __init__(self, stage: str, detail: str):
        self.stage = stage
        self.detail = detail
        super().__init__(f"{stage}: {detail}")

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Polydub backend starting up...")
    yield
    # Shutdown - cleanup any remaining temp files
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith("polydub_"):
            try:
                path = os.path.join(temp_dir, filename)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except:
                pass

app = FastAPI(title="Polydub API", version="2.0", lifespan=lifespan)

# Helper functions
async def save_upload_file(upload_file: UploadFile, destination: str) -> None:
    """Save uploaded file in chunks to avoid memory issues"""
    async with aiofiles.open(destination, 'wb') as f:
        while chunk := await upload_file.read(1024 * 1024):  # 1MB chunks
            await f.write(chunk)

def validate_inputs(filename: str, target_language: str, file_size: int) -> None:
    """Validate file format, language, and size"""
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )
    
    if target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    
    if file_size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )

def update_job_status(job_id: str, status: str, progress: int = 0, 
                     result: Optional[str] = None, error: Optional[str] = None):
    """Update job status in tracking dictionary"""
    job_status[job_id] = {
        "status": status,
        "progress": progress,
        "result": result,
        "error": error
    }

async def process_video(job_id: str, file_path: str, filename: str, target_language: str):
    """Main video processing pipeline"""
    temp_dir = os.path.dirname(file_path)
    
    try:
        # Update status: Extracting audio
        update_job_status(job_id, "extracting_audio", 10)
        
        audio_path = os.path.join(temp_dir, "audio.wav")
        video = VideoFileClip(file_path)
        original_duration = video.duration
        
        # Extract audio as WAV for consistency
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        
        # Update status: Transcribing
        update_job_status(job_id, "transcribing", 25)
        
        # Transcribe with Whisper
        try:
            with open(audio_path, "rb") as audio_file:
                output = replicate.run(
                    settings.whisper_model,
                    input={"audio": audio_file}
                )
            
            if isinstance(output, dict) and "text" in output:
                transcript_text = output["text"]
            elif isinstance(output, str):
                transcript_text = output
            else:
                raise ValueError("Invalid output format from Whisper")
                
        except Exception as e:
            raise DubbingError("transcription", str(e))
        
        # Update status: Translating
        update_job_status(job_id, "translating", 40)
        
        # Translate text
        try:
            translate_response = requests.post(
                "https://translation.googleapis.com/language/translate/v2",
                data={
                    "q": transcript_text,
                    "target": target_language,
                    "format": "text",
                    "key": settings.google_translate_api_key
                }
            )
            translate_response.raise_for_status()
            translated_text = translate_response.json()["data"]["translations"][0]["translatedText"]
            
        except Exception as e:
            raise DubbingError("translation", str(e))
        
        # Update status: Generating voice
        update_job_status(job_id, "generating_voice", 60)
        
        # Generate dubbed audio with XTTS
        try:
            with open(audio_path, "rb") as audio_file:
                input_data = {
                    "text": translated_text,
                    "speaker": audio_file,
                    "language": target_language
                }
                output = replicate.run(settings.xtts_model, input=input_data)
            
            if isinstance(output, str):
                audio_url = output
            elif isinstance(output, list) and len(output) > 0:
                audio_url = output[0]
            else:
                raise ValueError("Invalid output format from XTTS")
            
            # Download generated audio
            dubbed_audio_path = os.path.join(temp_dir, f"dubbed_{target_language}.wav")
            with requests.get(audio_url, stream=True) as r:
                r.raise_for_status()
                with open(dubbed_audio_path, "wb") as out_file:
                    for chunk in r.iter_content(chunk_size=8192):
                        out_file.write(chunk)
                        
        except Exception as e:
            raise DubbingError("voice_synthesis", str(e))
        
        # Update status: Merging video
        update_job_status(job_id, "merging_video", 80)
        
        # Merge with speed adjustment if needed
        try:
            output_filename = f"{os.path.splitext(filename)[0]}_dubbed_{target_language}.mp4"
            output_path = os.path.join(temp_dir, output_filename)
            
            dubbed_audio = AudioFileClip(dubbed_audio_path)
            dubbed_duration = dubbed_audio.duration
            
            # Adjust audio speed if duration mismatch is significant
            if abs(original_duration - dubbed_duration) > settings.audio_tolerance_seconds:
                speed_factor = dubbed_duration / original_duration
                dubbed_audio = speedx(dubbed_audio, factor=speed_factor)
            
            # Create final video
            final_video = video.set_audio(dubbed_audio)
            final_video.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                logger=None,
                temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a")
            )
            
            # Cleanup video objects
            video.close()
            dubbed_audio.close()
            final_video.close()
            
        except Exception as e:
            raise DubbingError("video_merge", str(e))
        
        # Update status: Complete
        update_job_status(job_id, "completed", 100, result=output_path)
        
    except DubbingError as e:
        error_msg = f"{e.stage} failed: {e.detail}"
        update_job_status(job_id, "failed", error=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        update_job_status(job_id, "failed", error=error_msg)

# API Endpoints
@app.get("/")
def read_root():
    return {
        "message": "Polydub backend v2.0 is live",
        "endpoints": {
            "/upload": "POST - Upload video for dubbing",
            "/status/{job_id}": "GET - Check job status",
            "/download/{job_id}": "GET - Download completed video",
            "/languages": "GET - List supported languages",
            "/formats": "GET - List supported video formats"
        }
    }

@app.get("/languages")
def get_languages():
    return {"supported_languages": SUPPORTED_LANGUAGES}

@app.get("/formats")
def get_formats():
    return {"supported_formats": SUPPORTED_VIDEO_FORMATS}

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    # Validate inputs
    file_size = 0
    temp_dir = None
    
    try:
        # Check file size by reading in chunks
        temp_file = tempfile.SpooledTemporaryFile(max_size=1024*1024)
        while chunk := await file.read(1024*1024):
            file_size += len(chunk)
            temp_file.write(chunk)
        temp_file.seek(0)
        
        validate_inputs(file.filename, target_language, file_size)
        
        # Create job
        job_id = str(uuid.uuid4())
        update_job_status(job_id, "uploading", 0)
        
        # Create temporary directory for this job
        temp_dir = tempfile.mkdtemp(prefix=f"polydub_{job_id}_")
        file_path = os.path.join(temp_dir, file.filename)
        
        # Save file
        with open(file_path, 'wb') as f:
            temp_file.seek(0)
            shutil.copyfileobj(temp_file, f)
        temp_file.close()
        
        # Start background processing
        background_tasks.add_task(
            process_video, 
            job_id, 
            file_path, 
            file.filename, 
            target_language
        )
        
        return {
            "job_id": job_id,
            "message": "Video processing started",
            "status_url": f"/status/{job_id}",
            "download_url": f"/download/{job_id}"
        }
        
    except HTTPException:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
    except Exception as e:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

@app.get("/download/{job_id}")
def download_video(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job_status[job_id]
    
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {status['status']}"
        )
    
    if not status["result"] or not os.path.exists(status["result"]):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Clean up job status after successful download
    def cleanup():
        try:
            temp_dir = os.path.dirname(status["result"])
            if os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
                shutil.rmtree(temp_dir)
            del job_status[job_id]
        except:
            pass
    
    return FileResponse(
        status["result"], 
        media_type="video/mp4",
        filename=os.path.basename(status["result"]),
        background=BackgroundTasks([cleanup])
    )

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_jobs": len([j for j in job_status.values() if j["status"] not in ["completed", "failed"]])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



