import os
import json
import uuid
import shutil
import tempfile
import traceback
import subprocess
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass

import aiofiles
import requests
import replicate
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips, CompositeAudioClip
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TimedSegment:
    text: str
    start_time: float
    end_time: float
    duration: float
    words: List[dict] = None

class Settings:
    def __init__(self):
        self.google_translate_api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
        self.replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
        # Using newer Whisper model that supports better timestamps
        self.whisper_model = os.getenv("WHISPER_MODEL", "vaibhavs10/incredibly-fast-whisper:3ab86b17445d2100f7f8e20cce9d1f4f0c4d201d7d5f0a4de7129fcf745d4f22")
        self.xtts_model = os.getenv("XTTS_MODEL", "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e")
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "500"))

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
    print("Polydub backend starting up...")
    yield
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

app = FastAPI(title="Polydub API", version="3.0", lifespan=lifespan)

# Helper functions
def update_job_status(job_id: str, status: str, progress: int = 0, 
                     result: Optional[str] = None, error: Optional[str] = None):
    """Update job status in tracking dictionary"""
    job_status[job_id] = {
        "status": status,
        "progress": progress,
        "result": result,
        "error": error
    }

def extract_segments_with_timing(whisper_output: dict) -> List[TimedSegment]:
    """Extract segments with word-level timing from Whisper/WhisperX output"""
    segments = []
    
    # Handle different output formats
    if isinstance(whisper_output, dict):
        # WhisperX format with word-level timing
        if "segments" in whisper_output:
            for seg in whisper_output["segments"]:
                # Extract words if available (WhisperX provides these)
                words = seg.get("words", [])
                
                # If we have word-level timing, create smaller segments
                if words:
                    # Group words into ~3-second segments
                    current_words = []
                    current_start = None
                    
                    for word in words:
                        if current_start is None:
                            current_start = word.get("start", 0)
                        
                        current_words.append(word)
                        current_end = word.get("end", word.get("start", 0))
                        
                        # Create segment if duration > 3 seconds or at sentence end
                        word_text = word.get("word", word.get("text", ""))
                        if (current_end - current_start > 3.0 or 
                            word_text.rstrip().endswith(('.', '!', '?', ','))):
                            
                            segment_text = " ".join([w.get("word", w.get("text", "")) for w in current_words])
                            segment = TimedSegment(
                                text=segment_text.strip(),
                                start_time=current_start,
                                end_time=current_end,
                                duration=current_end - current_start,
                                words=current_words
                            )
                            if segment.text:
                                segments.append(segment)
                            current_words = []
                            current_start = None
                    
                    # Add remaining words
                    if current_words:
                        segment_text = " ".join([w.get("word", w.get("text", "")) for w in current_words])
                        segment = TimedSegment(
                            text=segment_text.strip(),
                            start_time=current_start,
                            end_time=current_words[-1].get("end", current_start),
                            duration=current_words[-1].get("end", current_start) - current_start,
                            words=current_words
                        )
                        if segment.text:
                            segments.append(segment)
                else:
                    # No word-level timing, use segment as-is
                    segment = TimedSegment(
                        text=seg.get("text", "").strip(),
                        start_time=seg.get("start", 0),
                        end_time=seg.get("end", 0),
                        duration=seg.get("end", 0) - seg.get("start", 0),
                        words=[]
                    )
                    if segment.text:
                        segments.append(segment)
        
        # Some Whisper models return chunks
        elif "chunks" in whisper_output:
            for chunk in whisper_output["chunks"]:
                segment = TimedSegment(
                    text=chunk.get("text", "").strip(),
                    start_time=chunk.get("timestamp", [0])[0],
                    end_time=chunk.get("timestamp", [0, 0])[1],
                    duration=chunk.get("timestamp", [0, 0])[1] - chunk.get("timestamp", [0])[0],
                    words=[]
                )
                if segment.text:
                    segments.append(segment)
    
    print(f"Extracted {len(segments)} segments, with word timing: {any(s.words for s in segments)}")
    return segments

def split_long_segments(segments: List[TimedSegment], max_duration: float = 3.0) -> List[TimedSegment]:
    """Split long segments into smaller chunks for better dubbing"""
    new_segments = []
    
    for segment in segments:
        if segment.duration <= max_duration:
            new_segments.append(segment)
        else:
            # Split by sentences or at natural breaks
            sentences = segment.text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 1:
                # If can't split by sentences, split by word count
                words = segment.text.split()
                words_per_chunk = len(words) // int(segment.duration / max_duration) + 1
                sentences = []
                for i in range(0, len(words), words_per_chunk):
                    sentences.append(' '.join(words[i:i+words_per_chunk]))
            
            # Calculate time per sentence
            time_per_char = segment.duration / len(segment.text)
            current_time = segment.start_time
            
            for sentence in sentences:
                sentence_duration = len(sentence) * time_per_char
                new_segment = TimedSegment(
                    text=sentence,
                    start_time=current_time,
                    end_time=current_time + sentence_duration,
                    duration=sentence_duration
                )
                new_segments.append(new_segment)
                current_time += sentence_duration
    
    return new_segments

async def generate_segment_audio_with_timing(
    segment: TimedSegment,
    target_language: str,
    speaker_audio_path: str,
    output_path: str,
    temp_dir: str
) -> str:
    """Generate TTS for a segment and adjust to match duration"""
    
    # Translate the segment
    try:
        translate_response = requests.post(
            "https://translation.googleapis.com/language/translate/v2",
            data={
                "q": segment.text,
                "target": target_language,
                "format": "text",
                "key": settings.google_translate_api_key
            }
        )
        translate_response.raise_for_status()
        translated_text = translate_response.json()["data"]["translations"][0]["translatedText"]
    except Exception as e:
        raise DubbingError("translation", str(e))
    
    # Generate TTS
    try:
        with open(speaker_audio_path, "rb") as audio_file:
            output = replicate.run(
                settings.xtts_model,
                input={
                    "text": translated_text,
                    "speaker": audio_file,
                    "language": target_language
                }
            )
        
        if isinstance(output, str):
            audio_url = output
        elif isinstance(output, list) and len(output) > 0:
            audio_url = output[0]
        else:
            raise ValueError("Invalid output from XTTS")
        
        # Download audio
        temp_output = output_path + "_temp.wav"
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_output, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Check duration and adjust if needed
        audio_clip = AudioFileClip(temp_output)
        actual_duration = audio_clip.duration
        audio_clip.close()
        
        if abs(actual_duration - segment.duration) > 0.1:  # More than 100ms difference
            speed_factor = actual_duration / segment.duration
            print(f"Adjusting speed by {speed_factor}x for segment: {segment.text[:30]}...")
            
            # Use ffmpeg to adjust speed (atempo range: 0.5 to 2.0)
            if 0.5 <= speed_factor <= 2.0:
                cmd = [
                    'ffmpeg', '-i', temp_output,
                    '-filter:a', f'atempo={speed_factor}',
                    '-y', output_path
                ]
            else:
                # For extreme speed changes, chain multiple atempo filters
                atempos = []
                remaining = speed_factor
                while remaining > 2.0:
                    atempos.append('atempo=2.0')
                    remaining /= 2.0
                while remaining < 0.5:
                    atempos.append('atempo=0.5')
                    remaining *= 2.0
                if remaining != 1.0:
                    atempos.append(f'atempo={remaining}')
                
                filter_chain = ','.join(atempos)
                cmd = [
                    'ffmpeg', '-i', temp_output,
                    '-filter:a', filter_chain,
                    '-y', output_path
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                os.remove(temp_output)
            else:
                print(f"FFmpeg speed adjustment failed: {result.stderr}")
                shutil.move(temp_output, output_path)
        else:
            shutil.move(temp_output, output_path)
        
        return output_path
        
    except Exception as e:
        raise DubbingError("tts_generation", str(e))

async def process_video_with_segment_dubbing(
    job_id: str,
    file_path: str,
    filename: str,
    target_language: str
):
    """Process video with segment-based dubbing for better synchronization"""
    temp_dir = os.path.dirname(file_path)
    
    try:
        # Extract audio
        update_job_status(job_id, "extracting_audio", 10)
        
        audio_path = os.path.join(temp_dir, "audio.wav")
        video = VideoFileClip(file_path)
        original_duration = video.duration
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        
        # Transcribe with timing using WhisperX for word-level timestamps
        update_job_status(job_id, "transcribing_with_word_timing", 20)
        
        try:
            with open(audio_path, "rb") as audio_file:
                # Try WhisperX first for word-level timing
                try:
                    output = replicate.run(
                        "thomasmol/whisper-diarization:cbd8d1e3e0e3e69e5bd2e28e61d7de74e6597f80c0cf61d845b756028c749595",
                        input={
                            "file": audio_file,
                            "num_speakers": 1,
                            "language": "en",  # Auto-detect
                            "batch_size": 8
                        }
                    )
                    print("Using WhisperX for word-level timing")
                except Exception as e:
                    print(f"WhisperX failed, falling back to regular Whisper: {e}")
                    # Fallback to regular Whisper
                    audio_file.seek(0)
                    output = replicate.run(
                        settings.whisper_model,
                        input={
                            "audio": audio_file,
                            "task": "transcribe",
                            "timestamp_granularities": "segment",
                            "return_timestamps": True
                        }
                    )
            
            print(f"Transcription output type: {type(output)}")
            if isinstance(output, dict):
                print(f"Output keys: {output.keys()}")
                if "segments" in output and len(output["segments"]) > 0:
                    first_segment = output["segments"][0]
                    print(f"First segment keys: {first_segment.keys()}")
                    if "words" in first_segment:
                        print(f"Word-level timing available! First word: {first_segment['words'][0] if first_segment['words'] else 'No words'}")
            
        except Exception as e:
            raise DubbingError("transcription", str(e))
        
        # Extract segments with timing
        segments = extract_segments_with_timing(output)
        
        if not segments:
            # Fallback: create one segment from the whole transcription
            if isinstance(output, dict) and "text" in output:
                segments = [TimedSegment(
                    text=output["text"],
                    start_time=0,
                    end_time=original_duration,
                    duration=original_duration
                )]
            else:
                raise DubbingError("transcription", "No segments extracted from transcription")
        
        print(f"Extracted {len(segments)} segments")
        
        # Split long segments
        segments = split_long_segments(segments, max_duration=4.0)
        print(f"Split into {len(segments)} segments for dubbing")
        
        # Generate dubbed audio for each segment
        update_job_status(job_id, "generating_segment_dubbing", 40)
        
        audio_segments = []
        for i, segment in enumerate(segments):
            progress = 40 + (40 * i / len(segments))
            update_job_status(job_id, f"dubbing_segment_{i+1}_of_{len(segments)}", int(progress))
            
            segment_output_path = os.path.join(temp_dir, f"segment_{i:04d}.wav")
            
            try:
                await generate_segment_audio_with_timing(
                    segment,
                    target_language,
                    audio_path,
                    segment_output_path,
                    temp_dir
                )
                
                # Load and position the audio segment
                audio_seg = AudioFileClip(segment_output_path)
                # Set the audio to play at the exact time
                audio_seg = audio_seg.set_start(segment.start_time)
                audio_segments.append(audio_seg)
                
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                # Continue with other segments
        
        if not audio_segments:
            raise DubbingError("segment_generation", "No audio segments were generated successfully")
        
        # Create final audio
        update_job_status(job_id, "merging_audio_segments", 85)
        
        # Create composite audio from all segments
        final_audio = CompositeAudioClip(audio_segments)
        
        # Ensure audio matches video duration
        final_audio = final_audio.set_duration(original_duration)
        
        # Save temporary final audio
        temp_final_audio = os.path.join(temp_dir, "final_dubbed_audio.wav")
        final_audio.write_audiofile(temp_final_audio, logger=None)
        
        # Create final video
        update_job_status(job_id, "creating_final_video", 95)
        
        output_filename = f"{os.path.splitext(filename)[0]}_dubbed_{target_language}.mp4"
        output_path = os.path.join(temp_dir, output_filename)
        
        # Load the final audio and set it to video
        final_audio_clip = AudioFileClip(temp_final_audio)
        final_video = video.set_audio(final_audio_clip)
        
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            logger=None,
            temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a")
        )
        
        # Cleanup
        video.close()
        final_audio_clip.close()
        final_video.close()
        for seg in audio_segments:
            try:
                seg.close()
            except:
                pass
        
        update_job_status(job_id, "completed", 100, result=output_path)
        
    except DubbingError as e:
        error_msg = f"{e.stage} failed: {e.detail}"
        update_job_status(job_id, "failed", error=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        update_job_status(job_id, "failed", error=error_msg)

# [Include all other endpoints from original code...]

@app.get("/")
def read_root():
    return {
        "message": "Polydub backend v3.0 is live - with segment-based dubbing",
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
        
        # Validate
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in SUPPORTED_VIDEO_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported format")
        
        if target_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported language")
        
        if file_size > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File too large")
        
        # Create job
        job_id = str(uuid.uuid4())
        update_job_status(job_id, "uploading", 0)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"polydub_{job_id}_")
        file_path = os.path.join(temp_dir, file.filename)
        
        # Save file
        with open(file_path, 'wb') as f:
            temp_file.seek(0)
            shutil.copyfileobj(temp_file, f)
        temp_file.close()
        
        # Start processing with segment-based dubbing
        background_tasks.add_task(
            process_video_with_segment_dubbing,
            job_id,
            file_path,
            file.filename,
            target_language
        )
        
        return {
            "job_id": job_id,
            "message": "Video processing started with segment-based dubbing",
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
    
    # Clean up after download
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

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_jobs": len([j for j in job_status.values() if j["status"] not in ["completed", "failed"]])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
