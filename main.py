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
from moviepy.audio.AudioClip import AudioClip
from dotenv import load_dotenv

load_dotenv()

@dataclass
class WordTiming:
    word: str
    start: float
    end: float

@dataclass
class TimedSegment:
    text: str
    start_time: float
    end_time: float
    duration: float
    words: List[WordTiming] = None

class Settings:
    def __init__(self):
        self.replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
        # WhisperX for word-level transcription
        self.whisperx_model = "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"
        # GPT-4o for smart translation
        self.gpt4o_model = "openai/gpt-4o"
        # XTTS for voice synthesis
        self.xtts_model = "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e"
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "500"))

settings = Settings()

# Constants
SUPPORTED_LANGUAGES = ["es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "en", "ar", "hi"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

# Language names for GPT-4o
LANGUAGE_NAMES = {
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "en": "English", "ar": "Arabic", "hi": "Hindi"
}

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

app = FastAPI(title="Polydub API", version="4.0", lifespan=lifespan)

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

def extract_word_timings_from_whisperx(whisperx_output: dict) -> List[TimedSegment]:
    """Extract word-level timings from WhisperX output"""
    segments = []
    
    if isinstance(whisperx_output, dict) and "segments" in whisperx_output:
        for seg in whisperx_output["segments"]:
            words = []
            
            # WhisperX provides word-level timing
            if "words" in seg:
                for word_data in seg["words"]:
                    word = WordTiming(
                        word=word_data.get("word", ""),
                        start=word_data.get("start", 0),
                        end=word_data.get("end", 0)
                    )
                    words.append(word)
            
            segment = TimedSegment(
                text=seg.get("text", "").strip(),
                start_time=seg.get("start", 0),
                end_time=seg.get("end", 0),
                duration=seg.get("end", 0) - seg.get("start", 0),
                words=words
            )
            
            if segment.text:
                segments.append(segment)
    
    return segments

def group_words_into_dubbing_segments(segments: List[TimedSegment], target_duration: float = 3.0) -> List[TimedSegment]:
    """Group words into optimal segments for dubbing"""
    dubbing_segments = []
    
    for segment in segments:
        if not segment.words:
            # No word timing, use segment as-is
            dubbing_segments.append(segment)
            continue
        
        current_words = []
        current_start = None
        
        for i, word in enumerate(segment.words):
            if current_start is None:
                current_start = word.start
            
            current_words.append(word)
            duration = word.end - current_start
            
            # Check if we should create a segment
            should_segment = False
            
            # Natural break points
            if word.word.rstrip().endswith(('.', '!', '?', ',')):
                should_segment = True
            # Duration limit reached
            elif duration >= target_duration:
                should_segment = True
            # Last word
            elif i == len(segment.words) - 1:
                should_segment = True
            # Long pause after word (>0.3s)
            elif i < len(segment.words) - 1 and segment.words[i + 1].start - word.end > 0.3:
                should_segment = True
            
            if should_segment and current_words:
                text = ' '.join([w.word for w in current_words])
                dubbing_segment = TimedSegment(
                    text=text.strip(),
                    start_time=current_start,
                    end_time=word.end,
                    duration=word.end - current_start,
                    words=current_words
                )
                dubbing_segments.append(dubbing_segment)
                current_words = []
                current_start = None
    
    return dubbing_segments

async def smart_translate_segment(segment: TimedSegment, target_language: str, context: str = "") -> str:
    """Use GPT-4o to translate with timing constraints"""
    
    # Calculate speaking rate constraints
    word_count = len(segment.text.split())
    chars_per_second = len(segment.text) / segment.duration if segment.duration > 0 else 30
    
    # Estimate target language expansion factor
    expansion_factors = {
        "es": 1.25, "fr": 1.30, "de": 1.20, "it": 1.20,
        "pt": 1.25, "ru": 0.90, "ja": 0.70, "ko": 0.75,
        "zh": 0.50, "ar": 1.10, "hi": 1.15
    }
    
    expansion = expansion_factors.get(target_language, 1.2)
    max_chars = int(len(segment.text) / expansion)
    
    prompt = f"""Translate this text to {LANGUAGE_NAMES.get(target_language, target_language)}.

Original text: "{segment.text}"
Duration available: {segment.duration:.1f} seconds
Maximum characters: {max_chars}

Requirements:
1. Keep the translation concise to fit the time constraint
2. Preserve the exact meaning and key information
3. Use natural, conversational language
4. If needed, use shorter synonyms or remove filler words
5. The translation must be speakable in {segment.duration:.1f} seconds

Context from previous segments: {context[-200:] if context else 'Start of video'}

Return ONLY the translated text, no explanations."""

    try:
        output = replicate.run(
            settings.gpt4o_model,
            input={
                "prompt": prompt,
                "max_tokens": 150,
                "temperature": 0.3,
                "system_prompt": "You are a professional translator specializing in video dubbing. Always provide concise, accurate translations that fit within time constraints."
            }
        )
        
        # GPT-4o returns streaming output, join it
        translated_text = ''.join(output).strip()
        
        # Validate length
        if len(translated_text) > max_chars * 1.5:
            # Too long, ask for a shorter version
            retry_prompt = f"Make this {LANGUAGE_NAMES.get(target_language, target_language)} text shorter (max {max_chars} characters): {translated_text}"
            
            output = replicate.run(
                settings.gpt4o_model,
                input={
                    "prompt": retry_prompt,
                    "max_tokens": 100,
                    "temperature": 0.3
                }
            )
            translated_text = ''.join(output).strip()
        
        return translated_text
        
    except Exception as e:
        print(f"GPT-4o translation failed: {e}, using fallback")
        # Fallback: simple truncation
        return segment.text[:max_chars]

async def generate_dubbed_segment(
    segment: TimedSegment,
    translated_text: str,
    target_language: str,
    speaker_audio_path: str,
    output_path: str
) -> str:
    """Generate TTS for a segment with precise duration matching"""
    
    try:
        # Generate TTS with XTTS
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
        
        # Adjust duration to match exactly
        audio_clip = AudioFileClip(temp_output)
        actual_duration = audio_clip.duration
        audio_clip.close()
        
        if abs(actual_duration - segment.duration) > 0.05:  # 50ms tolerance
            speed_factor = actual_duration / segment.duration
            
            # Use FFmpeg for speed adjustment
            if 0.5 <= speed_factor <= 2.0:
                cmd = [
                    'ffmpeg', '-i', temp_output,
                    '-filter:a', f'atempo={speed_factor}',
                    '-y', output_path
                ]
            else:
                # Chain multiple atempo filters for extreme adjustments
                atempos = []
                remaining = speed_factor
                while remaining > 2.0:
                    atempos.append('atempo=2.0')
                    remaining /= 2.0
                while remaining < 0.5:
                    atempos.append('atempo=0.5')
                    remaining *= 2.0
                if abs(remaining - 1.0) > 0.01:
                    atempos.append(f'atempo={remaining}')
                
                filter_chain = ','.join(atempos) if atempos else 'anull'
                cmd = [
                    'ffmpeg', '-i', temp_output,
                    '-filter:a', filter_chain,
                    '-y', output_path
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                os.remove(temp_output)
            else:
                print(f"FFmpeg adjustment failed: {result.stderr}")
                shutil.move(temp_output, output_path)
        else:
            shutil.move(temp_output, output_path)
        
        return output_path
        
    except Exception as e:
        raise DubbingError("tts_generation", str(e))

async def process_video_with_perfect_sync(
    job_id: str,
    file_path: str,
    filename: str,
    target_language: str
):
    """Process video with word-level synchronization using WhisperX and GPT-4o"""
    temp_dir = os.path.dirname(file_path)
    
    try:
        # Extract audio
        update_job_status(job_id, "extracting_audio", 10)
        
        audio_path = os.path.join(temp_dir, "audio.wav")
        video = VideoFileClip(file_path)
        original_duration = video.duration
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        
        # Transcribe with WhisperX for word-level timing
        update_job_status(job_id, "transcribing_with_whisperx", 20)
        
        try:
            with open(audio_path, "rb") as audio_file:
                whisperx_output = replicate.run(
                    settings.whisperx_model,
                    input={
                        "audio_file": audio_file,
                        "batch_size": 16,
                        "align_output": True,  # Enable word-level alignment
                        "diarization": False   # Single speaker
                    }
                )
            
            print(f"WhisperX output type: {type(whisperx_output)}")
            
        except Exception as e:
            raise DubbingError("transcription", f"WhisperX failed: {str(e)}")
        
        # Extract word-level timings
        segments = extract_word_timings_from_whisperx(whisperx_output)
        if not segments:
            raise DubbingError("transcription", "No segments extracted from WhisperX")
        
        print(f"Extracted {len(segments)} segments with {sum(len(s.words) for s in segments)} words")
        
        # Group into optimal dubbing segments
        dubbing_segments = group_words_into_dubbing_segments(segments, target_duration=3.0)
        print(f"Created {len(dubbing_segments)} dubbing segments")
        
        # Translate each segment with GPT-4o
        update_job_status(job_id, "translating_with_gpt4o", 35)
        
        translated_segments = []
        context = ""
        
        for i, segment in enumerate(dubbing_segments):
            progress = 35 + (20 * i / len(dubbing_segments))
            update_job_status(job_id, f"translating_segment_{i+1}_of_{len(dubbing_segments)}", int(progress))
            
            # Smart translation with timing constraints
            translated_text = await smart_translate_segment(segment, target_language, context)
            translated_segments.append({
                "segment": segment,
                "translation": translated_text
            })
            
            # Update context for next segment
            context += f" {translated_text}"
        
        # Generate dubbed audio for each segment
        update_job_status(job_id, "generating_synchronized_dubbing", 55)
        
        audio_segments = []
        for i, item in enumerate(translated_segments):
            segment = item["segment"]
            translation = item["translation"]
            
            progress = 55 + (30 * i / len(translated_segments))
            update_job_status(job_id, f"dubbing_segment_{i+1}_of_{len(translated_segments)}", int(progress))
            
            segment_output_path = os.path.join(temp_dir, f"segment_{i:04d}.wav")
            
            try:
                await generate_dubbed_segment(
                    segment,
                    translation,
                    target_language,
                    audio_path,
                    segment_output_path
                )
                
                # Load and position audio at exact timing
                audio_seg = AudioFileClip(segment_output_path)
                audio_seg = audio_seg.set_start(segment.start_time)
                audio_segments.append(audio_seg)
                
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                # Continue with other segments
        
        if not audio_segments:
            raise DubbingError("segment_generation", "No audio segments were generated")
        
        # Create perfectly synchronized final audio
        update_job_status(job_id, "creating_synchronized_audio", 90)
        
        # Method: Create a silent base track and overlay segments
        from moviepy.audio.AudioClip import AudioClip
        
        # Create segments with proper timing
        final_segments = []
        
        for i, audio_seg in enumerate(audio_segments):
            # Load the audio segment (it already has its start time set)
            final_segments.append(audio_seg)
        
        # Create the final audio using CompositeAudioClip with proper initialization
        from moviepy.audio.AudioClip import CompositeAudioClip
        
        # Create a silent background track
        silent_audio = AudioClip(lambda t: 0, duration=original_duration)
        silent_audio.fps = 44100
        
        # Combine all segments with the silent background
        all_clips = [silent_audio] + final_segments
        final_audio = CompositeAudioClip(all_clips)
        
        # Save final audio
        temp_final_audio = os.path.join(temp_dir, "final_dubbed_audio.wav")
        final_audio.write_audiofile(temp_final_audio, fps=44100, logger=None)
        
        # Create final video
        update_job_status(job_id, "creating_final_video", 95)
        
        output_filename = f"{os.path.splitext(filename)[0]}_dubbed_{target_language}_perfect_sync.mp4"
        output_path = os.path.join(temp_dir, output_filename)
        
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

# API Endpoints
@app.get("/")
def read_root():
    return {
        "message": "Polydub v4.0 - Perfect Sync Edition",
        "features": [
            "WhisperX word-level transcription",
            "GPT-4o smart translation",
            "XTTS voice cloning",
            "Perfect synchronization"
        ],
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
        # Check file size
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
        
        # Start perfect sync processing
        background_tasks.add_task(
            process_video_with_perfect_sync,
            job_id,
            file_path,
            file.filename,
            target_language
        )
        
        return {
            "job_id": job_id,
            "message": "Video processing started with perfect synchronization",
            "features": {
                "transcription": "WhisperX (word-level)",
                "translation": "GPT-4o (smart)",
                "voice": "XTTS (cloned)",
                "sync": "Perfect word-level"
            },
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
