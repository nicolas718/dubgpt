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
        self.whisperx_model = "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"
        self.gpt4o_model = "openai/gpt-4o"
        self.xtts_model = "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e"
        self.latentsync_model = "bytedance/latentsync:9d95ee5d66c993bbd3e0779dacd2dd6af6f542de93403aae36c6343455e0ca04"
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "500"))

settings = Settings()

SUPPORTED_LANGUAGES = ["es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "en", "ar", "hi"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

LANGUAGE_NAMES = {
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "en": "English", "ar": "Arabic", "hi": "Hindi"
}

job_status: Dict[str, Dict] = {}

class DubbingError(Exception):
    def __init__(self, stage: str, detail: str):
        self.stage = stage
        self.detail = detail
        super().__init__(f"{stage}: {detail}")

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

app = FastAPI(title="Polydub API", version="7.0", lifespan=lifespan)

def update_job_status(job_id: str, status: str, progress: int = 0, 
                     result: Optional[str] = None, error: Optional[str] = None):
    job_status[job_id] = {
        "status": status,
        "progress": progress,
        "result": result,
        "error": error
    }

def upload_to_tmpfiles(file_path: str) -> Optional[str]:
    """Upload file to tmpfiles.org for temporary hosting"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('https://tmpfiles.org/api/v1/upload', files=files)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    url = data['data']['url']
                    direct_url = url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                    return direct_url
                    
    except Exception as e:
        print(f"tmpfiles upload failed: {e}")
    return None

def extract_word_timings_from_whisperx(whisperx_output: dict) -> List[TimedSegment]:
    segments = []
    
    if isinstance(whisperx_output, dict) and "segments" in whisperx_output:
        for idx, seg in enumerate(whisperx_output["segments"]):
            words = []
            
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
    
    print(f"Extracted {len(segments)} segments from WhisperX")
    return segments

def group_words_into_dubbing_segments(segments: List[TimedSegment], target_duration: float = 4.0) -> List[TimedSegment]:
    """Group words into segments with clean boundaries - prioritizing natural speech"""
    dubbing_segments = []
    
    # Collect all words with timing validation
    all_words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                # Validate word timings
                if word.start >= 0 and word.end > word.start:
                    all_words.append(word)
        else:
            if segment.start_time >= 0 and segment.end_time > segment.start_time:
                word = WordTiming(
                    word=segment.text,
                    start=segment.start_time,
                    end=segment.end_time
                )
                all_words.append(word)
    
    if not all_words:
        return segments
    
    # Sort and fix any timing overlaps
    all_words.sort(key=lambda w: w.start)
    
    # Fix overlapping words with minimal adjustment
    for i in range(1, len(all_words)):
        if all_words[i].start < all_words[i-1].end:
            # Small gap to prevent overlap
            gap = 0.02  # 20ms gap
            all_words[i] = WordTiming(
                word=all_words[i].word,
                start=all_words[i-1].end + gap,
                end=max(all_words[i].end, all_words[i-1].end + gap * 2)
            )
    
    current_words = []
    current_start = None
    
    for i, word in enumerate(all_words):
        if current_start is None:
            current_start = word.start
        
        current_words.append(word)
        duration = word.end - current_start
        
        should_segment = False
        
        word_text = word.word.strip()
        
        # Natural break points - prioritize sentence boundaries
        if word_text.endswith(('.', '!', '?')) and duration >= 1.5:
            should_segment = True
        elif word_text.endswith((',', ';', ':')) and duration >= 2.5:
            should_segment = True
        elif duration >= target_duration:  # Use full target duration
            should_segment = True
        elif i < len(all_words) - 1:
            next_word = all_words[i + 1]
            pause_duration = next_word.start - word.end
            # Natural pause detection
            if pause_duration > 0.4 and duration >= 1.0:
                should_segment = True
        elif i == len(all_words) - 1:
            should_segment = True
        
        if should_segment and current_words:
            # Create segment with timing buffer for natural speech
            segment_start = current_start
            segment_end = current_words[-1].end
            
            # Add small buffer to ensure no cutoff
            buffer = 0.05  # 50ms buffer (reduced from 100ms)
            if i < len(all_words) - 1:
                next_start = all_words[i + 1].start
                max_end = next_start - 0.03  # Small gap before next
                segment_end = min(segment_end + buffer, max_end)
            else:
                segment_end += buffer  # Buffer for last segment
            
            text = ' '.join([w.word for w in current_words])
            dubbing_segment = TimedSegment(
                text=text.strip(),
                start_time=segment_start,
                end_time=segment_end,
                duration=segment_end - segment_start,
                words=current_words.copy()
            )
            
            if dubbing_segment.duration > 0.3:  # Skip very short segments
                dubbing_segments.append(dubbing_segment)
            
            current_words = []
            current_start = None
    
    print(f"Created {len(dubbing_segments)} dubbing segments optimized for natural speech")
    return dubbing_segments

async def smart_translate_segment(segment: TimedSegment, target_language: str, context: str = "") -> str:
    """Translate for dubbing - accurate translation that fits timing"""
    
    # Language expansion factors
    expansion_factors = {
        "es": 1.20, "fr": 1.25, "de": 1.15, "it": 1.15,
        "pt": 1.20, "ru": 0.85, "ja": 0.65, "ko": 0.70,
        "zh": 0.45, "ar": 1.05, "hi": 1.10
    }
    
    expansion = expansion_factors.get(target_language, 1.15)
    
    # Target for natural pacing
    safe_duration = segment.duration * 0.85  # Use 85% of time
    target_chars = int(len(segment.text) / expansion * 0.85)
    max_chars = int(target_chars * 1.1)  # Strict maximum
    
    # Clean the text
    clean_text = segment.text.strip()
    clean_text = ' '.join(clean_text.split())  # Remove extra spaces
    
    prompt = f"""Translate this English text to {LANGUAGE_NAMES.get(target_language, target_language)} for video dubbing.

Original: "{clean_text}"
Duration: {segment.duration:.1f} seconds
Character limit: {max_chars} characters maximum

Requirements:
1. Accurate translation in proper {target_language}
2. Must fit within {max_chars} characters
3. Keep the meaning and emotion intact
4. Natural spoken {target_language} - how a native speaker would say it
5. Remove filler words (um, uh, well)

Context: {context[-150:] if context else 'Start of video'}

Provide ONLY the {target_language} translation, nothing else."""

    try:
        output = replicate.run(
            settings.gpt4o_model,
            input={
                "prompt": prompt,
                "temperature": 0.3,
                "system_prompt": f"You are an expert {target_language} dubbing specialist. Create natural, conversational translations that fit time constraints perfectly. Never sacrifice natural flow for literal accuracy."
            }
        )
        
        if not output:
            return clean_text[:target_chars]
        
        translated = ''.join(output).strip()
        
        # Enforce length limit
        if len(translated) > max_chars:
            # Smart truncation
            sentences = translated.split('. ')
            result = ""
            for sent in sentences:
                if len(result) + len(sent) + 2 <= max_chars:
                    result += sent + ". "
                else:
                    break
            translated = result.strip() or translated[:max_chars]
        
        return translated
        
    except Exception as e:
        print(f"Translation error: {e}")
        return clean_text[:target_chars]

async def generate_dubbed_segment(
    segment: TimedSegment,
    translated_text: str,
    target_language: str,
    speaker_audio_path: str,
    output_path: str
) -> str:
    """Generate dubbed audio with minimal speed adjustment"""
    try:
        # Generate audio with XTTS
        with open(speaker_audio_path, "rb") as audio_file:
            output = replicate.run(
                settings.xtts_model,
                input={
                    "text": translated_text,
                    "speaker": audio_file,
                    "language": target_language
                }
            )
        
        # Get audio URL
        if isinstance(output, str):
            audio_url = output
        elif isinstance(output, list) and len(output) > 0:
            audio_url = output[0]
        else:
            raise ValueError("Invalid XTTS output")
        
        # Download audio
        temp_path = output_path + "_temp.wav"
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Check duration
        audio_clip = AudioFileClip(temp_path)
        actual_duration = audio_clip.duration
        audio_clip.close()
        
        # Handle duration mismatch
        if actual_duration > segment.duration:
            speed_factor = actual_duration / segment.duration
            
            # Maximum 15% speedup for natural sound
            if speed_factor > 1.15:
                print(f"WARNING: Segment {output_path} needs {speed_factor:.2f}x speed")
                speed_factor = 1.15
            
            # Apply speed adjustment
            cmd = [
                'ffmpeg', '-i', temp_path,
                '-filter:a', f'atempo={speed_factor}',
                '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg speed adjustment failed: {result.stderr}")
                shutil.move(temp_path, output_path)
            else:
                os.remove(temp_path)
                
                # Verify final duration and hard trim if needed
                final_clip = AudioFileClip(output_path)
                final_duration = final_clip.duration
                final_clip.close()
                
                if final_duration > segment.duration + 0.01:
                    # Hard trim with fade
                    fade_duration = min(0.05, segment.duration * 0.1)
                    fade_start = segment.duration - fade_duration
                    
                    trim_cmd = [
                        'ffmpeg', '-i', output_path,
                        '-t', str(segment.duration),
                        '-af', f'afade=out=st={fade_start}:d={fade_duration}',
                        '-y', output_path + '_trim.wav'
                    ]
                    
                    subprocess.run(trim_cmd, capture_output=True)
                    
                    if os.path.exists(output_path + '_trim.wav'):
                        os.remove(output_path)
                        os.rename(output_path + '_trim.wav', output_path)
        else:
            # Audio fits perfectly
            shutil.move(temp_path, output_path)
        
        return output_path
        
    except Exception as e:
        raise DubbingError("tts_generation", str(e))

async def apply_lip_sync(
    video_path: str,
    audio_path: str,
    output_path: str
) -> str:
    """Apply lip sync using LatentSync"""
    
    try:
        print("\nApplying lip sync...")
        
        # Upload files
        video_url = upload_to_tmpfiles(video_path)
        if not video_url:
            raise ValueError("Failed to upload video")
            
        audio_url = upload_to_tmpfiles(audio_path)
        if not audio_url:
            raise ValueError("Failed to upload audio")
        
        # Run LatentSync
        output = replicate.run(
            settings.latentsync_model,
            input={
                "video": video_url,
                "audio": audio_url
            }
        )
        
        # Download result
        if hasattr(output, 'read'):
            with open(output_path, 'wb') as f:
                f.write(output.read())
        elif isinstance(output, str) and output.startswith('http'):
            response = requests.get(output)
            with open(output_path, 'wb') as f:
                f.write(response.content)
        else:
            raise ValueError(f"Unexpected output type: {type(output)}")
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            raise ValueError("Lip sync output is empty")
        
    except Exception as e:
        print(f"Lip sync error: {e}")
        raise DubbingError("lip_sync", str(e))

async def process_video_with_perfect_sync(
    job_id: str,
    file_path: str,
    filename: str,
    target_language: str
):
    """Main processing pipeline with refined synchronization"""
    temp_dir = os.path.dirname(file_path)
    
    try:
        # Extract audio
        update_job_status(job_id, "extracting_audio", 10)
        
        audio_path = os.path.join(temp_dir, "audio.wav")
        video = VideoFileClip(file_path)
        original_duration = video.duration
        
        # Extract audio with consistent format
        video.audio.write_audiofile(
            audio_path, 
            codec='pcm_s16le',
            fps=16000,
            nbytes=2,
            logger=None
        )
        
        # Transcribe
        update_job_status(job_id, "transcribing", 20)
        
        with open(audio_path, "rb") as audio_file:
            whisperx_output = replicate.run(
                settings.whisperx_model,
                input={
                    "audio_file": audio_file,
                    "batch_size": 16,
                    "align_output": True,
                    "diarization": False
                }
            )
        
        segments = extract_word_timings_from_whisperx(whisperx_output)
        if not segments:
            raise DubbingError("transcription", "No segments extracted")
        
        # Use shorter target duration for better control
        dubbing_segments = group_words_into_dubbing_segments(segments, target_duration=3.5)
        
        # Translate
        update_job_status(job_id, "translating", 35)
        
        translated_segments = []
        context = ""
        
        for i, segment in enumerate(dubbing_segments):
            progress = 35 + (15 * i / len(dubbing_segments))
            update_job_status(job_id, f"translating_{i+1}", int(progress))
            
            translated_text = await smart_translate_segment(segment, target_language, context)
            translated_segments.append({
                "segment": segment,
                "translation": translated_text
            })
            context = translated_text
        
        # Generate audio
        update_job_status(job_id, "generating_audio", 50)
        
        audio_segments = []
        segment_files = []
        
        for i, item in enumerate(translated_segments):
            segment = item["segment"]
            translation = item["translation"]
            
            if not translation.strip():
                continue
            
            progress = 50 + (25 * i / len(translated_segments))
            update_job_status(job_id, f"dubbing_{i+1}", int(progress))
            
            segment_path = os.path.join(temp_dir, f"seg_{i:04d}.wav")
            
            try:
                await generate_dubbed_segment(
                    segment,
                    translation,
                    target_language,
                    audio_path,
                    segment_path
                )
                
                if os.path.exists(segment_path):
                    segment_files.append(segment_path)
                    
                    # Load and position audio with validation
                    audio_seg = AudioFileClip(segment_path)
                    
                    # Ensure segment fits in its time slot
                    max_duration = segment.duration
                    if audio_seg.duration > max_duration:
                        audio_seg = audio_seg.subclip(0, max_duration)
                    
                    # Set precise start time
                    audio_seg = audio_seg.set_start(segment.start_time)
                    
                    # Add fade in/out to prevent pops
                    fade_duration = min(0.05, audio_seg.duration * 0.1)
                    audio_seg = audio_seg.audio_fadein(fade_duration).audio_fadeout(fade_duration)
                    
                    audio_segments.append(audio_seg)
                    
            except Exception as e:
                print(f"Segment {i} error: {e}")
                continue
        
        if not audio_segments:
            raise DubbingError("audio_generation", "No audio segments generated")
        
        # Combine audio with gap filling
        update_job_status(job_id, "combining_audio", 75)
        
        # Create silence base
        base_audio = AudioClip(lambda t: 0, duration=original_duration)
        base_audio = base_audio.set_fps(16000)
        
        # Composite with precise timing
        final_audio = CompositeAudioClip([base_audio] + audio_segments)
        final_audio = final_audio.set_fps(16000)
        
        temp_audio_path = os.path.join(temp_dir, "dubbed_audio.wav")
        final_audio.write_audiofile(
            temp_audio_path,
            fps=16000,
            codec='pcm_s16le',
            nbytes=2,
            logger=None
        )
        
        # Create video
        update_job_status(job_id, "creating_video", 85)
        
        temp_video_path = os.path.join(temp_dir, "dubbed_video.mp4")
        
        final_audio_clip = AudioFileClip(temp_audio_path)
        final_video = video.set_audio(final_audio_clip)
        
        # Write with consistent settings
        final_video.write_videofile(
            temp_video_path,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="128k",
            bitrate="5000k",
            fps=video.fps,
            logger=None
        )
        
        # Cleanup
        video.close()
        final_audio_clip.close()
        final_video.close()
        final_audio.close()
        base_audio.close()
        
        for seg in audio_segments:
            try:
                seg.close()
            except:
                pass
        
        # Apply lip sync
        update_job_status(job_id, "applying_lip_sync", 90)
        
        output_filename = f"{os.path.splitext(filename)[0]}_dubbed_{target_language}.mp4"
        output_path = os.path.join(temp_dir, output_filename)
        
        try:
            await apply_lip_sync(
                video_path=file_path,
                audio_path=temp_audio_path,
                output_path=output_path
            )
        except Exception as e:
            print(f"Lip sync failed: {e}, using dubbed video")
            shutil.move(temp_video_path, output_path)
        
        # Cleanup temp files
        for seg_file in segment_files:
            try:
                os.remove(seg_file)
            except:
                pass
        
        update_job_status(job_id, "completed", 100, result=output_path)
        
    except DubbingError as e:
        update_job_status(job_id, "failed", error=f"{e.stage}: {e.detail}")
    except Exception as e:
        update_job_status(job_id, "failed", error=str(e))

@app.get("/")
def read_root():
    return {
        "message": "Polydub v7.0 - Refined Video Dubbing",
        "features": [
            "WhisperX transcription",
            "Natural translation (quality over literal)",
            "XTTS voice cloning",
            "Minimal speed adjustment (max 20%)",
            "LatentSync lip synchronization"
        ],
        "endpoints": {
            "/upload": "POST - Upload video",
            "/status/{job_id}": "GET - Check status",
            "/download/{job_id}": "GET - Download result",
            "/languages": "GET - Supported languages",
            "/formats": "GET - Supported formats"
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
    file_size = 0
    temp_dir = None
    
    try:
        # Read file
        temp_file = tempfile.SpooledTemporaryFile(max_size=1024*1024)
        while chunk := await file.read(1024*1024):
            file_size += len(chunk)
            temp_file.write(chunk)
        temp_file.seek(0)
        
        # Validate
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in SUPPORTED_VIDEO_FORMATS:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        if target_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail="Unsupported language")
        
        if file_size > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Create job
        job_id = str(uuid.uuid4())
        update_job_status(job_id, "uploading", 0)
        
        # Save file
        temp_dir = tempfile.mkdtemp(prefix=f"polydub_{job_id}_")
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, 'wb') as f:
            temp_file.seek(0)
            shutil.copyfileobj(temp_file, f)
        temp_file.close()
        
        # Start processing
        background_tasks.add_task(
            process_video_with_perfect_sync,
            job_id,
            file_path,
            file.filename,
            target_language
        )
        
        return {
            "job_id": job_id,
            "message": "Processing started",
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
        raise HTTPException(status_code=400, detail=f"Job not completed")
    
    if not status["result"] or not os.path.exists(status["result"]):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        status["result"], 
        media_type="video/mp4",
        filename=os.path.basename(status["result"])
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_jobs": len([j for j in job_status.values() if j["status"] not in ["completed", "failed"]])
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
