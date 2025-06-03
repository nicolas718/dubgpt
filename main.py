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
    
    print(f"WhisperX output type: {type(whisperx_output)}")
    print(f"WhisperX output keys: {whisperx_output.keys() if isinstance(whisperx_output, dict) else 'Not a dict'}")
    
    if isinstance(whisperx_output, dict) and "segments" in whisperx_output:
        print(f"Found {len(whisperx_output['segments'])} segments in WhisperX output")
        
        for idx, seg in enumerate(whisperx_output["segments"]):
            words = []
            
            # Debug segment
            print(f"Segment {idx}: {seg.keys()}")
            print(f"  Text: {seg.get('text', '')[:100]}")
            
            # WhisperX provides word-level timing
            if "words" in seg:
                print(f"  Found {len(seg['words'])} words")
                for word_data in seg["words"]:
                    word = WordTiming(
                        word=word_data.get("word", ""),
                        start=word_data.get("start", 0),
                        end=word_data.get("end", 0)
                    )
                    words.append(word)
            else:
                print(f"  No word-level timing in segment {idx}")
            
            segment = TimedSegment(
                text=seg.get("text", "").strip(),
                start_time=seg.get("start", 0),
                end_time=seg.get("end", 0),
                duration=seg.get("end", 0) - seg.get("start", 0),
                words=words
            )
            
            if segment.text:
                segments.append(segment)
    else:
        print("ERROR: WhisperX output doesn't have expected format")
        print(f"Output: {str(whisperx_output)[:500]}")
    
    print(f"Total segments extracted: {len(segments)}")
    return segments

def group_words_into_dubbing_segments(segments: List[TimedSegment], target_duration: float = 4.0) -> List[TimedSegment]:
    """Group words into optimal segments for dubbing - respecting natural speech patterns"""
    dubbing_segments = []
    
    # First, combine all words from all segments
    all_words = []
    for segment in segments:
        if segment.words:
            all_words.extend(segment.words)
        else:
            # If no word-level timing, treat the whole segment as one "word"
            word = WordTiming(
                word=segment.text,
                start=segment.start_time,
                end=segment.end_time
            )
            all_words.append(word)
    
    if not all_words:
        return segments
    
    # Sort words by start time
    all_words.sort(key=lambda w: w.start)
    
    # Group words into segments based on natural breaks and timing
    current_words = []
    current_start = None
    
    for i, word in enumerate(all_words):
        if current_start is None:
            current_start = word.start
        
        current_words.append(word)
        duration = word.end - current_start
        
        # Determine if we should create a segment
        should_segment = False
        reason = ""
        
        # Check for natural break points
        word_text = word.word.strip()
        
        # Strong punctuation - always break if we have enough content
        if word_text.endswith(('.', '!', '?')) and duration >= 1.5:
            should_segment = True
            reason = "sentence_end"
        
        # Comma or semicolon - break if segment is getting long
        elif word_text.endswith((',', ';', ':')) and duration >= 3.0:
            should_segment = True
            reason = "clause_end"
        
        # Maximum duration reached - try to find a good break point
        elif duration >= target_duration:
            # Look ahead for the next good break point (up to 1 second)
            if i < len(all_words) - 1:
                for j in range(i + 1, min(i + 5, len(all_words))):
                    future_word = all_words[j]
                    future_duration = future_word.end - current_start
                    if future_duration > target_duration + 1.0:
                        break
                    if future_word.word.strip().endswith(('.', '!', '?', ',', ';', ':')):
                        # Found a better break point nearby
                        should_segment = False
                        break
                else:
                    should_segment = True
                    reason = "max_duration"
            else:
                should_segment = True
                reason = "max_duration"
        
        # Long pause after word (>0.4s) and we have reasonable content
        elif i < len(all_words) - 1:
            next_word = all_words[i + 1]
            pause_duration = next_word.start - word.end
            if pause_duration > 0.4 and duration >= 1.5:
                should_segment = True
                reason = "natural_pause"
        
        # Last word
        elif i == len(all_words) - 1:
            should_segment = True
            reason = "last_word"
        
        if should_segment and current_words:
            text = ' '.join([w.word for w in current_words])
            dubbing_segment = TimedSegment(
                text=text.strip(),
                start_time=current_start,
                end_time=word.end,
                duration=word.end - current_start,
                words=current_words.copy()
            )
            dubbing_segments.append(dubbing_segment)
            
            if reason:
                print(f"Segment break ({reason}): {dubbing_segment.duration:.1f}s - '{text[:40]}...'")
            
            current_words = []
            current_start = None
    
    print(f"Created {len(dubbing_segments)} dubbing segments from {len(all_words)} words")
    
    # Analyze segment distribution
    durations = [seg.duration for seg in dubbing_segments]
    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        print(f"Segment durations: avg={avg_duration:.1f}s, min={min_duration:.1f}s, max={max_duration:.1f}s")
    
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
    
    # Limit speed adjustment to natural range (0.9x to 1.2x)
    max_speed_factor = 1.2
    min_speed_factor = 0.9
    
    # Calculate realistic character limit
    natural_chars = int(len(segment.text) / expansion)
    max_chars = int(natural_chars * max_speed_factor)
    min_chars = int(natural_chars * min_speed_factor)
    
    prompt = f"""Translate this text to {LANGUAGE_NAMES.get(target_language, target_language)}.

Original text: "{segment.text}"
Duration available: {segment.duration:.1f} seconds
Target character count: {natural_chars} characters (range: {min_chars}-{max_chars})

CRITICAL Requirements:
1. Translation must sound NATURAL when spoken at normal speed
2. Aim for {natural_chars} characters to avoid speed adjustment
3. Use contractions and shorter phrases if needed
4. Remove filler words if necessary
5. NEVER sacrifice meaning for brevity
6. If impossible to fit naturally, prefer slight speed-up over cut content

Context: {context[-200:] if context else 'Start of video'}

Return ONLY the translated text."""

    try:
        output = replicate.run(
            settings.gpt4o_model,
            input={
                "prompt": prompt,
                "max_tokens": 150,
                "temperature": 0.3,
                "system_prompt": "You are a professional translator specializing in video dubbing. Create translations that sound natural when spoken, matching the original timing as closely as possible."
            }
        )
        
        translated_text = ''.join(output).strip()
        
        # Log translation for debugging
        print(f"Segment ({segment.duration:.1f}s): '{segment.text[:30]}...' -> '{translated_text[:30]}...' ({len(translated_text)} chars)")
        
        return translated_text
        
    except Exception as e:
        print(f"GPT-4o translation failed: {e}, using fallback")
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
        # Generate TTS with XTTS - simple and consistent
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
        
        # Limit speed adjustment to natural range
        MAX_SPEED = 1.25  # No more than 25% faster
        MIN_SPEED = 0.85  # No more than 15% slower
        
        if abs(actual_duration - segment.duration) > 0.05:  # 50ms tolerance
            speed_factor = actual_duration / segment.duration
            
            # Log extreme speed adjustments
            if speed_factor > MAX_SPEED or speed_factor < MIN_SPEED:
                print(f"WARNING: Segment requires {speed_factor:.2f}x speed adjustment")
                print(f"  Original: '{segment.text[:50]}...' ({segment.duration:.1f}s)")
                print(f"  Translation: '{translated_text[:50]}...' ({actual_duration:.1f}s)")
            
            # Clamp speed factor to acceptable range
            speed_factor = max(MIN_SPEED, min(MAX_SPEED, speed_factor))
            
            # Check if ffmpeg is available
            try:
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
                if result.returncode == 0:
                    # Use FFmpeg for speed adjustment
                    cmd = [
                        'ffmpeg', '-i', temp_output,
                        '-filter:a', f'atempo={speed_factor}',
                        '-y', output_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        os.remove(temp_output)
                        
                        # Verify the adjusted duration
                        adjusted_clip = AudioFileClip(output_path)
                        final_duration = adjusted_clip.duration
                        adjusted_clip.close()
                        
                        # If still too long/short after max adjustment, trim/pad
                        if final_duration > segment.duration + 0.1:
                            print(f"  Trimming audio from {final_duration:.1f}s to {segment.duration:.1f}s")
                            # Trim the audio
                            cmd = [
                                'ffmpeg', '-i', output_path,
                                '-t', str(segment.duration),
                                '-y', output_path + '_trimmed.wav'
                            ]
                            subprocess.run(cmd, capture_output=True)
                            os.remove(output_path)
                            os.rename(output_path + '_trimmed.wav', output_path)
                    else:
                        print(f"FFmpeg adjustment failed: {result.stderr}")
                        shutil.move(temp_output, output_path)
                else:
                    print("FFmpeg not available, skipping speed adjustment")
                    shutil.move(temp_output, output_path)
            except FileNotFoundError:
                print("FFmpeg not installed, skipping speed adjustment")
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
        
        # If no word-level timing, fall back to segment-level
        total_words = sum(len(s.words) for s in segments)
        if total_words == 0:
            print("WARNING: No word-level timing found. Using segment-level timing instead.")
            # This is fine, we'll just process larger chunks
        
        # Group into optimal dubbing segments - respecting natural speech patterns
        dubbing_segments = group_words_into_dubbing_segments(segments, target_duration=4.0)
        print(f"Created {len(dubbing_segments)} dubbing segments")
        
        # Verify we have good coverage
        total_duration_covered = sum(seg.duration for seg in dubbing_segments)
        print(f"Total duration covered by segments: {total_duration_covered:.1f}s out of {original_duration:.1f}s")
        
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
                if os.path.exists(segment_output_path):
                    audio_seg = AudioFileClip(segment_output_path)
                    
                    # Debug: Verify audio segment
                    print(f"Loaded segment {i}: duration={audio_seg.duration}s, start_time={segment.start_time}s")
                    print(f"  Text: '{segment.text[:50]}...'")
                    print(f"  Translation: '{translation[:50]}...'")
                    
                    # Set the start time for this segment
                    audio_seg = audio_seg.set_start(segment.start_time)
                    audio_segments.append(audio_seg)
                else:
                    print(f"WARNING: Segment file not found: {segment_output_path}")
                
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                # Continue with other segments
        
        if not audio_segments:
            raise DubbingError("segment_generation", "No audio segments were generated")
        
        # Create perfectly synchronized final audio
        update_job_status(job_id, "creating_synchronized_audio", 90)
        
        # Debug: Print segment information
        print(f"Total segments to combine: {len(audio_segments)}")
        for i, seg in enumerate(audio_segments):
            print(f"Segment {i}: start={seg.start}, duration={seg.duration}")
        
        if not audio_segments:
            raise DubbingError("audio_generation", "No audio segments to combine")
        
        # Alternative approach: Save each segment at its correct position
        # First, let's check what we actually have
        total_audio_duration = sum(seg.duration for seg in audio_segments)
        print(f"Total audio duration: {total_audio_duration}s (should be close to {original_duration}s)")
        
        # If we have very little audio, something went wrong earlier
        if total_audio_duration < original_duration * 0.5:
            print("WARNING: Less than 50% of video has audio. Checking for issues...")
            # Log what segments we're missing
            covered_time = 0
            for seg in audio_segments:
                if seg.start > covered_time + 1:
                    print(f"  GAP: No audio from {covered_time}s to {seg.start}s")
                covered_time = max(covered_time, seg.start + seg.duration)
            if covered_time < original_duration:
                print(f"  GAP: No audio from {covered_time}s to {original_duration}s")
        
        # Method: Create a base audio track and add segments
        from moviepy.audio.AudioClip import AudioClip
        
        # Sort segments by start time
        audio_segments.sort(key=lambda x: x.start)
        
        # Create the final audio by overlaying all segments on a silent track
        # This preserves the exact timing of each segment
        base_audio = AudioClip(lambda t: 0, duration=original_duration)
        base_audio.fps = 44100
        
        # Create a composite with all segments
        from moviepy.editor import CompositeAudioClip
        all_clips = [base_audio] + audio_segments
        final_audio = CompositeAudioClip(all_clips)
        
        # Save final audio
        temp_final_audio = os.path.join(temp_dir, "final_dubbed_audio.wav")
        print(f"Writing final audio to {temp_final_audio}")
        final_audio.write_audiofile(temp_final_audio, fps=44100, logger=None)
        
        # Verify the audio file
        if os.path.exists(temp_final_audio):
            file_size = os.path.getsize(temp_final_audio)
            print(f"Final audio file size: {file_size} bytes")
            audio_check = AudioFileClip(temp_final_audio)
            print(f"Final audio duration: {audio_check.duration}s")
            audio_check.close()
        
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
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
