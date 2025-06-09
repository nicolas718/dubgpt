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

app = FastAPI(title="Polydub API", version="5.0", lifespan=lifespan)

def update_job_status(job_id: str, status: str, progress: int = 0, 
                     result: Optional[str] = None, error: Optional[str] = None):
    job_status[job_id] = {
        "status": status,
        "progress": progress,
        "result": result,
        "error": error
    }

def extract_word_timings_from_whisperx(whisperx_output: dict) -> List[TimedSegment]:
    segments = []
    
    print(f"WhisperX output type: {type(whisperx_output)}")
    print(f"WhisperX output keys: {whisperx_output.keys() if isinstance(whisperx_output, dict) else 'Not a dict'}")
    
    if isinstance(whisperx_output, dict) and "segments" in whisperx_output:
        print(f"Found {len(whisperx_output['segments'])} segments in WhisperX output")
        
        for idx, seg in enumerate(whisperx_output["segments"]):
            words = []
            
            print(f"Segment {idx}: {seg.keys()}")
            print(f"  Text: {seg.get('text', '')[:100]}")
            
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
    dubbing_segments = []
    
    all_words = []
    for segment in segments:
        if segment.words:
            all_words.extend(segment.words)
        else:
            word = WordTiming(
                word=segment.text,
                start=segment.start_time,
                end=segment.end_time
            )
            all_words.append(word)
    
    if not all_words:
        return segments
    
    all_words.sort(key=lambda w: w.start)
    
    current_words = []
    current_start = None
    
    for i, word in enumerate(all_words):
        if current_start is None:
            current_start = word.start
        
        current_words.append(word)
        duration = word.end - current_start
        
        should_segment = False
        reason = ""
        
        word_text = word.word.strip()
        
        if word_text.endswith(('.', '!', '?')) and duration >= 1.5:
            should_segment = True
            reason = "sentence_end"
        
        elif word_text.endswith((',', ';', ':')) and duration >= 3.0:
            should_segment = True
            reason = "clause_end"
        
        elif duration >= target_duration:
            if i < len(all_words) - 1:
                for j in range(i + 1, min(i + 5, len(all_words))):
                    future_word = all_words[j]
                    future_duration = future_word.end - current_start
                    if future_duration > target_duration + 1.0:
                        break
                    if future_word.word.strip().endswith(('.', '!', '?', ',', ';', ':')):
                        should_segment = False
                        break
                else:
                    should_segment = True
                    reason = "max_duration"
            else:
                should_segment = True
                reason = "max_duration"
        
        elif i < len(all_words) - 1:
            next_word = all_words[i + 1]
            pause_duration = next_word.start - word.end
            if pause_duration > 0.4 and duration >= 1.5:
                should_segment = True
                reason = "natural_pause"
        
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
    
    durations = [seg.duration for seg in dubbing_segments]
    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        print(f"Segment durations: avg={avg_duration:.1f}s, min={min_duration:.1f}s, max={max_duration:.1f}s")
    
    return dubbing_segments

async def smart_translate_segment(segment: TimedSegment, target_language: str, context: str = "") -> str:
    word_count = len(segment.text.split())
    chars_per_second = len(segment.text) / segment.duration if segment.duration > 0 else 30
    
    expansion_factors = {
        "es": 1.25, "fr": 1.30, "de": 1.20, "it": 1.20,
        "pt": 1.25, "ru": 0.90, "ja": 0.70, "ko": 0.75,
        "zh": 0.50, "ar": 1.10, "hi": 1.15
    }
    
    expansion = expansion_factors.get(target_language, 1.2)
    
    max_speed_factor = 1.2
    min_speed_factor = 0.9
    
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
        
        temp_output = output_path + "_temp.wav"
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_output, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        audio_clip = AudioFileClip(temp_output)
        actual_duration = audio_clip.duration
        audio_clip.close()
        
        MAX_SPEED = 1.25
        MIN_SPEED = 0.85
        
        if abs(actual_duration - segment.duration) > 0.05:
            speed_factor = actual_duration / segment.duration
            
            if speed_factor > MAX_SPEED or speed_factor < MIN_SPEED:
                print(f"WARNING: Segment requires {speed_factor:.2f}x speed adjustment")
                print(f"  Original: '{segment.text[:50]}...' ({segment.duration:.1f}s)")
                print(f"  Translation: '{translated_text[:50]}...' ({actual_duration:.1f}s)")
            
            speed_factor = max(MIN_SPEED, min(MAX_SPEED, speed_factor))
            
            try:
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
                if result.returncode == 0:
                    cmd = [
                        'ffmpeg', '-i', temp_output,
                        '-filter:a', f'atempo={speed_factor}',
                        '-y', output_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        os.remove(temp_output)
                        
                        adjusted_clip = AudioFileClip(output_path)
                        final_duration = adjusted_clip.duration
                        adjusted_clip.close()
                        
                        if final_duration > segment.duration + 0.1:
                            print(f"  Trimming audio from {final_duration:.1f}s to {segment.duration:.1f}s")
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

async def apply_lip_sync(
    video_path: str,
    audio_path: str,
    output_path: str
) -> str:
    """Apply lip sync to dubbed video using LatentSync"""
    
    try:
        print("="*50)
        print("APPLYING LIP SYNC WITH LATENTSYNC")
        print("="*50)
        print(f"Video: {video_path} ({os.path.getsize(video_path) / 1024 / 1024:.1f} MB)")
        print(f"Audio: {audio_path} ({os.path.getsize(audio_path) / 1024 / 1024:.1f} MB)")
        
        # Check if files exist and are readable
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        # Run LatentSync with better error handling
        try:
            with open(video_path, "rb") as video_file, open(audio_path, "rb") as audio_file:
                print("Calling LatentSync API...")
                print(f"Model: {settings.latentsync_model}")
                
                # Try with explicit version
                output = replicate.run(
                    "bytedance/latentsync:9d95ee5d66c993bbd3e0779dacd2dd6af6f542de93403aae36c6343455e0ca04",
                    input={
                        "video": video_file,
                        "audio": audio_file
                    }
                )
                
                print(f"LatentSync API call successful!")
                print(f"Output type: {type(output)}")
                
        except Exception as api_error:
            print(f"LatentSync API error: {api_error}")
            print(f"Error type: {type(api_error)}")
            
            # Try alternative approach - using file paths instead of file objects
            print("\nTrying alternative approach with URLs...")
            
            # Upload files to a temporary URL service or use local file paths
            # For now, let's try with the Replicate file upload
            import replicate.file
            
            print("Uploading video to Replicate...")
            with open(video_path, "rb") as f:
                video_url = replicate.file.upload(f, "video.mp4")
            print(f"Video URL: {video_url}")
            
            print("Uploading audio to Replicate...")
            with open(audio_path, "rb") as f:
                audio_url = replicate.file.upload(f, "audio.wav")
            print(f"Audio URL: {audio_url}")
            
            # Try again with URLs
            output = replicate.run(
                "bytedance/latentsync:9d95ee5d66c993bbd3e0779dacd2dd6af6f542de93403aae36c6343455e0ca04",
                input={
                    "video": video_url,
                    "audio": audio_url
                }
            )
        
        print(f"LatentSync output type: {type(output)}")
        print(f"LatentSync output: {str(output)[:200]}...")  # First 200 chars
        
        # Handle output
        with open(output_path, "wb") as f:
            if hasattr(output, 'read'):
                print("Output is file-like, reading...")
                content = output.read()
                f.write(content)
                print(f"Wrote {len(content)} bytes")
            elif isinstance(output, bytes):
                print("Output is bytes")
                f.write(output)
                print(f"Wrote {len(output)} bytes")
            elif isinstance(output, str):
                print(f"Output is URL: {output}")
                response = requests.get(output)
                print(f"Download response status: {response.status_code}")
                response.raise_for_status()
                f.write(response.content)
                print(f"Downloaded {len(response.content)} bytes")
            else:
                # Try to iterate if it's a generator
                print(f"Output is {type(output)}, trying to iterate...")
                content = b""
                for chunk in output:
                    if isinstance(chunk, bytes):
                        content += chunk
                    elif isinstance(chunk, str):
                        content += chunk.encode()
                f.write(content)
                print(f"Wrote {len(content)} bytes from iterator")
        
        # Verify
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"LIP SYNC SUCCESS! Output: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
            
            # Verify it's a valid video
            try:
                test_video = VideoFileClip(output_path)
                print(f"Lip synced video duration: {test_video.duration}s")
                test_video.close()
            except Exception as e:
                print(f"Warning: Could not verify video: {e}")
            
            return output_path
        else:
            raise ValueError("Lip sync output file is empty or missing")
        
    except Exception as e:
        print(f"LIP SYNC FAILED: {e}")
        print(f"Full error: {traceback.format_exc()}")
        raise DubbingError("lip_sync", str(e))

async def process_video_with_perfect_sync(
    job_id: str,
    file_path: str,
    filename: str,
    target_language: str,
    enable_lip_sync: bool = False
):
    temp_dir = os.path.dirname(file_path)
    
    try:
        update_job_status(job_id, "extracting_audio", 10)
        
        audio_path = os.path.join(temp_dir, "audio.wav")
        video = VideoFileClip(file_path)
        original_duration = video.duration
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        
        update_job_status(job_id, "transcribing_with_whisperx", 20)
        
        try:
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
            
            print(f"WhisperX output type: {type(whisperx_output)}")
            
        except Exception as e:
            raise DubbingError("transcription", f"WhisperX failed: {str(e)}")
        
        segments = extract_word_timings_from_whisperx(whisperx_output)
        if not segments:
            raise DubbingError("transcription", "No segments extracted from WhisperX")
        
        print(f"Extracted {len(segments)} segments with {sum(len(s.words) for s in segments)} words")
        
        total_words = sum(len(s.words) for s in segments)
        if total_words == 0:
            print("WARNING: No word-level timing found. Using segment-level timing instead.")
        
        dubbing_segments = group_words_into_dubbing_segments(segments, target_duration=4.0)
        print(f"Created {len(dubbing_segments)} dubbing segments")
        
        total_duration_covered = sum(seg.duration for seg in dubbing_segments)
        print(f"Total duration covered by segments: {total_duration_covered:.1f}s out of {original_duration:.1f}s")
        
        update_job_status(job_id, "translating_with_gpt4o", 35)
        
        translated_segments = []
        context = ""
        
        for i, segment in enumerate(dubbing_segments):
            progress = 35 + (20 * i / len(dubbing_segments))
            update_job_status(job_id, f"translating_segment_{i+1}_of_{len(dubbing_segments)}", int(progress))
            
            translated_text = await smart_translate_segment(segment, target_language, context)
            translated_segments.append({
                "segment": segment,
                "translation": translated_text
            })
            
            context += f" {translated_text}"
        
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
                
                if os.path.exists(segment_output_path):
                    audio_seg = AudioFileClip(segment_output_path)
                    
                    print(f"Loaded segment {i}: duration={audio_seg.duration}s, start_time={segment.start_time}s")
                    print(f"  Text: '{segment.text[:50]}...'")
                    print(f"  Translation: '{translation[:50]}...'")
                    
                    audio_seg = audio_seg.set_start(segment.start_time)
                    audio_segments.append(audio_seg)
                else:
                    print(f"WARNING: Segment file not found: {segment_output_path}")
                
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
        
        if not audio_segments:
            raise DubbingError("segment_generation", "No audio segments were generated")
        
        update_job_status(job_id, "creating_synchronized_audio", 90)
        
        print(f"Total segments to combine: {len(audio_segments)}")
        for i, seg in enumerate(audio_segments):
            print(f"Segment {i}: start={seg.start}, duration={seg.duration}")
        
        total_audio_duration = sum(seg.duration for seg in audio_segments)
        print(f"Total audio duration: {total_audio_duration}s (should be close to {original_duration}s)")
        
        if total_audio_duration < original_duration * 0.5:
            print("WARNING: Less than 50% of video has audio. Checking for issues...")
            covered_time = 0
            for seg in audio_segments:
                if seg.start > covered_time + 1:
                    print(f"  GAP: No audio from {covered_time}s to {seg.start}s")
                covered_time = max(covered_time, seg.start + seg.duration)
            if covered_time < original_duration:
                print(f"  GAP: No audio from {covered_time}s to {original_duration}s")
        
        from moviepy.audio.AudioClip import AudioClip
        
        audio_segments.sort(key=lambda x: x.start)
        
        base_audio = AudioClip(lambda t: 0, duration=original_duration)
        base_audio.fps = 44100
        
        from moviepy.editor import CompositeAudioClip
        all_clips = [base_audio] + audio_segments
        final_audio = CompositeAudioClip(all_clips)
        
        temp_final_audio = os.path.join(temp_dir, "final_dubbed_audio.wav")
        print(f"Writing final audio to {temp_final_audio}")
        final_audio.write_audiofile(temp_final_audio, fps=44100, logger=None)
        
        if os.path.exists(temp_final_audio):
            file_size = os.path.getsize(temp_final_audio)
            print(f"Final audio file size: {file_size} bytes")
            audio_check = AudioFileClip(temp_final_audio)
            print(f"Final audio duration: {audio_check.duration}s")
            audio_check.close()
        
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
        
        video.close()
        final_audio_clip.close()
        final_video.close()
        for seg in audio_segments:
            try:
                seg.close()
            except:
                pass
        
        # Apply lip sync if enabled
        if enable_lip_sync:
            print("\n" + "="*50)
            print("LIP SYNC ENABLED - STARTING LIP SYNC PROCESS")
            print("="*50)
            update_job_status(job_id, "applying_lip_sync", 97)
            
            lip_sync_output = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}_dubbed_{target_language}_lip_synced.mp4")
            
            try:
                await apply_lip_sync(
                    video_path=file_path,  # Original video
                    audio_path=temp_final_audio,  # Dubbed audio WAV file
                    output_path=lip_sync_output
                )
                
                # Use lip-synced version as final output
                if os.path.exists(lip_sync_output):
                    output_path = lip_sync_output
                    print("LIP SYNC COMPLETE - Using lip-synced video as final output")
                else:
                    print("LIP SYNC FAILED - No output file, using regular dubbed video")
                
            except Exception as e:
                print(f"LIP SYNC ERROR: {e}")
                print("Continuing with non-lip-synced video")
        else:
            print("\nLIP SYNC DISABLED - Skipping lip sync")
        
        update_job_status(job_id, "completed", 100, result=output_path)
        
    except DubbingError as e:
        error_msg = f"{e.stage} failed: {e.detail}"
        update_job_status(job_id, "failed", error=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        update_job_status(job_id, "failed", error=error_msg)

@app.get("/")
def read_root():
    return {
        "message": "Polydub v5.0 - Working Version",
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
    target_language: str = Form(...),
    enable_lip_sync: str = Form("false")  # Accept as string from Postman
):
    print("\n" + "="*50)
    print("NEW UPLOAD REQUEST")
    print(f"File: {file.filename}")
    print(f"Target language: {target_language}")
    print(f"Enable lip sync (raw string): '{enable_lip_sync}'")
    
    # Convert string to boolean
    enable_lip_sync_bool = enable_lip_sync.lower() in ["true", "1", "yes", "on"]
    print(f"Enable lip sync (converted to bool): {enable_lip_sync_bool}")
    print("="*50 + "\n")
    
    file_size = 0
    temp_dir = None
    
    try:
        temp_file = tempfile.SpooledTemporaryFile(max_size=1024*1024)
        while chunk := await file.read(1024*1024):
            file_size += len(chunk)
            temp_file.write(chunk)
        temp_file.seek(0)
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in SUPPORTED_VIDEO_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported format")
        
        if target_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported language")
        
        if file_size > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File too large")
        
        job_id = str(uuid.uuid4())
        update_job_status(job_id, "uploading", 0)
        
        temp_dir = tempfile.mkdtemp(prefix=f"polydub_{job_id}_")
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, 'wb') as f:
            temp_file.seek(0)
            shutil.copyfileobj(temp_file, f)
        temp_file.close()
        
        background_tasks.add_task(
            process_video_with_perfect_sync,
            job_id,
            file_path,
            file.filename,
            target_language,
            enable_lip_sync_bool  # Pass the boolean value
        )
        
        return {
            "job_id": job_id,
            "message": "Video processing started with perfect synchronization",
            "features": {
                "transcription": "WhisperX (word-level)",
                "translation": "GPT-4o (smart)",
                "voice": "XTTS (cloned)",
                "sync": "Perfect word-level",
                "lip_sync": "ENABLED" if enable_lip_sync_bool else "DISABLED"
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
