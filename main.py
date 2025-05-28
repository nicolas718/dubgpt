import os
import time
import requests
import replicate
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip, AudioFileClip
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

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

    # Transcribe with Replicate Whisper
    try:
        with open(audio_path, "rb") as file:
            output = replicate.run(
                "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e",
                input={"audio": file}
            )
            transcript_text = output["transcription"]
    except Exception as e:
        tb_str = traceback.format_exc()
        return JSONResponse(status_code=500, content={
            "error": "Whisper transcription failed",
            "exception": str(e),
            "traceback": tb_str
        })

    # Translate
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

    # TTS with Replicate XTTS-v2
    try:
        with open(audio_path, "rb") as speaker_file:
            input = {
                "text": translated_text,
                "speaker": speaker_file,
                "language": target_language
            }
            output = replicate.run(
                "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e",
                input=input
            )

        print("[DEBUG] Replicate XTTS Output:", output)

        # Accept string or list format
        if isinstance(output, list) and len(output) > 0:
            audio_url = output[0]
        elif isinstance(output, str):
            audio_url = output
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
        return JSONResponse(status_code=500, content={
            "error": "Replicate TTS failed",
            "exception": str(e),
            "traceback": tb_str
        })

    # Merge dubbed audio with original video
    try:
        output_video_path = file_location.rsplit(".", 1)[0] + f"_final_{target_language}.mp4"
        original_video = VideoFileClip(file_location)
        dubbed_audio = AudioFileClip(dubbed_audio_path)
        final_video = original_video.set_audio(dubbed_audio)
        final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Final video merge failed: {str(e)}"})

    return {
        "message": "Pipeline complete: Transcription, translation, TTS, merge done.",
        "original_transcript": transcript_text,
        "translated_transcript": translated_text,
        "dubbed_audio_path": dubbed_audio_path,
        "final_video_path": output_video_path
    }





