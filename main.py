from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "DubGPT backend is live."}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # Save uploaded file to local /tmp directory
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    return JSONResponse({
        "message": "Video uploaded successfully.",
        "filename": file.filename,
        "path": file_location
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
