FROM python:3.10-slim

# Install ffmpeg and mime-support
RUN apt-get update && \
    apt-get install -y ffmpeg mime-support && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Ensure mimetypes are loaded properly by Python
ENV MIME_TYPES_PATH=/etc/mime.types

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
