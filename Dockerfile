FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# System packages needed for OpenCV / image processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better Docker caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Run FastAPI app on Cloud Run / local Docker
CMD ["sh", "-c", "uvicorn cnn_inference:app --host 0.0.0.0 --port ${PORT}"]