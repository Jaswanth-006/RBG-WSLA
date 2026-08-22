FROM python:3.11-slim-bookworm

# System deps required by docling (PDF rendering, image processing)
RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
    && rm -rf /var/lib/apt/lists/*

# Install docling with standard extras (PDF + OCR + Office + CLI)
# CPU-only PyTorch to keep the image smaller
RUN pip install --no-cache-dir \
    "docling[standard]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install API dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Pre-download Docling models so first parse is fast
ENV HF_HOME=/tmp/
ENV TORCH_HOME=/tmp/
RUN docling-tools models download

# On container environments, set a thread budget to avoid congestion
ENV OMP_NUM_THREADS=4

# Copy the service code
WORKDIR /app
COPY . /app/wsla_service/

# Create upload/output directories
RUN mkdir -p /app/wsla_uploads /app/wsla_output

# Environment variables for the service
ENV WSLA_UPLOAD_DIR=/app/wsla_uploads
ENV WSLA_OUTPUT_DIR=/app/wsla_output
ENV PYTHONPATH=/app/wsla_service

# Run from the directory that contains app.py
WORKDIR /app/wsla_service

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
