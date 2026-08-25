FROM python:3.11-slim-bookworm AS model-builder

# ============================================================
# MODEL BUILDER
# ============================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The CLI is needed ONLY in this builder stage to download models.
# SciPy is needed here because docling-tools imports Docling's OCR
# modules when the CLI starts.
RUN pip install --no-cache-dir \
    "docling-slim[format-pdf,models-local,feat-ocr-rapidocr-onnx,cli]" \
    scipy \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Store the offline Docling model artifacts separately.
ENV DOCLING_ARTIFACTS_PATH=/opt/docling/models

RUN mkdir -p /opt/docling/models

# Download ONLY the models required by this application:
#
# layout      -> PDF document layout understanding
# tableformer -> table structure recognition
# rapidocr    -> OCR
#
# ONNX Runtime is the RapidOCR backend.
#
# ch = Chinese
# en = English
#
# Keep both if your PDFs can contain both languages.
RUN docling-tools models download \
    --output-dir /opt/docling/models \
    layout \
    tableformer \
    rapidocr \
    --rapidocr-backend-lang onnxruntime:ch \
    --rapidocr-backend-lang onnxruntime:en


# ============================================================
# FINAL RUNTIME IMAGE
# ============================================================

FROM python:3.11-slim-bookworm

# System deps required by PDF/image processing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Runtime needs:
#   format-pdf
#   convert-core
#   models-local
#   RapidOCR + ONNX Runtime
RUN pip install --no-cache-dir \
    "docling-slim[format-pdf,convert-core,models-local,feat-ocr-rapidocr-onnx,cli]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install API dependencies
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy ONLY the pre-downloaded Docling models
# from the builder stage.
COPY --from=model-builder \
    /opt/docling/models \
    /opt/docling/models

ENV DOCLING_ARTIFACTS_PATH=/opt/docling/models

# On container environments, limit OpenMP threads.
ENV OMP_NUM_THREADS=4

# Application
WORKDIR /app

COPY . /app/wsla_service/

# Runtime directories
RUN mkdir -p \
    /app/wsla_uploads \
    /app/wsla_output

ENV WSLA_UPLOAD_DIR=/app/wsla_uploads
ENV WSLA_OUTPUT_DIR=/app/wsla_output
ENV PYTHONPATH=/app/wsla_service

WORKDIR /app/wsla_service

# Hugging Face Spaces
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]