FROM python:3.11-slim-bookworm AS model-builder

# ============================================================
# MODEL BUILDER
# ============================================================

# OLD:
# RUN apt-get update \
#     && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
#     && rm -rf /var/lib/apt/lists/*

# NEW:
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# OLD:
# RUN pip install --no-cache-dir \
#     "docling[standard]" \
#     --extra-index-url https://download.pytorch.org/whl/cpu

# NEW:
# The CLI is needed ONLY in this builder stage to download models.
# SciPy is needed here because docling-tools imports Docling's OCR
# modules when the CLI starts.
RUN pip install --no-cache-dir \
    "docling-slim[format-pdf,models-local,feat-ocr-rapidocr-onnx,cli]" \
    scipy \
    --extra-index-url https://download.pytorch.org/whl/cpu

# OLD:
# ENV HF_HOME=/tmp/
# ENV TORCH_HOME=/tmp/
#
# OLD:
# RUN docling-tools models download

# NEW:
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
#
# OLD:
# RUN apt-get update \
#     && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
#     && rm -rf /var/lib/apt/lists/*

# NEW:
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# OLD:
# RUN pip install --no-cache-dir \
#     "docling[standard]" \
#     --extra-index-url https://download.pytorch.org/whl/cpu

# NEW:
# Runtime does NOT need the Docling CLI because models were already
# downloaded in the builder stage.
#
# It DOES need:
#   format-pdf
#   models-local
#   RapidOCR + ONNX Runtime
# OLD - keep for reference:
# RUN pip install --no-cache-dir \
#     "docling-slim[format-pdf,models-local,feat-ocr-rapidocr-onnx,cli]" \
#     --extra-index-url https://download.pytorch.org/whl/cpu

# NEW - convert-core supplies scipy, numpy, pillow and rtree,
# which are required by the local Docling PDF/OCR pipeline.
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

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]