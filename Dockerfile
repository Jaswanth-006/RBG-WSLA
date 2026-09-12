FROM python:3.11-slim-bookworm AS model-builder

# ============================================================
# MODEL BUILDER
# ============================================================

# OLD:
# RUN apt-get update \
#     && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
#     && rm -rf /var/lib/apt/lists/*

# NEW:
# libgomp1 is the OpenMP runtime PaddlePaddle links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# OLD:
# RUN pip install --no-cache-dir \
#     "docling[standard]" \
#     --extra-index-url https://download.pytorch.org/whl/cpu

# NEW:
# The CLI is needed ONLY in this builder stage to download models.
# SciPy is needed here because docling-tools imports Docling's OCR
# modules when the CLI starts. convert-core supplies rtree, which
# docling-tools imports too — without it the model download fails.
RUN pip install --no-cache-dir \
    "docling-slim[format-pdf,convert-core,models-local,feat-ocr-easyocr,cli]" \
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
#
# RapidOCR is deliberately NOT prefetched: its checkpoints are served from
# ModelScope, which is unreachable from our network, and they default to
# Chinese. OCR is handled by PaddleOCR instead (see ocr_fallback.py), whose
# Latin models are fetched from Hugging Face further down.
RUN docling-tools models download \
    --output-dir /opt/docling/models \
    layout \
    tableformer

# ------------------------------------------------------------
# EasyOCR — the OCR engine Docling uses (Spanish)
# ------------------------------------------------------------
# Models are written to /opt/docling/models/EasyOcr and travel to the
# runtime stage with the rest of DOCLING_ARTIFACTS_PATH, so OCR works
# with no network at run time.

RUN docling-tools models download \
    --output-dir /opt/docling/models \
    easyocr \
    --easyocr-lang es

# ------------------------------------------------------------
# PaddleOCR fallback models
# ------------------------------------------------------------
# PaddlePaddle version follows the PaddleOCR quick start. The PyPI
# paddlepaddle wheel is the CPU build, so no extra package index is needed.
# The warm-up script downloads PP-OCRv5 detection, the Latin recogniser
# (Spanish) and the text-line orientation model into
# $PADDLE_PDX_CACHE_HOME/official_models, which the runtime stage copies.
ENV PADDLE_PDX_CACHE_HOME=/opt/paddle

RUN pip install --no-cache-dir paddlepaddle==3.2.0 \
    && pip install --no-cache-dir "paddleocr==3.7.0"

COPY download_paddle_models.py /tmp/download_paddle_models.py

RUN python /tmp/download_paddle_models.py


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
# libgomp1 is the OpenMP runtime PaddlePaddle links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
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
# GPU PyTorch FIRST, so Docling and EasyOCR bind to the CUDA build.
# These wheels bundle the CUDA runtime; the driver comes from the host via
# `--gpus all`, so no CUDA base image is needed. cu126 matches torch 2.14.
RUN pip install --no-cache-dir \
    torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cu126

# No CPU torch index here: the CUDA build above must win.
RUN pip install --no-cache-dir \
    "docling-slim[format-pdf,convert-core,models-local,feat-ocr-easyocr,cli]"

# PaddlePaddle framework (CPU build from PyPI) for the PaddleOCR fallback.
# paddleocr itself is installed from requirements.txt.
RUN pip install --no-cache-dir paddlepaddle==3.2.0

# Install API dependencies
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# EasyOCR runtime (Docling's OCR engine). Its models arrive from the
# builder stage inside /opt/docling/models.

# Copy ONLY the pre-downloaded Docling models
# from the builder stage.
COPY --from=model-builder \
    /opt/docling/models \
    /opt/docling/models

ENV DOCLING_ARTIFACTS_PATH=/opt/docling/models

# Copy the pre-downloaded PaddleOCR models from the builder stage.
COPY --from=model-builder \
    /opt/paddle \
    /opt/paddle

ENV PADDLE_PDX_CACHE_HOME=/opt/paddle
ENV WSLA_PADDLE_MODEL_DIR=/opt/paddle/official_models
# Models are baked in, so skip PaddleX's startup check of the model hosts.
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# On container environments, limit OpenMP threads.
ENV OMP_NUM_THREADS=4

# auto = use the GPU when one is visible, else CPU
ENV WSLA_DEVICE=auto

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

# Port will be set via environment variable
ARG PORT=7860
ENV PORT=${PORT}
EXPOSE ${PORT}

CMD ["python", "app.py"]
