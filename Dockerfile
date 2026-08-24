# ============================================================
# MODEL BUILDER STAGE - Download models with CPU PyTorch
# ============================================================
FROM python:3.11-slim-bookworm AS model-builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install minimal docling with CPU PyTorch just for downloading models
RUN pip install --no-cache-dir \
    "docling-slim[format-pdf,models-local,feat-ocr-rapidocr-onnx,cli]" \
    scipy \
    --extra-index-url https://download.pytorch.org/whl/cpu

ENV DOCLING_ARTIFACTS_PATH=/opt/docling/models

RUN mkdir -p /opt/docling/models

# Download only required models (remove Chinese if not needed)
RUN docling-tools models download \
    --output-dir /opt/docling/models \
    layout \
    tableformer \
    rapidocr \
    --rapidocr-backend-lang onnxruntime:en


# ============================================================
# FINAL RUNTIME IMAGE - GPU enabled with minimal footprint
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set timezone non-interactively before any package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11.9 from deadsnakes PPA (has sys.get_int_max_str_digits)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install docling WITHOUT CLI (runtime doesn't need it) - use GPU PyTorch
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir \
    "docling-slim[format-pdf,convert-core,models-local,feat-ocr-rapidocr-onnx]"

# Install API dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy pre-downloaded models from builder
COPY --from=model-builder /opt/docling/models /opt/docling/models

# Environment
ENV DOCLING_ARTIFACTS_PATH=/opt/docling/models
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# Application setup
WORKDIR /app
COPY . /app/wsla_service/

RUN mkdir -p /app/wsla_uploads /app/wsla_output

ENV WSLA_UPLOAD_DIR=/app/wsla_uploads
ENV WSLA_OUTPUT_DIR=/app/wsla_output
ENV WSLA_DEVICE=auto
ENV PYTHONPATH=/app/wsla_service

WORKDIR /app/wsla_service

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
