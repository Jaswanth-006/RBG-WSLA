---
title: RBG WSLA
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
# RBG WSLA

WSLA Document Parsing API using FastAPI, Docling and OCR.

## API

- `GET /` — health check
- `POST /upload` — upload PDF
- `GET /parse/{id}` — parse document
- `POST /process` — process PDF and return ZIP
- `GET /status/{id}` — document status
  
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## CPU-aware PDF scheduling

The API now includes a small in-process scheduler for `/process`.

- CPU count is detected dynamically from the container.
- If more than 2 CPUs are available, a running PDF job is assigned a minimum logical budget of 2 CPUs.
- Up to `floor(total_cpus / 2)` PDF jobs may run concurrently.
- Extra PDFs wait in a FIFO queue.
- When a job completes, the scheduler immediately starts queued work or redistributes the logical CPU budget.
- `GET /scheduler` exposes the current scheduler state.

### Important implementation detail

The application intentionally keeps **one shared Docling `DocumentConverter`** so the downloaded models are not duplicated across worker instances.

The scheduler therefore controls **job admission/concurrency and a logical CPU budget**. It does not claim to hard-pin different CPU cores to different PDFs. Docling's own pipeline/threading remains responsible for CPU-level execution.

For a container with 8 vCPUs, the scheduler behaves like:

```text
1 PDF  -> 8 logical CPUs
2 PDFs -> 4 + 4
3 PDFs -> 2 + 4 + 2
4 PDFs -> 2 + 2 + 2 + 2
5+     -> FIFO queue
```

When the container exposes 2 or fewer CPUs, only one PDF is processed at a time.

This design avoids creating multiple model-heavy converter instances and keeps memory usage predictable.
