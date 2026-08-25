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
