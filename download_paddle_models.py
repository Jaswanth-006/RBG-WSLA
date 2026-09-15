"""Download and warm the PaddleOCR models used by the OCR fallback.

Run at image build time so the service never downloads models at runtime.
Models land in PaddleX's cache, ``$PADDLE_PDX_CACHE_HOME/official_models``.

Usage:
    PADDLE_PDX_CACHE_HOME=/opt/paddle python download_paddle_models.py

Model names default to the values in config.py and honour the same
``WSLA_PADDLE_*`` environment variables.
"""

import os

import numpy as np
from paddleocr import PaddleOCR

DET_MODEL = os.getenv("WSLA_PADDLE_DET_MODEL", "PP-OCRv5_mobile_det")
REC_MODEL = os.getenv("WSLA_PADDLE_REC_MODEL", "latin_PP-OCRv5_mobile_rec")
USE_TEXTLINE_ORIENTATION = os.getenv(
    "WSLA_PADDLE_USE_TEXTLINE_ORIENTATION", "true"
).lower() not in ("0", "false", "no")


def main() -> None:
    ocr = PaddleOCR(
        device="cpu",
        text_detection_model_name=DET_MODEL,
        text_recognition_model_name=REC_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
    )
    # One inference on a blank image proves the runtime loads end to end.
    ocr.predict(np.full((64, 256, 3), 255, dtype=np.uint8))
    print(f"PaddleOCR models ready: {DET_MODEL}, {REC_MODEL}")


if __name__ == "__main__":
    main()
