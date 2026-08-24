import torch

from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

print("=" * 60)
print("GPU TEST")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("PyTorch cannot see CUDA")

accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA
)

pipeline_options = PdfPipelineOptions(
    accelerator_options=accelerator_options
)

pipeline_options.do_ocr = False

converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    },
)

print("=" * 60)
print("Docling converter created with CUDA")
print("=" * 60)