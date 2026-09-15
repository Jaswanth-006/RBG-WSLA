"""The document pipeline, one module per stage.

    classify      Stage 2  is the PDF digital, scanned, or mixed?
    extract       Stage 3  Docling: layout, tables, text, EasyOCR
    quality       Stage 4  signals for text that is present but wrong
    recover       Stage 4  PaddleOCR re-reads weak or garbled pages
    assemble      Stage 5  fold rescued text back into the document
    orchestrator           runs stages 2-5 for one PDF

Stage 1 (intake) lives in ``wsla.api.routes`` and ``wsla.storage``.
"""
