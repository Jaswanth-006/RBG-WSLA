"""Device selection: run Docling and EasyOCR on the GPU when one is available.

``WSLA_DEVICE`` accepts ``auto``, ``gpu`` or ``cpu``. ``auto`` uses CUDA when a GPU
is visible; ``gpu`` fails loudly when none is. The PaddleOCR fallback is
configured separately (``WSLA_PADDLE_DEVICE``) and stays on CPU.
"""

import torch

from wsla.config import settings


def get_gpu_info() -> tuple[bool, str | None, str | None, int]:
    """Read actual CUDA/GPU state from PyTorch."""
    try:
        if not torch.cuda.is_available():
            return False, None, None, 0
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else None
        return True, gpu_name, torch.version.cuda, gpu_count
    except Exception:
        return False, None, None, 0


def resolve_device(requested_device: str) -> str:
    """Resolve the requested mode (auto/gpu/cpu) to a runtime device."""
    normalized = (requested_device or "auto").strip().lower()
    if normalized not in {"auto", "gpu", "cpu"}:
        raise ValueError("Unsupported device setting. Use one of: auto, gpu, cpu.")

    if normalized == "cpu":
        return "cpu"
    if normalized == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "WSLA_DEVICE=gpu was requested, but CUDA/GPU is unavailable."
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_runtime_summary(requested_device: str | None = None) -> dict[str, object]:
    """Return a runtime summary used for startup output and metrics."""
    requested = (requested_device or settings.device or "auto").strip().lower()
    resolved = resolve_device(requested)
    gpu_available, gpu_name, cuda_version, gpu_count = get_gpu_info()
    return {
        "requested_device": requested,
        "resolved_device": resolved,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "gpu_count": gpu_count,
    }
