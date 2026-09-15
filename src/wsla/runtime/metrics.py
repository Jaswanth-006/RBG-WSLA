"""Throughput arithmetic for the performance reports."""


def calculate_throughput(page_count: int, elapsed_seconds: float) -> tuple[float, float]:
    """Return pages/sec and pages/minute with zero-safe handling."""
    if elapsed_seconds <= 0:
        return 0.0, 0.0
    pages_per_second = page_count / elapsed_seconds
    return pages_per_second, pages_per_second * 60.0
