"""CPU-aware PDF job scheduler for the WSLA API.

The scheduler provides:
- dynamic CPU detection
- FIFO waiting queue
- a minimum logical allocation of 2 CPUs when >2 CPUs exist
- up to floor(total_cpus / 2) concurrent PDF jobs
- deterministic redistribution of the logical CPU budget as jobs start/finish

Important:
This module controls admission/concurrency and records a logical CPU budget.
With a single shared Docling DocumentConverter, Python cannot safely change
Docling's per-request thread count while another conversion is already using
the same converter. Therefore the scheduler does NOT pretend to hard-pin
different CPU cores to different jobs. The shared converter remains a single
instance, and Docling manages its own internal threading.

The design deliberately favors correctness and memory efficiency over creating
multiple model-heavy converter instances.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")


def detect_cpu_count() -> int:
    """Return the CPU count visible to the container."""
    count = os.cpu_count() or 1
    return max(1, int(count))


@dataclass
class PdfJob(Generic[T]):
    job_id: str
    future: Future[T]
    task: Callable[[int], T]
    allocated_cores: int = 0
    status: str = "queued"
    sequence: int = 0


class CpuAwarePdfScheduler(Generic[T]):
    """FIFO PDF scheduler with a minimum logical CPU allocation."""

    def __init__(self) -> None:
        self.total_cores = detect_cpu_count()
        self.minimum_cores = 2 if self.total_cores > 2 else self.total_cores
        self.max_concurrent = (
            1 if self.total_cores <= 2 else max(1, self.total_cores // 2)
        )

        self._condition = threading.Condition()
        self._queue: deque[PdfJob[T]] = deque()
        self._running: dict[str, PdfJob[T]] = {}
        self._sequence = 0

        # The worker count is deliberately bounded to the number of jobs that
        # can each receive at least the configured minimum logical allocation.
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent,
            thread_name_prefix="wsla-pdf",
        )

        _log.info(
            "[scheduler] CPU count=%d min_cores=%d max_concurrent=%d",
            self.total_cores,
            self.minimum_cores,
            self.max_concurrent,
        )

    def submit(self, task: Callable[[int], T]) -> tuple[str, Future[T]]:
        """Queue a PDF task and return its job ID plus Future."""
        future: Future[T] = Future()
        with self._condition:
            self._sequence += 1
            job = PdfJob(
                job_id=str(uuid.uuid4()),
                future=future,
                task=task,
                sequence=self._sequence,
            )
            self._queue.append(job)
            _log.info(
                "[scheduler] queued job=%s queue=%d running=%d",
                job.job_id,
                len(self._queue),
                len(self._running),
            )
            self._rebalance_locked()
            self._condition.notify_all()
        return job.job_id, future

    def snapshot(self) -> dict[str, object]:
        """Return scheduler state for diagnostics."""
        with self._condition:
            return {
                "total_cores": self.total_cores,
                "minimum_cores": self.minimum_cores,
                "max_concurrent": self.max_concurrent,
                "queued": len(self._queue),
                "running": [
                    {
                        "job_id": job.job_id,
                        "allocated_cores": job.allocated_cores,
                        "status": job.status,
                    }
                    for job in self._running.values()
                ],
            }

    def _rebalance_locked(self) -> None:
        """Start queued jobs and rebalance the logical CPU budget.

        For >2 CPUs:
          8 -> 4+4 -> 2+4+2 -> 2+2+2+2
          6 -> 3+3 -> 2+3+1 is NOT allowed, so we stop at 3+3.
          4 -> 2+2
        """
        if self.total_cores <= 2:
            if not self._running and self._queue:
                job = self._queue.popleft()
                self._start_job_locked(job, self.total_cores)
            return

        # Start at most one new job per rebalance pass, then recompute a
        # valid allocation for every running job. This preserves the desired
        # behavior while also handling odd CPU counts safely:
        #   8 -> 4+4 -> 2+4+2 -> 2+2+2+2
        #   6 -> 3+3 -> 2+2+2
        #   5 -> 3+2
        while self._queue and len(self._running) < self.max_concurrent:
            if not self._running:
                job = self._queue.popleft()
                self._start_job_locked(job, self.total_cores)
                continue

            job = self._queue.popleft()
            # Reserve the job as running; allocation is computed below.
            self._start_job_locked(job, self.minimum_cores)
            self._recompute_allocations_locked()

        if not self._queue:
            self._recompute_allocations_locked()

    def _recompute_allocations_locked(self) -> None:
        """Compute a valid logical CPU budget for all running jobs."""
        count = len(self._running)
        if count == 0:
            return
        if self.total_cores <= 2:
            only = next(iter(self._running.values()))
            only.allocated_cores = self.total_cores
            return

        # At least MIN_CORES per running job.
        base = self.total_cores // count
        remainder = self.total_cores % count
        if base < self.minimum_cores:
            return

        jobs = sorted(self._running.values(), key=lambda job: job.sequence)

        # Keep the split behavior close to the requested policy. If there is a
        # remainder, give it to the previously largest allocation (newest job
        # wins ties). For 8 CPUs this produces:
        #   8 -> 4+4 -> 2+4+2 -> 2+2+2+2.
        recipient = None
        if remainder:
            recipient = max(
                jobs,
                key=lambda job: (job.allocated_cores, job.sequence),
            )

        for job in jobs:
            job.allocated_cores = base
        if recipient is not None:
            recipient.allocated_cores += remainder

        _log.info(
            "[scheduler] allocation: %s",
            ", ".join(
                f"{job.job_id[:8]}={job.allocated_cores}"
                for job in jobs
            ),
        )

    def _start_job_locked(self, job: PdfJob[T], allocated_cores: int) -> None:
        job.allocated_cores = allocated_cores
        job.status = "running"
        self._running[job.job_id] = job

        _log.info(
            "[scheduler] starting job=%s allocated_cores=%d running=%d queued=%d",
            job.job_id,
            job.allocated_cores,
            len(self._running),
            len(self._queue),
        )

        worker_future = self._executor.submit(
            self._run_job,
            job,
            allocated_cores,
        )
        worker_future.add_done_callback(
            lambda done, job_id=job.job_id: self._job_done(job_id, done)
        )

    @staticmethod
    def _run_job(job: PdfJob[T], allocated_cores: int) -> T:
        # allocated_cores is passed explicitly so the application can record
        # the scheduler decision in logs/metrics.
        return job.task(allocated_cores)

    def _job_done(self, job_id: str, worker_future: Future[T]) -> None:
        with self._condition:
            job = self._running.pop(job_id, None)
            if job is None:
                return

            try:
                result = worker_future.result()
            except BaseException as exc:
                job.status = "failed"
                if not job.future.done():
                    job.future.set_exception(exc)
                _log.exception("[scheduler] job=%s failed", job_id)
            else:
                job.status = "completed"
                if not job.future.done():
                    job.future.set_result(result)
                _log.info("[scheduler] job=%s completed", job_id)

            self._rebalance_locked()
            self._condition.notify_all()

    def shutdown(self) -> None:
        """Shutdown the worker pool."""
        self._executor.shutdown(wait=True)
