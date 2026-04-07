"""System metrics: CPU, GPU, VRAM, RAM — delta caused by LLM generation."""

import logging
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

# Prime the CPU counter at module load so the first real read is meaningful
psutil.cpu_percent(interval=None)

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_OK = True
except Exception:
    _GPU_OK = False


@dataclass
class SystemSnapshot:
    """Captured just before generation starts. Used to compute deltas."""
    vram_used_gb: float | None
    ram_used_gb: float


@dataclass
class GenerationMetrics:
    prompt_tokens: int
    completion_tokens: int
    elapsed_sec: float
    cpu_pct: float
    gpu_pct: float | None
    vram_delta_gb: float | None
    ram_delta_gb: float


def _vram_gb() -> float | None:
    if not _GPU_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3
    except Exception:
        return None


def _gpu_util() -> float | None:
    if not _GPU_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        return None


def take_snapshot() -> SystemSnapshot:
    """Snapshot state and prime CPU/GPU counters.

    Call this immediately before the LLM call.
    The subsequent call to measure_delta() will return usage
    *during* the generation interval only.
    """
    psutil.cpu_percent(interval=None)   # prime — next read covers generation period
    _gpu_util()                         # prime
    return SystemSnapshot(
        vram_used_gb=_vram_gb(),
        ram_used_gb=psutil.virtual_memory().used / 1024**3,
    )


def measure_delta(
    baseline: SystemSnapshot,
    prompt_tokens: int,
    completion_tokens: int,
    elapsed_sec: float,
) -> GenerationMetrics:
    """Compute resource usage since baseline snapshot.

    CPU and GPU percentages cover exactly the generation interval.
    VRAM and RAM are absolute deltas (after − before).
    """
    cpu_pct = psutil.cpu_percent(interval=None)   # % during generation
    gpu_pct = _gpu_util()                          # % during generation

    vram_now = _vram_gb()
    vram_delta = (
        vram_now - baseline.vram_used_gb
        if vram_now is not None and baseline.vram_used_gb is not None
        else None
    )

    ram_now = psutil.virtual_memory().used / 1024**3
    ram_delta = ram_now - baseline.ram_used_gb

    return GenerationMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_sec=elapsed_sec,
        cpu_pct=cpu_pct,
        gpu_pct=gpu_pct,
        vram_delta_gb=vram_delta,
        ram_delta_gb=ram_delta,
    )


def format_metrics(m: GenerationMetrics) -> str:
    """Format metrics as a compact single line for display below the reply."""

    def _sign(v: float) -> str:
        return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"

    parts = [
        f"tokens: {m.prompt_tokens} in / {m.completion_tokens} out",
        f"time: {m.elapsed_sec:.1f}s",
        f"CPU: {m.cpu_pct:.0f}%",
    ]
    if m.gpu_pct is not None:
        parts.append(f"GPU: {m.gpu_pct:.0f}%")
    if m.vram_delta_gb is not None:
        parts.append(f"VRAM: {_sign(m.vram_delta_gb)} GB")
    parts.append(f"RAM: {_sign(m.ram_delta_gb)} GB")
    return " | ".join(parts)
