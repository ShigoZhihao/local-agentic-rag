"""System metrics collection: tokens, time, GPU, RAM."""

import logging
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

try:
    import pynvml
    pynvml.nvmlInit()
    _PYNVML_OK = True
except Exception:
    _PYNVML_OK = False


@dataclass
class GPUStats:
    gpu_util_pct: int
    vram_used_gb: float
    vram_total_gb: float


@dataclass
class RAMStats:
    ram_used_gb: float
    ram_total_gb: float


def get_gpu_stats() -> GPUStats | None:
    """Return GPU utilisation and VRAM usage for device 0, or None if unavailable."""
    if not _PYNVML_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GPUStats(
            gpu_util_pct=util.gpu,
            vram_used_gb=mem.used / 1024**3,
            vram_total_gb=mem.total / 1024**3,
        )
    except Exception as e:
        logger.debug("GPU stats unavailable: %s", e)
        return None


def get_ram_stats() -> RAMStats:
    """Return system RAM usage."""
    mem = psutil.virtual_memory()
    return RAMStats(
        ram_used_gb=mem.used / 1024**3,
        ram_total_gb=mem.total / 1024**3,
    )


def format_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    elapsed_sec: float,
    gpu: GPUStats | None,
    ram: RAMStats,
) -> str:
    """Build a single-line metrics string for display below the LLM reply.

    Args:
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.
        elapsed_sec: Wall-clock seconds for the LLM call.
        gpu: GPU stats, or None if no GPU available.
        ram: System RAM stats.

    Returns:
        Formatted metrics string, e.g.:
        "tokens: 32 in / 118 out | time: 2.4s | GPU: 47% | VRAM: 3.1/8.0 GB | RAM: 12.4/31.8 GB"
    """
    parts = [
        f"tokens: {prompt_tokens} in / {completion_tokens} out",
        f"time: {elapsed_sec:.1f}s",
    ]
    if gpu is not None:
        parts.append(f"GPU: {gpu.gpu_util_pct}%")
        parts.append(f"VRAM: {gpu.vram_used_gb:.1f}/{gpu.vram_total_gb:.1f} GB")
    parts.append(f"RAM: {ram.ram_used_gb:.1f}/{ram.ram_total_gb:.1f} GB")
    return " | ".join(parts)
