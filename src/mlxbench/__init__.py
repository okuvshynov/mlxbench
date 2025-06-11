"""MLXBench - Benchmarking tool for Apple MLX-based language models."""

from .hwinfo import apple_hwinfo, format_hwinfo

__version__ = "0.1.0"
__all__ = ["apple_hwinfo", "format_hwinfo"]