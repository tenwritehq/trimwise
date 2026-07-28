"""Expose compressor adapters and the factory used by benchmark runners."""

from .base import CompressionResult, CompressorAdapter
from .factory import build_adapter

__all__ = ["CompressionResult", "CompressorAdapter", "build_adapter"]
