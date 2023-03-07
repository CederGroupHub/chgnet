"""The pytorch implementation for CHGNet neural network potential."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)  # read from setup.py
except PackageNotFoundError:
    __version__ = "unknown"
