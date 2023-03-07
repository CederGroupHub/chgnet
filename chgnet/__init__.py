"""The pytorch implementation for CHGNet neural network potential."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)  # read from setup.py
except PackageNotFoundError:
    __version__ = "unknown"
