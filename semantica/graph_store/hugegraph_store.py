"""Compatibility import for the modular HugeGraph backend."""

from .hugegraph import HUGEGRAPH_AVAILABLE, HugeGraphStore

__all__ = ["HugeGraphStore", "HUGEGRAPH_AVAILABLE"]
