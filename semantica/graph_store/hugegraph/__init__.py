"""Apache HugeGraph backend for Semantica."""

from .store import HUGEGRAPH_AVAILABLE, HugeGraphStore

__all__ = ["HugeGraphStore", "HUGEGRAPH_AVAILABLE"]
