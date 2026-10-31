"""Compatibility imports for the modular HugeGraph backend."""

from .hugegraph import (
    HUGEGRAPH_AVAILABLE,
    OLAP_ALGORITHMS,
    HugeGraphAlgorithm,
    HugeGraphAnalytics,
    HugeGraphStore,
)

__all__ = [
    "HugeGraphStore",
    "HUGEGRAPH_AVAILABLE",
    "HugeGraphAnalytics",
    "HugeGraphAlgorithm",
    "OLAP_ALGORITHMS",
]
