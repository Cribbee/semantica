"""HugeGraph Server OLAP capability catalog and result helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Union

TaskId = Union[int, str]


@dataclass(frozen=True)
class HugeGraphAlgorithm:
    """One algorithm registered by HugeGraph Server's ``AlgorithmPool``."""

    name: str
    category: str
    mutates_graph: bool = False
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "mutates_graph": self.mutates_graph,
            "description": self.description,
        }


# Names match HugeGraph Server's AlgorithmPool. Algorithms marked mutating
# create schema and/or write result properties, so callers must opt in and use
# an isolated graph.
OLAP_ALGORITHMS: Dict[str, HugeGraphAlgorithm] = {
    "count_vertex": HugeGraphAlgorithm("count_vertex", "statistics"),
    "count_edge": HugeGraphAlgorithm("count_edge", "statistics"),
    "degree_centrality": HugeGraphAlgorithm("degree_centrality", "centrality"),
    "stress_centrality": HugeGraphAlgorithm("stress_centrality", "centrality"),
    "betweenness_centrality": HugeGraphAlgorithm(
        "betweenness_centrality", "centrality"
    ),
    "closeness_centrality": HugeGraphAlgorithm(
        "closeness_centrality", "centrality"
    ),
    "eigenvector_centrality": HugeGraphAlgorithm(
        "eigenvector_centrality", "centrality"
    ),
    "triangle_count": HugeGraphAlgorithm("triangle_count", "structure"),
    "cluster_coefficient": HugeGraphAlgorithm(
        "cluster_coefficient", "structure"
    ),
    "k_core": HugeGraphAlgorithm("k_core", "structure"),
    "subgraph_stat": HugeGraphAlgorithm("subgraph_stat", "statistics"),
    "page_rank": HugeGraphAlgorithm(
        "page_rank",
        "ranking",
        mutates_graph=True,
        description="Creates rank schema and writes rank properties",
    ),
    "lpa": HugeGraphAlgorithm(
        "lpa",
        "community",
        mutates_graph=True,
        description="Writes label-propagation community properties",
    ),
    "louvain": HugeGraphAlgorithm(
        "louvain",
        "community",
        mutates_graph=True,
        description="Writes Louvain community properties",
    ),
    "weak_connected_component": HugeGraphAlgorithm(
        "weak_connected_component",
        "community",
        mutates_graph=True,
        description="Writes weak-component properties",
    ),
}

_ALIASES = {
    "pagerank": "page_rank",
    "degree": "degree_centrality",
    "stress": "stress_centrality",
    "betweenness": "betweenness_centrality",
    "closeness": "closeness_centrality",
    "eigenvector": "eigenvector_centrality",
    "wcc": "weak_connected_component",
    "weakly_connected_component": "weak_connected_component",
    "weakly_connected_components": "weak_connected_component",
    "connected_components": "weak_connected_component",
    "label_propagation": "lpa",
    "clustering_coefficient": "cluster_coefficient",
    "kcore": "k_core",
}
_SUCCESS = {"success", "succeeded", "complete", "completed"}
_FAILURE = {"failed", "failure", "error", "cancelled", "canceled", "deleted"}


def _decode_json(value: Any, *, depth: int = 0) -> Any:
    """Decode HTTP envelopes and JSON-string task results conservatively."""
    if depth > 3:
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return _decode_json(json.loads(stripped), depth=depth + 1)
        except (TypeError, ValueError):
            return value
    return value
