"""Semantica-facing facade for HugeGraph traversers and Server OLAP."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .catalog import HugeGraphAlgorithm, OLAP_ALGORITHMS, TaskId
from .olap import HugeGraphOlapMixin, RestExecutor


class HugeGraphAnalytics:
    """Semantica-facing facade for HugeGraph-specific analytics."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def capabilities(self) -> Dict[str, List[Dict[str, Any]]]:
        traversers = [
            {
                "name": name,
                "execution": "traverser",
                "mutates_graph": False,
            }
            for name in sorted(self.store._NATIVE_ALGORITHMS)
        ]
        return {"traversers": traversers, "olap": self.store.list_olap_algorithms()}

    def shortest_path(self, *args: Any, **kwargs: Any) -> Any:
        return self.store.shortest_path(*args, **kwargs)

    def traverser(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.store.execute_native_algorithm(name, *args, **kwargs)

    def submit(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> TaskId:
        return self.store.submit_olap_algorithm(name, parameters, **options)

    def run(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        return self.store.run_olap_algorithm(name, parameters, **options)

    def task(self, task_id: TaskId) -> Dict[str, Any]:
        return self.store.get_olap_task(task_id)

    def result(self, task_id: TaskId) -> Any:
        return self.store.get_olap_task_result(task_id)

    def page_rank(self, **parameters: Any) -> Dict[str, Any]:
        allow = bool(parameters.pop("allow_mutation", False))
        return self.run("page_rank", parameters, allow_mutation=allow)

    def louvain(self, **parameters: Any) -> Dict[str, Any]:
        allow = bool(parameters.pop("allow_mutation", False))
        return self.run("louvain", parameters, allow_mutation=allow)

    def label_propagation(self, **parameters: Any) -> Dict[str, Any]:
        allow = bool(parameters.pop("allow_mutation", False))
        return self.run("lpa", parameters, allow_mutation=allow)

    def weak_connected_components(self, **parameters: Any) -> Dict[str, Any]:
        allow = bool(parameters.pop("allow_mutation", False))
        return self.run(
            "weak_connected_component",
            parameters,
            allow_mutation=allow,
        )

    def degree_centrality(self, **parameters: Any) -> Dict[str, Any]:
        return self.run("degree_centrality", parameters)

    def betweenness_centrality(self, **parameters: Any) -> Dict[str, Any]:
        return self.run("betweenness_centrality", parameters)

    def closeness_centrality(self, **parameters: Any) -> Dict[str, Any]:
        return self.run("closeness_centrality", parameters)

    def k_core(self, **parameters: Any) -> Dict[str, Any]:
        return self.run("k_core", parameters)

    def triangle_count(self, **parameters: Any) -> Dict[str, Any]:
        return self.run("triangle_count", parameters)

    def clustering_coefficient(self, **parameters: Any) -> Dict[str, Any]:
        return self.run("cluster_coefficient", parameters)


__all__ = [
    "HugeGraphAlgorithm",
    "HugeGraphAnalytics",
    "HugeGraphOlapMixin",
    "OLAP_ALGORITHMS",
    "RestExecutor",
    "TaskId",
]
