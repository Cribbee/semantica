"""Native Gremlin, bounded traversal, retrieval, and algorithm operations."""

from __future__ import annotations

import base64
import gzip
import json
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from ...utils.exceptions import ProcessingError, ValidationError
from .utils import (
    NodeId,
    _call,
    _edge_dict,
    _identifier,
    _lexical_score,
    _node_dict,
    _plain,
    _positive_int,
    _records,
    _result_data,
    _text_fragments,
)


class HugeGraphQueryMixin:
    """Implement Gremlin, bounded local traversal, and native algorithms."""

    def _http_gremlin(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base_uri = self._validated_http_uri()
        alias = (
            f"{self.graphspace}-{self.graph_name}"
            if self.graphspace
            else self.graph_name
        )
        payload = {
            "gremlin": query,
            "bindings": parameters or {},
            "language": "gremlin-groovy",
            "aliases": {
                "graph": alias,
                "g": f"__g_{alias}",
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
        }
        if self.user:
            credentials = f"{self.user}:{self.password}".encode("utf-8")
            token = base64.b64encode(credentials).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        request = Request(
            f"{base_uri}/gremlin",
            data=json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            opener = build_opener(ProxyHandler({}))
            with opener.open(
                request,
                timeout=self._timeout_tuple()[1],
            ) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip" or raw.startswith(
                    b"\x1f\x8b"
                ):
                    raw = gzip.decompress(raw)
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
        ) as exc:
            raise ProcessingError(f"HugeGraph Gremlin request failed: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ProcessingError(
                "HugeGraph Gremlin endpoint returned a non-object response"
            )
        status = parsed.get("status")
        if isinstance(status, dict) and status.get("code") not in {None, 200}:
            message = status.get("message") or status
            raise ProcessingError(f"HugeGraph rejected Gremlin query: {message}")
        return parsed

    def _run_gremlin(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not isinstance(query, str) or not query.strip():
            raise ValidationError("Gremlin query must be a nonempty string")
        if self._gremlin_executor is not None:
            return self._gremlin_executor(query, parameters)
        if not parameters:
            # The client returns the result object, rather than the full envelope.
            # Its current exec API does not expose Gremlin bindings.
            try:
                return {"result": self._manager("gremlin").exec(query)}
            except Exception as exc:
                raise ProcessingError(
                    f"HugeGraph Gremlin request failed: {exc}"
                ) from exc
        return self._http_gremlin(query, parameters)

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        """Execute backend-native Gremlin without Cypher translation."""
        raw = self._run_gremlin(query, parameters)
        records, keys = _records(_result_data(raw))
        return {
            "success": True,
            "records": records,
            "keys": keys,
            "metadata": {
                "query": query,
                "query_language": "gremlin-groovy",
                "backend": "hugegraph",
            },
        }

    def get_neighbors(
        self,
        node_id: NodeId,
        rel_type: Optional[str] = None,
        direction: str = "both",
        depth: int = 1,
        **options: Any,
    ) -> List[Dict[str, Any]]:
        """Read a bounded subgraph without loading the whole graph."""
        depth = _positive_int(options.get("hops", depth), "depth")
        direction = self._direction(direction)
        per_node_limit = _positive_int(
            options.get("per_node_limit", 1000),
            "per_node_limit",
        )
        frontier = {node_id}
        visited = {node_id}
        paths: Dict[NodeId, List[NodeId]] = {node_id: [node_id]}
        path_relationships: Dict[
            NodeId,
            List[Dict[str, Any]],
        ] = {node_id: []}
        distances: Dict[NodeId, int] = {}

        for distance in range(1, depth + 1):
            next_frontier: set[NodeId] = set()
            for current in frontier:
                edges = self.get_relationships(
                    node_id=current,
                    rel_type=rel_type,
                    direction=direction,
                    limit=per_node_limit,
                )
                for edge in edges:
                    start = edge.get("start_node_id")
                    end = edge.get("end_node_id")
                    if direction == "out":
                        neighbor = end if start == current else None
                    elif direction == "in":
                        neighbor = start if end == current else None
                    else:
                        neighbor = (
                            end
                            if start == current
                            else start if end == current else None
                        )
                    if neighbor is None or neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    distances[neighbor] = distance
                    paths[neighbor] = [*paths[current], neighbor]
                    path_relationships[neighbor] = [
                        *path_relationships[current],
                        edge,
                    ]
            frontier = next_frontier
            if not frontier:
                break

        neighbors: List[Dict[str, Any]] = []
        for neighbor_id in distances:
            node = self.get_node(neighbor_id)
            if node is None:
                continue
            node["distance"] = distances[neighbor_id]
            node["path"] = paths[neighbor_id]
            node["path_relationships"] = path_relationships[neighbor_id]
            neighbors.append(node)
        return neighbors

    def _normalize_path(
        self,
        raw: Any,
        start_node_id: NodeId,
        end_node_id: NodeId,
    ) -> Optional[Dict[str, Any]]:
        value = _result_data(raw)
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if not value:
            return None
        if not isinstance(value, dict):
            return {
                "length": None,
                "nodes": [],
                "relationships": [],
                "raw": _plain(value),
            }

        nested = value.get("path")
        if isinstance(nested, dict):
            value = nested
        elif isinstance(nested, list):
            if not nested:
                return None
            nodes = [
                (
                    _node_dict(item)
                    if isinstance(item, dict) or hasattr(item, "id")
                    else self.get_node(item)
                    or {"id": item, "labels": [], "properties": {}}
                )
                for item in nested
            ]
            return {
                "length": max(0, len(nodes) - 1),
                "nodes": nodes,
                "relationships": [],
                "start_node_id": start_node_id,
                "end_node_id": end_node_id,
                "native_path_ids": list(nested),
            }
        raw_nodes = value.get("vertices") or value.get("nodes") or []
        raw_edges = value.get("edges") or value.get("relationships") or []
        nodes = [_node_dict(item) for item in raw_nodes]
        relationships = [_edge_dict(item) for item in raw_edges]
        if not nodes and value.get("vertex_ids"):
            nodes = [
                self.get_node(item)
                or {
                    "id": item,
                    "labels": [],
                    "properties": {},
                }
                for item in value.get("vertex_ids", [])
            ]
        if not nodes and not relationships:
            return {
                "length": value.get("length"),
                "nodes": [],
                "relationships": [],
                "raw": _plain(value),
            }
        return {
            "length": value.get("length", len(relationships)),
            "nodes": nodes,
            "relationships": relationships,
            "start_node_id": start_node_id,
            "end_node_id": end_node_id,
        }

    def shortest_path(
        self,
        start_node_id: NodeId,
        end_node_id: NodeId,
        rel_type: Optional[str] = None,
        max_depth: int = 10,
        **options: Any,
    ) -> Optional[Dict[str, Any]]:
        max_depth = _positive_int(max_depth, "max_depth")
        direction = self._direction(options.get("direction", "both"))
        if rel_type is None and direction == "both":
            try:
                raw = _call(
                    self._manager("traverser"),
                    ("shortest_path", "shortestPath"),
                    start_node_id,
                    end_node_id,
                    max_depth,
                )
                normalized = self._normalize_path(
                    raw,
                    start_node_id,
                    end_node_id,
                )
                if normalized is None:
                    return None
                if (
                    normalized.get("relationships")
                    or len(normalized.get("nodes", [])) <= 1
                    or options.get("native_only", False)
                ):
                    return normalized
                self.logger.debug(
                    "Native HugeGraph path omitted edge records; using bounded "
                    "REST traversal to preserve relationship evidence"
                )
            except Exception as exc:
                if options.get("native_only", False):
                    raise ProcessingError(
                        "HugeGraph shortest-path request failed: " f"{exc}"
                    ) from exc
                self.logger.warning(
                    "Native HugeGraph shortest path failed; using bounded "
                    "REST traversal: %s",
                    exc,
                )
        if rel_type is not None:
            rel_type = _identifier(rel_type, "edge label")

        per_node_limit = _positive_int(
            options.get("per_node_limit", 1000),
            "per_node_limit",
        )
        queue: deque[NodeId] = deque([start_node_id])
        parents: Dict[
            NodeId,
            Tuple[Optional[NodeId], Optional[Dict[str, Any]]],
        ] = {start_node_id: (None, None)}
        distances = {start_node_id: 0}

        while queue:
            current = queue.popleft()
            if distances[current] >= max_depth:
                continue
            edges = self.get_relationships(
                node_id=current,
                rel_type=rel_type,
                direction=direction,
                limit=per_node_limit,
            )
            for edge in edges:
                start = edge.get("start_node_id")
                end = edge.get("end_node_id")
                if direction == "out":
                    neighbor = end if start == current else None
                elif direction == "in":
                    neighbor = start if end == current else None
                else:
                    neighbor = (
                        end if start == current else start if end == current else None
                    )
                if neighbor is None or neighbor in parents:
                    continue
                parents[neighbor] = (current, edge)
                distances[neighbor] = distances[current] + 1
                if neighbor == end_node_id:
                    queue.clear()
                    break
                queue.append(neighbor)

        if end_node_id not in parents:
            return None
        node_ids: List[NodeId] = []
        relationships: List[Dict[str, Any]] = []
        cursor: Optional[NodeId] = end_node_id
        while cursor is not None:
            node_ids.append(cursor)
            parent, edge = parents[cursor]
            if edge is not None:
                relationships.append(edge)
            cursor = parent
        node_ids.reverse()
        relationships.reverse()
        nodes = [
            self.get_node(item)
            or {
                "id": item,
                "labels": [],
                "properties": {},
            }
            for item in node_ids
        ]
        return {
            "length": len(relationships),
            "nodes": nodes,
            "relationships": relationships,
            "start_node_id": start_node_id,
            "end_node_id": end_node_id,
        }

    def search_nodes(
        self,
        query: str,
        limit: int = 10,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve entry nodes lexically for ContextRetriever.

        This operation is separate from raw Gremlin execution. It scans a
        bounded candidate set, scores configured text properties, and returns
        deterministic semantic records. A vector store can still be supplied
        separately for Semantica hybrid retrieval.
        """
        limit = _positive_int(limit, "limit")
        node_types = options.get("node_types")
        requested_labels = list(labels or node_types or [])
        requested_candidates = options.get(
            "candidate_limit",
            max(limit * 20, 100),
        )
        candidate_limit = min(
            _positive_int(
                requested_candidates,
                "candidate_limit",
            ),
            self.search_candidate_limit,
        )

        candidates: List[Dict[str, Any]] = []
        if requested_labels:
            seen: set[str] = set()
            for label in requested_labels:
                nodes = self.get_nodes(
                    labels=[str(label)],
                    properties=properties,
                    limit=candidate_limit,
                )
                for node in nodes:
                    key = json.dumps(
                        node.get("id"),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if key not in seen:
                        seen.add(key)
                        candidates.append(node)
        else:
            candidates = self.get_nodes(
                properties=properties,
                limit=candidate_limit,
            )

        text_properties = options.get("text_properties") or self.search_properties
        if isinstance(text_properties, str):
            text_properties = [
                item.strip() for item in text_properties.split(",") if item.strip()
            ]

        results: List[Dict[str, Any]] = []
        for node in candidates:
            props = dict(node.get("properties") or {})
            selected = [props.get(key) for key in text_properties if key in props]
            values = list(_text_fragments(selected or props))
            score = _lexical_score(query, values)
            if query.strip() and score <= 0.0:
                continue
            content = next(
                (value for value in values if value),
                "",
            )
            labels_value = list(node.get("labels") or [])
            node_type = labels_value[0] if labels_value else None
            results.append(
                {
                    "id": node.get("id"),
                    "type": node_type,
                    "content": content,
                    "score": score,
                    "metadata": {
                        **props,
                        "labels": labels_value,
                    },
                    "node": {
                        "id": node.get("id"),
                        "type": node_type,
                        "content": content,
                        "properties": props,
                    },
                }
            )
        results.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("id")),
            )
        )
        return results[:limit]

    def execute_native_algorithm(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call one supported traverser without building an algorithm layer."""
        normalized = str(name).lower()
        methods = self._NATIVE_ALGORITHMS.get(normalized)
        if methods is None:
            supported = ", ".join(sorted(self._NATIVE_ALGORITHMS))
            raise ValidationError(
                f"Unsupported HugeGraph native algorithm {name!r}; "
                f"supported: {supported}"
            )
        try:
            return _plain(
                _call(
                    self._manager("traverser"),
                    methods,
                    *args,
                    **kwargs,
                )
            )
        except Exception as exc:
            raise ProcessingError(
                f"HugeGraph native algorithm {name!r} failed: {exc}"
            ) from exc

    def get_stats(self) -> Dict[str, Any]:
        scripts = {
            "node_count": "g.V().count().next()",
            "relationship_count": "g.E().count().next()",
            "label_counts": "g.V().groupCount().by(label).next()",
            "relationship_type_counts": ("g.E().groupCount().by(label).next()"),
        }
        output: Dict[str, Any] = {}
        for key, script in scripts.items():
            try:
                data = _result_data(self._run_gremlin(script))
                if isinstance(data, list) and len(data) == 1:
                    data = data[0]
                output[key] = _plain(data)
            except Exception as exc:
                self.logger.warning(
                    "Failed to read HugeGraph statistic %s: %s",
                    key,
                    exc,
                )
                output[key] = 0 if key.endswith("count") else {}
        return output

