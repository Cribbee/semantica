"""ContextRetriever adapter for an existing HugeGraph business graph.

``ContextRetriever`` expects ``knowledge_graph.query(question)`` to perform
semantic entity retrieval, while graph database backends use ``query`` for a
native query language. This adapter keeps those contracts separate:

* :meth:`query` accepts a natural-language question and delegates to
  :meth:`HugeGraphStore.search_nodes`;
* :meth:`execute_query` remains the explicit Gremlin entry point;
* :meth:`get_neighbors` expands only the selected local subgraph and carries
  path, direction, relationship properties and evidence attributes forward.

The adapter does not extract entities or build another business graph. It is a
read-through view over the existing HugeGraph data and can be combined with a
separate Semantica vector store for hybrid retrieval.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Optional, Union

from ..graph_store.hugegraph_store import HugeGraphStore
from ..utils.exceptions import ValidationError

NodeId = Union[int, str]

_SOURCE_ID_PROPERTIES = (
    "source",
    "source_id",
    "source_uri",
    "source_url",
    "document_id",
    "doc_id",
    "chunk_id",
)
_EVIDENCE_PROPERTIES = (
    "quote",
    "provenance",
    "evidence",
    "witness",
    "witnesses",
)


def _first_text(properties: Dict[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = properties.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return ""


def _values(properties: Dict[str, Any], names: Iterable[str]) -> List[Any]:
    output: List[Any] = []
    for name in names:
        if name not in properties:
            continue
        value = properties[name]
        if isinstance(value, list):
            output.extend(value)
        else:
            output.append(value)
    return output


def _path_marker(node: Dict[str, Any]) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    path = tuple(str(item) for item in node.get("path") or [])
    relationships = node.get("path_relationships") or []
    relationship_markers = tuple(
        ":".join(
            (
                str(relationship.get("id")),
                str(relationship.get("type")),
                str(relationship.get("start_node_id")),
                str(relationship.get("end_node_id")),
            )
        )
        for relationship in relationships
        if isinstance(relationship, dict)
    )
    return node.get("id"), path, relationship_markers


class HugeGraphContextAdapter:
    """Read-only semantic retrieval view over :class:`HugeGraphStore`."""

    def __init__(
        self,
        store: HugeGraphStore,
        *,
        node_labels: Optional[Sequence[str]] = None,
        text_properties: Optional[Sequence[str]] = None,
        content_properties: Optional[Sequence[str]] = None,
        source_properties: Optional[Sequence[str]] = None,
        evidence_properties: Optional[Sequence[str]] = None,
        time_properties: Optional[Sequence[str]] = None,
        relationship_types: Optional[Sequence[str]] = None,
        direction: str = "both",
        max_results: int = 20,
        max_hops: int = 2,
        candidate_limit: Optional[int] = None,
    ) -> None:
        self.store = store
        self.node_labels = list(node_labels or [])
        self.text_properties = list(text_properties or store.search_properties)
        self.content_properties = list(
            content_properties
            or (
                "content",
                "text",
                "statement",
                "summary",
                "description",
                "title",
                "name",
            )
        )
        # ``sources`` is deliberately identity-only. Quotes and provenance
        # payloads remain available through the original properties and the
        # separate ``evidence_values`` metadata field, so callers cannot mistake
        # an entire quote for a source identifier.
        self.source_properties = list(source_properties or _SOURCE_ID_PROPERTIES)
        self.evidence_properties = list(
            evidence_properties or _EVIDENCE_PROPERTIES
        )
        self.time_properties = list(
            time_properties
            or (
                "time",
                "timestamp",
                "event_time",
                "published_at",
                "valid_from",
                "valid_to",
            )
        )
        self.relationship_types = list(relationship_types or [])
        normalized_direction = direction.lower()
        if normalized_direction not in {"in", "out", "both"}:
            raise ValidationError(
                "Unsupported HugeGraph context direction: " f"{direction!r}"
            )
        self.direction = normalized_direction
        self.max_results = max(1, int(max_results))
        self.max_hops = max(1, int(max_hops))
        self.candidate_limit = candidate_limit

    def _semantic_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        properties = dict(node.get("properties") or {})
        labels = list(node.get("labels") or [])
        node_type = labels[0] if labels else None
        content = _first_text(properties, self.content_properties)
        if not content:
            content = f"{node_type or 'Node'}:{node.get('id')}"
        metadata = {
            **properties,
            "labels": labels,
            "node_id": node.get("id"),
            "node_type": node_type,
            "sources": _values(properties, self.source_properties),
            "evidence_values": _values(properties, self.evidence_properties),
            "time_constraints": _values(properties, self.time_properties),
            "backend": "hugegraph",
        }
        for key in ("distance", "path", "path_relationships"):
            if key in node:
                metadata[key] = node[key]
        return {
            "id": node.get("id"),
            "type": node_type,
            "content": content,
            "metadata": metadata,
            "node": {
                "id": node.get("id"),
                "type": node_type,
                "content": content,
                "properties": metadata,
            },
        }

    def query(self, query: str, **options: Any) -> List[Dict[str, Any]]:
        """Retrieve entry vertices for a natural-language question.

        This method intentionally does not execute Gremlin. Raw database queries
        must go through :meth:`execute_query`.
        """
        limit = max(
            1,
            int(
                options.pop(
                    "limit",
                    options.pop("max_results", self.max_results),
                )
            ),
        )
        labels = options.pop(
            "labels",
            options.pop("node_labels", self.node_labels),
        )
        candidate_limit = options.pop(
            "candidate_limit",
            self.candidate_limit,
        )
        search_options: Dict[str, Any] = {
            "text_properties": options.pop(
                "text_properties",
                self.text_properties,
            ),
            **options,
        }
        if candidate_limit is not None:
            search_options["candidate_limit"] = candidate_limit
        raw_results = self.store.search_nodes(
            query,
            limit=limit,
            labels=list(labels or []),
            **search_options,
        )
        results: List[Dict[str, Any]] = []
        for raw in raw_results:
            raw_node = raw.get("node") or {
                "id": raw.get("id"),
                "labels": [raw.get("type")] if raw.get("type") else [],
                "properties": raw.get("metadata") or {},
            }
            if "labels" not in raw_node:
                raw_node = {
                    "id": raw_node.get("id"),
                    "labels": (
                        [raw_node.get("type")] if raw_node.get("type") else []
                    ),
                    "properties": (
                        raw_node.get("properties")
                        or raw.get("metadata")
                        or {}
                    ),
                }
            semantic = self._semantic_node(raw_node)
            if isinstance(raw.get("content"), str) and raw["content"].strip():
                semantic["content"] = raw["content"].strip()
                semantic["node"]["content"] = raw["content"].strip()
            semantic["score"] = float(raw.get("score", 0.0))
            raw_metadata = raw.get("metadata")
            normalized_metadata = dict(semantic["metadata"])
            semantic["metadata"] = {
                **(dict(raw_metadata) if isinstance(raw_metadata, dict) else {}),
                **normalized_metadata,
            }
            semantic["node"]["properties"] = semantic["metadata"]
            results.append(semantic)
        return results

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        """Execute explicit Gremlin without semantic-query overloading."""
        return self.store.execute_query(query, parameters, **options)

    def _relationship_view(
        self,
        relationship: Dict[str, Any],
        previous_node_id: Optional[NodeId],
    ) -> Dict[str, Any]:
        start = relationship.get("start_node_id")
        end = relationship.get("end_node_id")
        if previous_node_id == start:
            traversal_direction = "out"
        elif previous_node_id == end:
            traversal_direction = "in"
        else:
            traversal_direction = "unknown"
        return {
            "id": relationship.get("id"),
            "type": relationship.get("type"),
            "source": start,
            "target": end,
            "direction": traversal_direction,
            "properties": dict(relationship.get("properties") or {}),
        }

    @staticmethod
    def _neighbor_for_edge(
        edge: Dict[str, Any],
        current: NodeId,
        direction: str,
    ) -> Optional[NodeId]:
        start = edge.get("start_node_id")
        end = edge.get("end_node_id")
        if direction == "out":
            return end if start == current else None
        if direction == "in":
            return start if end == current else None
        if start == current:
            return end
        if end == current:
            return start
        return None

    def _bounded_neighbor_paths(
        self,
        node_id: NodeId,
        *,
        depth: int,
        direction: str,
        relationship_types: Sequence[str],
        per_node_limit: int,
        max_paths: int,
        relationship_properties: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Enumerate simple bounded paths while preserving parallel evidence."""
        allowed_types = set(relationship_types)
        edge_cache: Dict[NodeId, List[Dict[str, Any]]] = {}
        node_cache: Dict[NodeId, Optional[Dict[str, Any]]] = {}
        queue = deque([(node_id, (node_id,), tuple())])
        seen_paths: set[
            tuple[Any, tuple[str, ...], tuple[str, ...]]
        ] = set()
        output: List[Dict[str, Any]] = []

        while queue and len(output) < max_paths:
            current, path, path_relationships = queue.popleft()
            if len(path_relationships) >= depth:
                continue
            if current not in edge_cache:
                relationship_options: Dict[str, Any] = {}
                if relationship_properties is not None:
                    relationship_options["properties"] = relationship_properties
                edge_cache[current] = self.store.get_relationships(
                    node_id=current,
                    direction=direction,
                    limit=per_node_limit,
                    **relationship_options,
                )
            edges = sorted(
                edge_cache[current],
                key=lambda edge: (
                    str(edge.get("id")),
                    str(edge.get("type")),
                    str(edge.get("start_node_id")),
                    str(edge.get("end_node_id")),
                ),
            )
            for edge in edges:
                if allowed_types and edge.get("type") not in allowed_types:
                    continue
                neighbor = self._neighbor_for_edge(edge, current, direction)
                if neighbor is None or neighbor in path:
                    continue
                next_path = (*path, neighbor)
                next_relationships = (*path_relationships, edge)
                candidate = {
                    "id": neighbor,
                    "path": list(next_path),
                    "path_relationships": list(next_relationships),
                }
                marker = _path_marker(candidate)
                if marker in seen_paths:
                    continue
                seen_paths.add(marker)
                if neighbor not in node_cache:
                    node_cache[neighbor] = self.store.get_node(neighbor)
                node = node_cache[neighbor]
                if node is None:
                    continue
                result = {
                    **dict(node),
                    "distance": len(next_relationships),
                    "path": list(next_path),
                    "path_relationships": list(next_relationships),
                }
                output.append(result)
                if len(output) >= max_paths:
                    break
                if len(next_relationships) < depth:
                    queue.append((neighbor, next_path, next_relationships))
        return output

    def get_neighbors(
        self,
        node_id: NodeId,
        hops: Optional[int] = None,
        **options: Any,
    ) -> List[Dict[str, Any]]:
        """Return entities from one bounded, provenance-rich local subgraph."""
        depth = max(1, int(options.pop("depth", hops or self.max_hops)))
        direction = str(options.pop("direction", self.direction)).lower()
        if direction not in {"in", "out", "both"}:
            raise ValidationError(
                "Unsupported HugeGraph context direction: " f"{direction!r}"
            )
        relationship_types = options.pop(
            "relationship_types",
            self.relationship_types,
        )
        if isinstance(relationship_types, str):
            relationship_types = [relationship_types]
        per_node_limit = max(1, int(options.pop("per_node_limit", 1000)))
        max_paths = max(1, int(options.pop("max_paths", 1000)))
        relationship_properties = options.pop(
            "relationship_properties",
            options.pop("properties", None),
        )
        if relationship_properties is not None and not isinstance(
            relationship_properties, dict
        ):
            raise ValidationError(
                "relationship_properties must be a property mapping"
            )

        raw_neighbors = self._bounded_neighbor_paths(
            node_id,
            depth=depth,
            direction=direction,
            relationship_types=list(relationship_types or []),
            per_node_limit=per_node_limit,
            max_paths=max_paths,
            relationship_properties=relationship_properties,
        )

        output: List[Dict[str, Any]] = []
        for raw in raw_neighbors:
            semantic = self._semantic_node(raw)
            path = list(raw.get("path") or [])
            raw_relationships = list(raw.get("path_relationships") or [])
            relationships: List[Dict[str, Any]] = []
            for index, relationship in enumerate(raw_relationships):
                previous = path[index] if index < len(path) else None
                relationships.append(
                    self._relationship_view(relationship, previous)
                )
            last_relationship = relationships[-1] if relationships else None
            metadata = dict(semantic["metadata"])
            metadata.update(
                {
                    "path": path,
                    "path_relationships": relationships,
                    "hop": raw.get("distance"),
                    "entry_node_id": node_id,
                }
            )
            output.append(
                {
                    "id": semantic["id"],
                    "type": semantic["type"],
                    "content": semantic["content"],
                    "metadata": metadata,
                    "relationship": (
                        last_relationship.get("type")
                        if last_relationship
                        else None
                    ),
                    "relationship_direction": (
                        last_relationship.get("direction")
                        if last_relationship
                        else None
                    ),
                    "hop": raw.get("distance"),
                    "path": path,
                    "relationships": relationships,
                }
            )
        return output

    def get_path(
        self,
        source_id: NodeId,
        target_id: NodeId,
        max_hops: int = 5,
        **options: Any,
    ) -> Optional[Dict[str, Any]]:
        """Return a path with evidence-bearing relationship properties."""
        return self.store.shortest_path(
            source_id,
            target_id,
            max_depth=max_hops,
            **options,
        )

    def get_schema(self) -> Dict[str, Any]:
        return self.store.get_schema()

    def get_nodes(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.store.get_nodes(*args, **kwargs)

    def get_relationships(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        return self.store.get_relationships(*args, **kwargs)

