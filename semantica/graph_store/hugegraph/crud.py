"""Node and relationship CRUD and native batch operations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

from ...utils.exceptions import ProcessingError, ValidationError
from .utils import (
    NodeId,
    _call,
    _edge_dict,
    _identifier,
    _node_dict,
    _positive_int,
)


class HugeGraphCrudMixin:
    """Implement node and relationship operations with normalized results."""

    @staticmethod
    def _physical_label(labels: Sequence[str]) -> str:
        if not labels:
            raise ValidationError("HugeGraph node creation requires one vertex label")
        if len(labels) != 1:
            raise ValidationError(
                "HugeGraph vertices have one physical label; map semantic "
                "labels explicitly instead of passing multiple labels"
            )
        return _identifier(str(labels[0]), "vertex label")

    def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any],
        **options: Any,
    ) -> Dict[str, Any]:
        self._ensure_write("create_node")
        label = self._physical_label(labels)
        physical_id = options.get("vertex_id", options.get("physical_id"))
        try:
            value = _call(
                self._manager("graph"),
                ("add_vertex", "addVertex"),
                label,
                dict(properties),
                physical_id,
            )
        except Exception as exc:
            raise ProcessingError(f"Failed to create HugeGraph vertex: {exc}") from exc
        if value is None:
            raise ProcessingError("HugeGraph returned no vertex after creation")
        return _node_dict(
            value,
            labels=[label],
            properties=properties,
        )

    def create_nodes(
        self,
        nodes: List[Dict[str, Any]],
        **options: Any,
    ) -> List[Dict[str, Any]]:
        self._ensure_write("create_nodes")
        if not nodes:
            return []

        prepared: List[Tuple[str, Dict[str, Any], Optional[NodeId]]] = []
        for node in nodes:
            labels = node.get("labels") or []
            if isinstance(labels, str):
                labels = [labels]
            physical_id = node.get(
                "vertex_id",
                node.get("physical_id", node.get("id")),
            )
            prepared.append(
                (
                    self._physical_label(list(labels)),
                    dict(node.get("properties") or {}),
                    physical_id,
                )
            )

        has_custom_ids = any(physical_id is not None for _, _, physical_id in prepared)
        if not options.get("use_native_batch", True) or has_custom_ids:
            return [
                self.create_node(
                    [label],
                    properties,
                    vertex_id=physical_id,
                )
                for label, properties, physical_id in prepared
            ]

        native_rows = [(label, properties) for label, properties, _ in prepared]
        try:
            values = _call(
                self._manager("graph"),
                ("add_vertices", "addVertices"),
                native_rows,
            )
        except Exception as exc:
            raise ProcessingError(
                f"Failed to create HugeGraph vertices in batch: {exc}"
            ) from exc
        if values is None or len(values) != len(prepared):
            raise ProcessingError(
                "HugeGraph batch vertex response does not match the request size"
            )
        return [
            _node_dict(
                value,
                labels=[label],
                properties=properties,
            )
            for value, (label, properties, _) in zip(values, prepared)
        ]

    def get_node(
        self,
        node_id: NodeId,
        **options: Any,
    ) -> Optional[Dict[str, Any]]:
        try:
            value = _call(
                self._manager("graph"),
                ("get_vertex_by_id", "getVertexById"),
                node_id,
            )
        except Exception as exc:
            if type(exc).__name__ in {
                "NotFoundError",
                "ResourceNotFoundError",
            }:
                return None
            raise ProcessingError(
                f"Failed to get HugeGraph vertex {node_id!r}: {exc}"
            ) from exc
        return None if value is None else _node_dict(value)

    def get_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **options: Any,
    ) -> List[Dict[str, Any]]:
        limit = _positive_int(limit, "limit")
        labels = list(labels or [])
        if len(labels) > 1:
            raise ValidationError(
                "HugeGraph has one physical vertex label; get_nodes accepts "
                "at most one label"
            )
        label = _identifier(labels[0], "vertex label") if labels else ""
        try:
            values = _call(
                self._manager("graph"),
                (
                    "get_vertex_by_condition",
                    "getVertexByCondition",
                ),
                label,
                limit,
                None,
                dict(properties or {}),
            )
        except Exception as exc:
            raise ProcessingError(f"Failed to query HugeGraph vertices: {exc}") from exc
        return [_node_dict(value) for value in values or []][:limit]

    def update_node(
        self,
        node_id: NodeId,
        properties: Dict[str, Any],
        merge: bool = True,
        **options: Any,
    ) -> Dict[str, Any]:
        self._ensure_write("update_node")
        manager = self._manager("graph")
        try:
            if merge:
                value = _call(
                    manager,
                    ("append_vertex", "appendVertex"),
                    node_id,
                    dict(properties),
                )
            else:
                current = self.get_node(node_id)
                if current is None:
                    raise ProcessingError(f"HugeGraph vertex {node_id!r} was not found")
                old_properties = dict(current.get("properties") or {})
                if properties:
                    _call(
                        manager,
                        ("append_vertex", "appendVertex"),
                        node_id,
                        dict(properties),
                    )
                obsolete = {
                    key: value
                    for key, value in old_properties.items()
                    if key not in properties
                }
                if obsolete:
                    _call(
                        manager,
                        ("eliminate_vertex", "eliminateVertex"),
                        node_id,
                        obsolete,
                    )
                value = self.get_node(node_id)
            if value is None:
                refreshed = self.get_node(node_id)
                if refreshed is None:
                    raise ProcessingError(f"HugeGraph vertex {node_id!r} was not found")
                return refreshed
            return _node_dict(value, properties=properties)
        except (ValidationError, ProcessingError):
            raise
        except Exception as exc:
            raise ProcessingError(
                f"Failed to update HugeGraph vertex {node_id!r}: {exc}"
            ) from exc

    def delete_node(
        self,
        node_id: NodeId,
        detach: bool = True,
        **options: Any,
    ) -> bool:
        self._ensure_write("delete_node")
        if not detach and self.get_relationships(
            node_id=node_id,
            limit=1,
        ):
            raise ValidationError(
                "detach=False cannot delete a HugeGraph vertex that still "
                "has relationships"
            )
        try:
            _call(
                self._manager("graph"),
                ("remove_vertex_by_id", "removeVertexById"),
                node_id,
            )
            return True
        except Exception as exc:
            raise ProcessingError(
                f"Failed to delete HugeGraph vertex {node_id!r}: {exc}"
            ) from exc

    def create_relationship(
        self,
        start_node_id: NodeId,
        end_node_id: NodeId,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        self._ensure_write("create_relationship")
        rel_type = _identifier(rel_type, "edge label")
        try:
            value = _call(
                self._manager("graph"),
                ("add_edge", "addEdge"),
                rel_type,
                start_node_id,
                end_node_id,
                dict(properties or {}),
            )
        except Exception as exc:
            raise ProcessingError(f"Failed to create HugeGraph edge: {exc}") from exc
        if value is None:
            raise ProcessingError("HugeGraph returned no edge after creation")
        return _edge_dict(
            value,
            rel_type=rel_type,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            properties=properties,
        )

    def create_relationships(
        self,
        relationships: List[Dict[str, Any]],
        **options: Any,
    ) -> List[Dict[str, Any]]:
        """Create edges in one batch when endpoint labels can be resolved."""
        self._ensure_write("create_relationships")
        if not relationships:
            return []
        if not options.get("use_native_batch", True):
            return [
                self.create_relationship(
                    relationship.get(
                        "start_id",
                        relationship.get("start_node_id"),
                    ),
                    relationship.get(
                        "end_id",
                        relationship.get("end_node_id"),
                    ),
                    relationship.get(
                        "type",
                        relationship.get("rel_type"),
                    ),
                    relationship.get("properties"),
                )
                for relationship in relationships
            ]

        label_cache: Dict[NodeId, str] = {}

        def endpoint_label(
            node_id: NodeId,
            explicit: Optional[str],
        ) -> str:
            if explicit:
                return _identifier(str(explicit), "vertex label")
            if node_id in label_cache:
                return label_cache[node_id]
            node = self.get_node(node_id)
            labels = list((node or {}).get("labels") or [])
            if not labels:
                raise ValidationError(
                    "Cannot resolve the HugeGraph vertex label for endpoint "
                    f"{node_id!r}"
                )
            label_cache[node_id] = _identifier(
                str(labels[0]),
                "vertex label",
            )
            return label_cache[node_id]

        prepared: List[Tuple[str, NodeId, NodeId, str, str, Dict[str, Any]]] = []
        normalized: List[Tuple[str, NodeId, NodeId, Dict[str, Any]]] = []
        for relationship in relationships:
            start = relationship.get(
                "start_id",
                relationship.get("start_node_id"),
            )
            end = relationship.get(
                "end_id",
                relationship.get("end_node_id"),
            )
            rel_type = relationship.get(
                "type",
                relationship.get("rel_type"),
            )
            if start is None or end is None or rel_type is None:
                raise ValidationError(
                    "Each relationship requires start_id, end_id and type"
                )
            rel_type = _identifier(str(rel_type), "edge label")
            properties = dict(relationship.get("properties") or {})
            out_label = endpoint_label(
                start,
                relationship.get(
                    "start_label",
                    relationship.get("outVLabel"),
                ),
            )
            in_label = endpoint_label(
                end,
                relationship.get(
                    "end_label",
                    relationship.get("inVLabel"),
                ),
            )
            prepared.append(
                (
                    rel_type,
                    start,
                    end,
                    out_label,
                    in_label,
                    properties,
                )
            )
            normalized.append((rel_type, start, end, properties))

        try:
            values = _call(
                self._manager("graph"),
                ("add_edges", "addEdges"),
                prepared,
            )
        except Exception as exc:
            raise ProcessingError(
                f"Failed to create HugeGraph edges in batch: {exc}"
            ) from exc
        if values is None or len(values) != len(prepared):
            raise ProcessingError(
                "HugeGraph batch edge response does not match the request size"
            )
        return [
            _edge_dict(
                value,
                rel_type=rel_type,
                start_node_id=start,
                end_node_id=end,
                properties=properties,
            )
            for value, (rel_type, start, end, properties) in zip(
                values,
                normalized,
            )
        ]

    def get_relationship(
        self,
        rel_id: NodeId,
        **options: Any,
    ) -> Optional[Dict[str, Any]]:
        try:
            value = _call(
                self._manager("graph"),
                ("get_edge_by_id", "getEdgeById"),
                rel_id,
            )
        except Exception as exc:
            if type(exc).__name__ in {
                "NotFoundError",
                "ResourceNotFoundError",
            }:
                return None
            raise ProcessingError(
                f"Failed to get HugeGraph edge {rel_id!r}: {exc}"
            ) from exc
        return None if value is None else _edge_dict(value)

    def update_relationship(
        self,
        rel_id: NodeId,
        properties: Dict[str, Any],
        merge: bool = True,
        **options: Any,
    ) -> Dict[str, Any]:
        self._ensure_write("update_relationship")
        manager = self._manager("graph")
        current = self.get_relationship(rel_id)
        if current is None:
            raise ProcessingError(f"HugeGraph edge {rel_id!r} was not found")
        try:
            if merge:
                value = _call(
                    manager,
                    ("append_edge", "appendEdge"),
                    rel_id,
                    dict(properties),
                )
            else:
                old_properties = dict(current.get("properties") or {})
                if properties:
                    _call(
                        manager,
                        ("append_edge", "appendEdge"),
                        rel_id,
                        dict(properties),
                    )
                obsolete = {
                    key: value
                    for key, value in old_properties.items()
                    if key not in properties
                }
                if obsolete:
                    _call(
                        manager,
                        ("eliminate_edge", "eliminateEdge"),
                        rel_id,
                        obsolete,
                    )
                value = self.get_relationship(rel_id)
            if value is None:
                refreshed = self.get_relationship(rel_id)
                if refreshed is None:
                    raise ProcessingError(f"HugeGraph edge {rel_id!r} was not found")
                return refreshed
            return _edge_dict(
                value,
                rel_type=current.get("type"),
                start_node_id=current.get("start_node_id"),
                end_node_id=current.get("end_node_id"),
                properties=properties,
            )
        except (ValidationError, ProcessingError):
            raise
        except Exception as exc:
            raise ProcessingError(
                f"Failed to update HugeGraph edge {rel_id!r}: {exc}"
            ) from exc

    @staticmethod
    def _direction(direction: str) -> str:
        normalized = str(direction).lower()
        if normalized not in {"in", "out", "both"}:
            raise ValidationError(f"Unsupported relationship direction: {direction!r}")
        return normalized

    def _edge_page(
        self,
        *,
        node_id: Optional[NodeId],
        rel_type: Optional[str],
        direction: Optional[str],
        limit: int,
        properties: Optional[Dict[str, Any]],
    ) -> List[Any]:
        value = _call(
            self._manager("graph"),
            ("get_edge_by_page", "getEdgeByPage"),
            rel_type,
            node_id,
            direction,
            limit,
            None,
            properties,
        )
        if isinstance(value, tuple):
            return list(value[0] or [])
        return list(value or [])

    def get_relationships(
        self,
        node_id: Optional[NodeId] = None,
        rel_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 100,
        **options: Any,
    ) -> List[Dict[str, Any]]:
        limit = _positive_int(limit, "limit")
        normalized_direction = self._direction(direction)
        if rel_type is not None:
            rel_type = _identifier(rel_type, "edge label")
        properties = options.get("properties")

        try:
            if node_id is None:
                values = self._edge_page(
                    node_id=None,
                    rel_type=rel_type,
                    direction=None,
                    limit=limit,
                    properties=properties,
                )
            elif normalized_direction == "both":
                outgoing = self._edge_page(
                    node_id=node_id,
                    rel_type=rel_type,
                    direction="OUT",
                    limit=limit,
                    properties=properties,
                )
                incoming = self._edge_page(
                    node_id=node_id,
                    rel_type=rel_type,
                    direction="IN",
                    limit=limit,
                    properties=properties,
                )
                values = [*outgoing, *incoming]
            else:
                values = self._edge_page(
                    node_id=node_id,
                    rel_type=rel_type,
                    direction=normalized_direction.upper(),
                    limit=limit,
                    properties=properties,
                )
        except Exception as exc:
            raise ProcessingError(f"Failed to query HugeGraph edges: {exc}") from exc

        unique: Dict[str, Dict[str, Any]] = {}
        for value in values:
            edge = _edge_dict(value)
            marker = edge.get("id")
            if marker is None:
                marker = [
                    edge.get("type"),
                    edge.get("start_node_id"),
                    edge.get("end_node_id"),
                    edge.get("properties"),
                ]
            key = json.dumps(
                marker,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            unique.setdefault(key, edge)
            if len(unique) >= limit:
                break
        return list(unique.values())

    def delete_relationship(
        self,
        rel_id: NodeId,
        **options: Any,
    ) -> bool:
        self._ensure_write("delete_relationship")
        try:
            _call(
                self._manager("graph"),
                ("remove_edge_by_id", "removeEdgeById"),
                rel_id,
            )
            return True
        except Exception as exc:
            raise ProcessingError(
                f"Failed to delete HugeGraph edge {rel_id!r}: {exc}"
            ) from exc

