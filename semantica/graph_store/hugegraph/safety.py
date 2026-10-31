"""Failure-safe replacement updates for the HugeGraph backend."""

from __future__ import annotations

from typing import Any, Dict

from ...utils.exceptions import ProcessingError
from .utils import NodeId, _call, _identifier


class HugeGraphSafetyMixin:
    """Add best-effort rollback to ``merge=False`` property replacement.

    HugeGraph exposes append/eliminate property operations rather than a single
    replace request. The adapter writes the desired values first, removes stale
    values second, and restores the original property set when either operation
    fails. Vertex and edge identities are never recreated.
    """

    @staticmethod
    def _validated_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
        validated: Dict[str, Any] = {}
        for key, value in dict(properties).items():
            validated[_identifier(str(key), "property key")] = value
        return validated

    @staticmethod
    def _rollback_properties(
        manager: Any,
        object_id: NodeId,
        original: Dict[str, Any],
        replacement: Dict[str, Any],
        *,
        append_names: tuple[str, ...],
        eliminate_names: tuple[str, ...],
    ) -> None:
        if original:
            _call(manager, append_names, object_id, dict(original))
        introduced = {
            key: value for key, value in replacement.items() if key not in original
        }
        if introduced:
            _call(manager, eliminate_names, object_id, introduced)

    def update_node(
        self,
        node_id: NodeId,
        properties: Dict[str, Any],
        merge: bool = True,
        **options: Any,
    ) -> Dict[str, Any]:
        if merge:
            return super().update_node(node_id, properties, merge=True, **options)

        self._ensure_write("update_node")
        replacement = self._validated_properties(properties)
        current = self.get_node(node_id)
        if current is None:
            raise ProcessingError(f"HugeGraph vertex {node_id!r} was not found")
        original = dict(current.get("properties") or {})
        manager = self._manager("graph")
        try:
            if replacement:
                _call(
                    manager,
                    ("append_vertex", "appendVertex"),
                    node_id,
                    replacement,
                )
            stale = {
                key: value
                for key, value in original.items()
                if key not in replacement
            }
            if stale:
                _call(
                    manager,
                    ("eliminate_vertex", "eliminateVertex"),
                    node_id,
                    stale,
                )
            refreshed = self.get_node(node_id)
            if refreshed is None:
                raise ProcessingError(f"HugeGraph vertex {node_id!r} disappeared")
            return refreshed
        except Exception as exc:
            rollback_error = None
            try:
                self._rollback_properties(
                    manager,
                    node_id,
                    original,
                    replacement,
                    append_names=("append_vertex", "appendVertex"),
                    eliminate_names=("eliminate_vertex", "eliminateVertex"),
                )
            except Exception as rollback_exc:  # pragma: no cover - rare double fault
                rollback_error = rollback_exc
            detail = (
                f"; rollback also failed: {rollback_error}"
                if rollback_error
                else "; previous properties restored"
            )
            raise ProcessingError(
                f"Failed to replace HugeGraph vertex {node_id!r} properties: "
                f"{exc}{detail}"
            ) from exc

    def update_relationship(
        self,
        rel_id: NodeId,
        properties: Dict[str, Any],
        merge: bool = True,
        **options: Any,
    ) -> Dict[str, Any]:
        if merge:
            return super().update_relationship(
                rel_id, properties, merge=True, **options
            )

        self._ensure_write("update_relationship")
        replacement = self._validated_properties(properties)
        current = self.get_relationship(rel_id)
        if current is None:
            raise ProcessingError(f"HugeGraph edge {rel_id!r} was not found")
        original = dict(current.get("properties") or {})
        manager = self._manager("graph")
        try:
            if replacement:
                _call(
                    manager,
                    ("append_edge", "appendEdge"),
                    rel_id,
                    replacement,
                )
            stale = {
                key: value
                for key, value in original.items()
                if key not in replacement
            }
            if stale:
                _call(
                    manager,
                    ("eliminate_edge", "eliminateEdge"),
                    rel_id,
                    stale,
                )
            refreshed = self.get_relationship(rel_id)
            if refreshed is None:
                raise ProcessingError(f"HugeGraph edge {rel_id!r} disappeared")
            return refreshed
        except Exception as exc:
            rollback_error = None
            try:
                self._rollback_properties(
                    manager,
                    rel_id,
                    original,
                    replacement,
                    append_names=("append_edge", "appendEdge"),
                    eliminate_names=("eliminate_edge", "eliminateEdge"),
                )
            except Exception as rollback_exc:  # pragma: no cover - rare double fault
                rollback_error = rollback_exc
            detail = (
                f"; rollback also failed: {rollback_error}"
                if rollback_error
                else "; previous properties restored"
            )
            raise ProcessingError(
                f"Failed to replace HugeGraph edge {rel_id!r} properties: {exc}{detail}"
            ) from exc
