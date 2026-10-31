"""Bridge HugeGraph retrieval results to Semantica's ``GraphReasoner``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, Optional


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    data = getattr(value, "__dict__", None)
    return dict(data) if isinstance(data, Mapping) else {}


def _items(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return value
    return (value,)


def hugegraph_contexts_to_reasoning_graph(
    contexts: Iterable[Any],
) -> Dict[str, Any]:
    """Build a ``GraphReasoner`` graph without discarding evidence metadata."""
    entities: Dict[str, Dict[str, Any]] = {}
    relationships: Dict[str, Dict[str, Any]] = {}

    def add_entity(
        entity_id: Any,
        *,
        entity_type: Optional[str] = None,
        content: Optional[str] = None,
        properties: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if entity_id is None:
            return
        key = str(entity_id)
        current = entities.setdefault(
            key,
            {
                "id": entity_id,
                "name": content or key,
                "type": entity_type or "Entity",
                "properties": {},
            },
        )
        if content:
            current["name"] = content
        if entity_type:
            current["type"] = entity_type
        if properties:
            current["properties"].update(dict(properties))

    def add_relationship(raw: Any) -> None:
        relation = _mapping(raw)
        if not relation:
            return
        source = relation.get("source", relation.get("source_id"))
        target = relation.get("target", relation.get("target_id"))
        rel_type = relation.get("type", relation.get("relationship"))
        if source is None or target is None or not rel_type:
            return
        rel_id = relation.get("id")
        marker = str(
            rel_id
            if rel_id is not None
            else (source, target, rel_type, relation.get("direction"))
        )
        properties = dict(relation.get("properties") or {})
        for key in (
            "direction",
            "quote",
            "source_url",
            "source_id",
            "timestamp",
            "published_at",
            "provenance",
            "evidence",
        ):
            if key in relation and key not in properties:
                properties[key] = relation[key]
        relationships[marker] = {
            "id": rel_id,
            "source": source,
            "target": target,
            "type": rel_type,
            "properties": properties,
        }
        add_entity(source)
        add_entity(target)

    for context in contexts:
        item = _mapping(context)
        metadata = _mapping(item.get("metadata"))
        node_id = metadata.get("node_id", item.get("id"))
        source = item.get("source")
        if node_id is None and isinstance(source, str) and source.startswith("graph:"):
            node_id = source.split(":", 1)[1]
        content = item.get("content") or metadata.get("content")
        node_type = metadata.get("node_type", item.get("type"))
        properties = {
            **metadata,
            "retrieval_source": source,
            "retrieval_score": item.get("score"),
        }
        add_entity(
            node_id,
            entity_type=node_type,
            content=content,
            properties=properties,
        )

        for related in _items(item.get("related_entities")):
            related_item = _mapping(related)
            related_metadata = _mapping(related_item.get("metadata"))
            related_id = related_item.get("id", related_metadata.get("node_id"))
            add_entity(
                related_id,
                entity_type=related_item.get(
                    "type", related_metadata.get("node_type")
                ),
                content=related_item.get("content"),
                properties=related_metadata,
            )
            for relation in _items(related_item.get("relationships")):
                add_relationship(relation)
            for relation in _items(related_metadata.get("path_relationships")):
                add_relationship(relation)

        for relation in _items(item.get("related_relationships")):
            add_relationship(relation)
        for relation in _items(metadata.get("path_relationships")):
            add_relationship(relation)

    return {
        "entities": list(entities.values()),
        "relationships": list(relationships.values()),
    }


def reason_over_hugegraph_contexts(
    reasoner: Any,
    contexts: Iterable[Any],
    query: str,
    **options: Any,
) -> str:
    """Run the existing ``GraphReasoner`` over retrieved HugeGraph evidence."""
    graph = hugegraph_contexts_to_reasoning_graph(contexts)
    return reasoner.reason(graph, query, **options)
