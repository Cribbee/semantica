from __future__ import annotations

import pytest

from semantica.graph_store.hugegraph_store import HugeGraphStore
from semantica.utils.exceptions import ProcessingError, ValidationError


class ReplaceManager:
    def __init__(self):
        self.vertex = {"name": "Alice", "obsolete": "old"}
        self.edge = {"source": "doc-1", "obsolete": "old"}
        self.calls = []
        self.fail_append_once = False
        self.fail_eliminate_once = False

    def _append(self, kind, properties):
        self.calls.append((f"append_{kind}", dict(properties)))
        if self.fail_append_once:
            self.fail_append_once = False
            raise RuntimeError("append failed")
        target = self.vertex if kind == "vertex" else self.edge
        target.update(properties)
        return target

    def _eliminate(self, kind, properties):
        self.calls.append((f"eliminate_{kind}", dict(properties)))
        target = self.vertex if kind == "vertex" else self.edge
        for key, value in properties.items():
            if target.get(key) == value:
                target.pop(key, None)
        if self.fail_eliminate_once:
            self.fail_eliminate_once = False
            raise RuntimeError("eliminate failed")
        return target

    def appendVertex(self, _entity_id, properties):
        return self._append("vertex", properties)

    def eliminateVertex(self, _entity_id, properties):
        return self._eliminate("vertex", properties)

    def appendEdge(self, _entity_id, properties):
        return self._append("edge", properties)

    def eliminateEdge(self, _entity_id, properties):
        return self._eliminate("edge", properties)


class ReplaceStore(HugeGraphStore):
    def __init__(self, manager):
        super().__init__(user="", password="", client=object())
        self.manager = manager

    def _manager(self, name):
        assert name == "graph"
        return self.manager

    def get_node(self, node_id, **options):
        del options
        return {
            "id": node_id,
            "labels": ["Actor"],
            "properties": dict(self.manager.vertex),
        }

    def get_relationship(self, rel_id, **options):
        del options
        return {
            "id": rel_id,
            "type": "SUPPORTED_BY",
            "start_node_id": "claim:1",
            "end_node_id": "doc:1",
            "properties": dict(self.manager.edge),
        }


def test_replace_writes_new_values_before_removing_stale_properties():
    manager = ReplaceManager()
    store = ReplaceStore(manager)

    result = store.update_node(
        "actor:1",
        {"name": "Alicia", "role": "Analyst"},
        merge=False,
    )

    assert result["properties"] == {"name": "Alicia", "role": "Analyst"}
    assert manager.calls == [
        ("append_vertex", {"name": "Alicia", "role": "Analyst"}),
        ("eliminate_vertex", {"obsolete": "old"}),
    ]


def test_replace_append_failure_keeps_previous_properties():
    manager = ReplaceManager()
    manager.fail_append_once = True
    store = ReplaceStore(manager)

    with pytest.raises(ProcessingError, match="previous properties restored"):
        store.update_node("actor:1", {"name": "Alicia"}, merge=False)

    assert manager.vertex == {"name": "Alice", "obsolete": "old"}


def test_replace_eliminate_failure_rolls_back_changed_and_new_values():
    manager = ReplaceManager()
    manager.fail_eliminate_once = True
    store = ReplaceStore(manager)

    with pytest.raises(ProcessingError, match="previous properties restored"):
        store.update_node(
            "actor:1",
            {"name": "Alicia", "role": "Analyst"},
            merge=False,
        )

    assert manager.vertex == {"name": "Alice", "obsolete": "old"}


def test_relationship_replace_uses_the_same_recoverable_contract():
    manager = ReplaceManager()
    store = ReplaceStore(manager)

    result = store.update_relationship(
        "edge:1",
        {"source": "doc-2", "quote": "verified"},
        merge=False,
    )

    assert result["properties"] == {
        "source": "doc-2",
        "quote": "verified",
    }
    assert manager.calls[0][0] == "append_edge"
    assert manager.calls[1] == ("eliminate_edge", {"obsolete": "old"})


def test_invalid_property_key_fails_before_any_write():
    manager = ReplaceManager()
    store = ReplaceStore(manager)

    with pytest.raises(ValidationError):
        store.update_node("actor:1", {"not valid!": 1}, merge=False)

    assert manager.calls == []
