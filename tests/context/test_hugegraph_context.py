from __future__ import annotations

from copy import deepcopy

from semantica.context.hugegraph_context import HugeGraphContextAdapter


class Store:
    search_properties = ("content", "name")

    def __init__(self):
        self.search_calls = []
        self.relationship_calls = []
        self.query_calls = []
        self.nodes = {
            "claim:1": {
                "id": "claim:1",
                "labels": ["Claim"],
                "properties": {
                    "statement": "A sourced claim",
                    "source_id": "doc:1",
                    "quote": "A quote is evidence, not a source id",
                    "published_at": "2026-01-02",
                },
            },
            "event:1": {
                "id": "event:1",
                "labels": ["Event"],
                "properties": {"name": "Investigation"},
            },
            "actor:2": {
                "id": "actor:2",
                "labels": ["Actor"],
                "properties": {
                    "name": "Agency",
                    "source_url": "https://source.invalid",
                },
            },
        }
        self.edges = [
            {
                "id": "edge:1",
                "type": "SUPPORTED_BY",
                "start_node_id": "claim:1",
                "end_node_id": "event:1",
                "properties": {"quote": "source quote"},
            },
            {
                "id": "edge:2",
                "type": "INVOLVES",
                "start_node_id": "event:1",
                "end_node_id": "actor:2",
                "properties": {"time": "2026-01-02"},
            },
        ]

    def search_nodes(self, query, limit, labels, **options):
        self.search_calls.append((query, limit, labels, options))
        return [
            {
                "id": "claim:1",
                "type": "Claim",
                "content": "A sourced claim",
                "score": 0.91,
                "metadata": {
                    "source_id": "doc:1",
                    "node_id": "wrong:node",
                    "sources": ["wrong:source"],
                    "backend": "wrong-backend",
                    "extra": "preserved",
                },
                "node": {
                    "id": "claim:1",
                    "type": "Claim",
                    "properties": deepcopy(self.nodes["claim:1"]["properties"]),
                },
            }
        ]

    def get_node(self, node_id):
        value = self.nodes.get(node_id)
        return deepcopy(value) if value is not None else None

    def get_relationships(
        self,
        node_id=None,
        rel_type=None,
        direction="both",
        limit=100,
        **options,
    ):
        self.relationship_calls.append(
            (node_id, rel_type, direction, limit, options)
        )
        output = []
        for edge in self.edges:
            if rel_type and edge["type"] != rel_type:
                continue
            if node_id is not None:
                if direction == "out" and edge["start_node_id"] != node_id:
                    continue
                if direction == "in" and edge["end_node_id"] != node_id:
                    continue
                if direction == "both" and node_id not in {
                    edge["start_node_id"],
                    edge["end_node_id"],
                }:
                    continue
            output.append(deepcopy(edge))
        return output[:limit]

    def execute_query(self, query, parameters, **options):
        self.query_calls.append((query, parameters, options))
        return {"success": True, "records": [{"value": 1}]}

    def shortest_path(self, source_id, target_id, max_depth, **options):
        return {
            "length": 1,
            "nodes": [{"id": source_id}, {"id": target_id}],
            "relationships": [],
        }

    def get_schema(self):
        return {"vertexlabels": []}

    def get_nodes(self, *args, **kwargs):
        return []


class ParallelEdgeStore(Store):
    def __init__(self):
        super().__init__()
        self.edges = [
            {
                "id": "parallel:1",
                "type": "SUPPORTED_BY",
                "start_node_id": "claim:1",
                "end_node_id": "actor:2",
                "properties": {"quote": "first source"},
            },
            {
                "id": "parallel:2",
                "type": "SUPPORTED_BY",
                "start_node_id": "claim:1",
                "end_node_id": "actor:2",
                "properties": {"quote": "second source"},
            },
        ]


def test_natural_language_query_uses_semantic_search_not_gremlin():
    store = Store()
    adapter = HugeGraphContextAdapter(
        store,
        node_labels=["Claim"],
        max_results=7,
        candidate_limit=80,
    )

    results = adapter.query("What evidence supports the claim?")

    assert store.search_calls == [
        (
            "What evidence supports the claim?",
            7,
            ["Claim"],
            {"text_properties": ["content", "name"], "candidate_limit": 80},
        )
    ]
    assert store.query_calls == []
    assert results[0]["id"] == "claim:1"
    assert results[0]["score"] == 0.91
    assert results[0]["content"] == "A sourced claim"
    assert results[0]["metadata"]["node_id"] == "claim:1"
    assert results[0]["metadata"]["sources"] == ["doc:1"]
    assert results[0]["metadata"]["backend"] == "hugegraph"
    assert results[0]["metadata"]["extra"] == "preserved"
    assert results[0]["metadata"]["evidence_values"] == [
        "A quote is evidence, not a source id"
    ]
    assert results[0]["metadata"]["time_constraints"] == ["2026-01-02"]


def test_neighbors_support_mixed_relationship_types_in_one_path():
    store = Store()
    adapter = HugeGraphContextAdapter(
        store,
        relationship_types=["SUPPORTED_BY", "INVOLVES"],
    )

    results = adapter.get_neighbors("claim:1", hops=2, direction="out")

    related = next(result for result in results if result["id"] == "actor:2")
    assert related["hop"] == 2
    assert related["path"] == ["claim:1", "event:1", "actor:2"]
    assert [item["type"] for item in related["relationships"]] == [
        "SUPPORTED_BY",
        "INVOLVES",
    ]
    assert [item["direction"] for item in related["relationships"]] == [
        "out",
        "out",
    ]
    assert related["relationships"][0]["properties"]["quote"] == "source quote"
    assert related["metadata"]["sources"] == ["https://source.invalid"]
    assert {call[0] for call in store.relationship_calls} == {
        "claim:1",
        "event:1",
    }


def test_neighbors_keep_parallel_same_type_paths_to_the_same_entity():
    store = ParallelEdgeStore()
    adapter = HugeGraphContextAdapter(
        store,
        relationship_types=["SUPPORTED_BY"],
    )

    results = adapter.get_neighbors("claim:1", hops=1, direction="out")

    target_paths = [result for result in results if result["id"] == "actor:2"]
    assert len(target_paths) == 2
    assert {
        result["relationships"][0]["id"] for result in target_paths
    } == {"parallel:1", "parallel:2"}
    assert {
        result["relationships"][0]["properties"]["quote"]
        for result in target_paths
    } == {"first source", "second source"}


def test_explicit_gremlin_and_path_are_separate_operations():
    store = Store()
    adapter = HugeGraphContextAdapter(store)

    result = adapter.execute_query("g.V().count()", {"unused": 1})
    path = adapter.get_path("a", "b", max_hops=3)

    assert result["records"] == [{"value": 1}]
    assert store.query_calls == [("g.V().count()", {"unused": 1}, {})]
    assert path["length"] == 1

