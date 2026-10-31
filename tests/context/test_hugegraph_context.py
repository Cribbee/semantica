from __future__ import annotations

from semantica.context.hugegraph_context import HugeGraphContextAdapter


class Store:
    search_properties = ("content", "name")

    def __init__(self):
        self.search_calls = []
        self.neighbor_calls = []
        self.query_calls = []

    def search_nodes(self, query, limit, labels, **options):
        self.search_calls.append((query, limit, labels, options))
        return [
            {
                "id": "claim:1",
                "type": "Claim",
                "content": "A sourced claim",
                "score": 0.91,
                "metadata": {"source_id": "doc:1"},
                "node": {
                    "id": "claim:1",
                    "type": "Claim",
                    "properties": {
                        "statement": "A sourced claim",
                        "source_id": "doc:1",
                        "published_at": "2026-01-02",
                    },
                },
            }
        ]

    def get_neighbors(self, node_id, **options):
        self.neighbor_calls.append((node_id, options))
        return [
            {
                "id": "actor:2",
                "labels": ["Actor"],
                "properties": {
                    "name": "Agency",
                    "source_url": "https://source.invalid",
                },
                "distance": 2,
                "path": [node_id, "event:1", "actor:2"],
                "path_relationships": [
                    {
                        "id": "edge:1",
                        "type": "SUPPORTED_BY",
                        "start_node_id": node_id,
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
                ],
            }
        ]

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

    def get_relationships(self, *args, **kwargs):
        return []


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
    assert results[0]["metadata"]["sources"] == ["doc:1"]
    assert results[0]["metadata"]["time_constraints"] == ["2026-01-02"]


def test_neighbors_preserve_path_direction_and_edge_evidence():
    store = Store()
    adapter = HugeGraphContextAdapter(
        store,
        relationship_types=["SUPPORTED_BY", "INVOLVES"],
    )

    results = adapter.get_neighbors("claim:1", hops=2)

    # One call per configured relationship type; duplicate entity is removed.
    assert len(store.neighbor_calls) == 2
    assert len(results) == 1
    related = results[0]
    assert related["id"] == "actor:2"
    assert related["hop"] == 2
    assert related["path"] == ["claim:1", "event:1", "actor:2"]
    assert [item["direction"] for item in related["relationships"]] == ["out", "out"]
    assert related["relationships"][0]["properties"]["quote"] == "source quote"
    assert related["metadata"]["sources"] == ["https://source.invalid"]


def test_explicit_gremlin_and_path_are_separate_operations():
    store = Store()
    adapter = HugeGraphContextAdapter(store)

    result = adapter.execute_query("g.V().count()", {"unused": 1})
    path = adapter.get_path("a", "b", max_hops=3)

    assert result["records"] == [{"value": 1}]
    assert store.query_calls == [("g.V().count()", {"unused": 1}, {})]
    assert path["length"] == 1
