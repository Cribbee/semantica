from __future__ import annotations

from semantica.context.context_retriever import ContextRetriever
from semantica.context.hugegraph_context import HugeGraphContextAdapter
from semantica.context.hugegraph_reasoning import (
    hugegraph_contexts_to_reasoning_graph,
    reason_over_hugegraph_contexts,
)
from semantica.reasoning import graph_reasoner as graph_reasoner_module


class BusinessGraphStore:
    search_properties = (
        "content",
        "name",
        "statement",
        "title",
    )

    def __init__(self):
        self.nodes = {
            "claim:1": {
                "id": "claim:1",
                "labels": ["Claim"],
                "properties": {
                    "statement": "Project Atlas is supported by source S1.",
                    "source_id": "S1",
                    "published_at": "2026-10-30T08:00:00Z",
                    "quote": "Atlas received verified support.",
                },
            },
            "actor:1": {
                "id": "actor:1",
                "labels": ["Actor"],
                "properties": {"name": "Project Atlas"},
            },
            "doc:1": {
                "id": "doc:1",
                "labels": ["Evidence"],
                "properties": {
                    "title": "Source S1",
                    "source_url": "https://example.test/s1",
                },
            },
        }
        self.edges = {
            "edge:1": {
                "id": "edge:1",
                "type": "INVOLVES",
                "start_node_id": "claim:1",
                "end_node_id": "actor:1",
                "properties": {"source_id": "S1"},
            },
            "edge:2": {
                "id": "edge:2",
                "type": "SUPPORTED_BY",
                "start_node_id": "claim:1",
                "end_node_id": "doc:1",
                "properties": {
                    "quote": "Atlas received verified support.",
                    "source_url": "https://example.test/s1",
                    "published_at": "2026-10-30T08:00:00Z",
                },
            },
        }

    def search_nodes(self, query, limit=20, labels=None, **options):
        del query, labels, options
        node = self.nodes["claim:1"]
        return [
            {
                "id": node["id"],
                "type": "Claim",
                "content": node["properties"]["statement"],
                "score": 0.95,
                "metadata": dict(node["properties"]),
                "node": node,
            }
        ][:limit]

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_relationships(
        self,
        node_id=None,
        rel_type=None,
        direction="both",
        limit=100,
        **options,
    ):
        del options
        values = []
        for edge in self.edges.values():
            if rel_type and edge["type"] != rel_type:
                continue
            if node_id is not None:
                outgoing = edge["start_node_id"] == node_id
                incoming = edge["end_node_id"] == node_id
                if direction == "out" and not outgoing:
                    continue
                if direction == "in" and not incoming:
                    continue
                if direction == "both" and not (outgoing or incoming):
                    continue
            values.append(edge)
        return values[:limit]

    def execute_query(self, query, parameters=None, **options):
        return {
            "success": True,
            "records": [],
            "keys": [],
            "metadata": {
                "query": query,
                "parameters": parameters or {},
                **options,
            },
        }

    def shortest_path(self, source_id, target_id, max_depth=5, **options):
        del max_depth, options
        return {
            "nodes": [self.nodes[source_id], self.nodes[target_id]],
            "relationships": [self.edges["edge:2"]],
        }

    def get_schema(self):
        return {}

    def get_nodes(self, *args, **kwargs):
        del args, kwargs
        return list(self.nodes.values())


class VectorStore:
    def search(self, query, limit):
        del query
        return [
            {
                "id": "chunk:1",
                "content": "Background text about Project Atlas.",
                "score": 0.8,
                "metadata": {"source_id": "S2"},
            }
        ][:limit]


class RecordingProvider:
    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, **options):
        del options
        self.prompt = prompt
        return "Project Atlas is supported by source S1."


def build_retriever():
    adapter = HugeGraphContextAdapter(
        BusinessGraphStore(),
        node_labels=["Claim", "Actor", "Evidence"],
        relationship_types=["INVOLVES", "SUPPORTED_BY"],
        max_hops=2,
    )
    return ContextRetriever(
        knowledge_graph=adapter,
        hybrid_alpha=1.0,
        max_expansion_hops=2,
    )


def test_real_context_retriever_path_preserves_graph_evidence():
    retriever = build_retriever()

    results = retriever.graph_search(
        "Who supports Project Atlas?",
        max_results=3,
    )

    assert results
    result = results[0]
    assert result.source == "graph:claim:1"
    assert result.metadata["source_id"] == "S1"
    assert result.metadata["published_at"] == "2026-10-30T08:00:00Z"
    related = {item["id"]: item for item in result.related_entities}
    assert {"actor:1", "doc:1"} <= set(related)
    evidence_path = related["doc:1"]["relationships"]
    assert evidence_path[-1]["type"] == "SUPPORTED_BY"
    assert evidence_path[-1]["properties"]["quote"].startswith("Atlas")


def test_existing_vector_and_graph_hybrid_retrieval_are_both_used():
    retriever = build_retriever()
    retriever.vector_store = VectorStore()
    retriever.hybrid_alpha = 0.5

    results = retriever.hybrid_search(
        "Project Atlas support",
        max_results=5,
    )

    sources = {result.source for result in results}
    assert "graph:claim:1" in sources
    assert "vector:chunk:1" in sources


def test_retrieved_subgraph_flows_into_graph_reasoner(monkeypatch):
    contexts = build_retriever().graph_search(
        "Who supports Project Atlas?",
        max_results=3,
    )
    graph = hugegraph_contexts_to_reasoning_graph(contexts)

    entity_ids = {entity["id"] for entity in graph["entities"]}
    relationship_types = {
        relationship["type"] for relationship in graph["relationships"]
    }
    assert {"claim:1", "actor:1", "doc:1"} <= entity_ids
    assert {"INVOLVES", "SUPPORTED_BY"} <= relationship_types

    provider = RecordingProvider()
    monkeypatch.setattr(
        graph_reasoner_module,
        "create_provider",
        lambda *args, **kwargs: provider,
    )
    reasoner = graph_reasoner_module.GraphReasoner(provider="test")

    answer = reason_over_hugegraph_contexts(
        reasoner,
        contexts,
        "Who supports Project Atlas?",
    )

    assert answer == "Project Atlas is supported by source S1."
    assert "SUPPORTED_BY" in provider.prompt
    assert "Atlas received verified support." in provider.prompt
    assert "https://example.test/s1" in provider.prompt
