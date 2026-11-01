from __future__ import annotations

from dataclasses import dataclass

import pytest

from semantica.graph_store.graph_store import GraphStore
from semantica.graph_store.hugegraph_store import HugeGraphStore
from semantica.utils.exceptions import ProcessingError, ValidationError


@dataclass
class Vertex:
    id: object
    label: str | None = None
    properties: dict | None = None
    type: str | None = "vertex"


@dataclass
class Edge:
    id: object
    label: str | None = None
    outV: object | None = None
    inV: object | None = None
    properties: dict | None = None
    type: str | None = "edge"
    outVLabel: str | None = None
    inVLabel: str | None = None


class Builder:
    def __init__(self, kind: str, name: str, calls: list[tuple]):
        self.kind = kind
        self.name = name
        self.calls = calls

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((self.kind, self.name, name, args))
            return self

        return call


class SchemaManager:
    def __init__(self):
        self.calls: list[tuple] = []
        self.current = {
            "propertykeys": [],
            "vertexlabels": [],
            "edgelabels": [],
            "indexlabels": [],
        }

    def getSchema(self):
        return self.current

    def propertyKey(self, name):
        return Builder("property", name, self.calls)

    def vertexLabel(self, name):
        return Builder("vertex", name, self.calls)

    def edgeLabel(self, name):
        return Builder("edge", name, self.calls)

    def indexLabel(self, name):
        return Builder("index", name, self.calls)


class GraphManager:
    def __init__(self):
        self.vertices: dict[object, Vertex] = {}
        self.edges: dict[object, Edge] = {}
        self.next_vertex = 1
        self.next_edge = 100

    def addVertex(self, label, properties, vertex_id=None):
        vertex_id = self.next_vertex if vertex_id is None else vertex_id
        if isinstance(vertex_id, int) and vertex_id >= self.next_vertex:
            self.next_vertex = vertex_id + 1
        else:
            self.next_vertex += 1
        value = Vertex(vertex_id, label, dict(properties))
        self.vertices[vertex_id] = value
        return value

    def addVertices(self, rows):
        return [self.addVertex(label, props) for label, props in rows]

    def getVertexById(self, vertex_id):
        return self.vertices.get(vertex_id)

    def getVertexByCondition(self, label="", limit=0, page=None, properties=None):
        del page
        properties = properties or {}
        values = [
            vertex
            for vertex in self.vertices.values()
            if (not label or vertex.label == label)
            and all(
                (vertex.properties or {}).get(key) == value
                for key, value in properties.items()
            )
        ]
        return values[:limit] if limit else values

    def appendVertex(self, vertex_id, properties):
        vertex = self.vertices[vertex_id]
        vertex.properties = {**(vertex.properties or {}), **properties}
        return vertex

    def eliminateVertex(self, vertex_id, properties):
        vertex = self.vertices[vertex_id]
        current = dict(vertex.properties or {})
        for key, value in properties.items():
            if current.get(key) == value:
                current.pop(key, None)
        vertex.properties = current
        return vertex

    def removeVertexById(self, vertex_id):
        self.vertices.pop(vertex_id, None)
        for edge_id, edge in list(self.edges.items()):
            if edge.outV == vertex_id or edge.inV == vertex_id:
                self.edges.pop(edge_id)
        return {}

    def addEdge(self, label, out_id, in_id, properties):
        edge_id = f"S1:{out_id}>S1>{label}>{self.next_edge}>{in_id}"
        self.next_edge += 1
        value = Edge(edge_id, label, out_id, in_id, dict(properties))
        self.edges[edge_id] = value
        return value

    def addEdges(self, rows):
        return [
            self.addEdge(label, out_id, in_id, properties)
            for label, out_id, in_id, _out_label, _in_label, properties in rows
        ]

    def getEdgeById(self, edge_id):
        return self.edges.get(edge_id)

    def appendEdge(self, edge_id, properties):
        edge = self.edges[edge_id]
        edge.properties = {**(edge.properties or {}), **properties}
        return edge

    def eliminateEdge(self, edge_id, properties):
        edge = self.edges[edge_id]
        current = dict(edge.properties or {})
        for key, value in properties.items():
            if current.get(key) == value:
                current.pop(key, None)
        edge.properties = current
        return edge

    def getEdgeByPage(
        self,
        label=None,
        vertex_id=None,
        direction=None,
        limit=0,
        page=None,
        properties=None,
    ):
        del page
        properties = properties or {}
        values = []
        for edge in self.edges.values():
            if label and edge.label != label:
                continue
            if vertex_id is not None:
                if direction == "OUT" and edge.outV != vertex_id:
                    continue
                if direction == "IN" and edge.inV != vertex_id:
                    continue
            if not all(
                (edge.properties or {}).get(key) == value
                for key, value in properties.items()
            ):
                continue
            values.append(edge)
        return (values[:limit] if limit else values, None)

    def removeEdgeById(self, edge_id):
        self.edges.pop(edge_id, None)
        return {}


class TraverserManager:
    def __init__(self, graph: GraphManager):
        self.graph = graph
        self.calls = []

    def shortest_path(self, source_id, target_id, max_depth):
        self.calls.append(("shortest_path", source_id, target_id, max_depth))
        for edge in self.graph.edges.values():
            if {edge.outV, edge.inV} == {source_id, target_id}:
                return {
                    "vertices": [
                        self.graph.vertices[source_id],
                        self.graph.vertices[target_id],
                    ],
                    "edges": [edge],
                }
        return None

    def k_out(self, source_id, max_depth):
        self.calls.append(("k_out", source_id, max_depth))
        return {"vertices": [source_id], "max_depth": max_depth}


class VersionManager:
    def version(self):
        return {"versions": {"core": "1.7.0"}}


class Client:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.schema_manager = SchemaManager()
        self.traverser_manager = TraverserManager(self.graph_manager)

    def graph(self):
        return self.graph_manager

    def schema(self):
        return self.schema_manager

    def traverser(self):
        return self.traverser_manager

    def version(self):
        return VersionManager()


def store(**kwargs):
    client = kwargs.pop("client", Client())
    return HugeGraphStore(client=client, **kwargs), client


def test_read_only_rejects_all_mutating_entry_points():
    subject, _ = store(read_only=True)
    with pytest.raises(ValidationError):
        subject.create_node(["Actor"], {"name": "Alice"})
    with pytest.raises(ValidationError):
        subject.create_relationship(1, 2, "KNOWS")
    with pytest.raises(ValidationError):
        subject.create_relationships([])
    with pytest.raises(ValidationError):
        subject.update_relationship("edge:1", {"source": "s2"})
    with pytest.raises(ValidationError):
        subject.apply_schema({})
    with pytest.raises(ValidationError):
        subject.create_index("Actor", "name")


def test_node_crud_batch_and_single_label_contract():
    subject, _ = store()
    alice = subject.create_node(["Actor"], {"name": "Alice"}, vertex_id="actor:alice")
    assert alice == {
        "id": "actor:alice",
        "labels": ["Actor"],
        "properties": {"name": "Alice"},
    }

    created = subject.create_nodes(
        [
            {"labels": ["Actor"], "properties": {"name": "Bob"}},
            {"labels": ["Event"], "properties": {"name": "Launch"}},
        ]
    )
    assert [node["labels"] for node in created] == [["Actor"], ["Event"]]
    assert subject.get_nodes(labels=["Actor"], properties={"name": "Bob"})[0]["id"] == 2

    explicit = subject.create_nodes(
        [
            {
                "id": "actor:carol",
                "labels": ["Actor"],
                "properties": {"name": "Carol"},
            }
        ]
    )
    assert explicit[0]["id"] == "actor:carol"

    merged = subject.update_node("actor:alice", {"title": "Analyst"})
    assert merged["properties"] == {"name": "Alice", "title": "Analyst"}
    replaced = subject.update_node("actor:alice", {"name": "Alicia"}, merge=False)
    assert replaced["properties"] == {"name": "Alicia"}

    with pytest.raises(ValidationError, match="one physical label"):
        subject.create_node(["Actor", "Person"], {})
    with pytest.raises(ValidationError, match="at most one label"):
        subject.get_nodes(labels=["Actor", "Person"])

    assert subject.delete_node("actor:alice") is True
    assert subject.get_node("actor:alice") is None


def test_relationships_keep_direction_multiple_edges_and_both_deduplication():
    subject, _ = store()
    a = subject.create_node(["Actor"], {"name": "A"})
    b = subject.create_node(["Actor"], {"name": "B"})
    first = subject.create_relationship(a["id"], b["id"], "KNOWS", {"source": "s1"})
    second = subject.create_relationship(a["id"], b["id"], "KNOWS", {"source": "s2"})

    outgoing = subject.get_relationships(node_id=a["id"], direction="out")
    incoming = subject.get_relationships(node_id=b["id"], direction="in")
    both = subject.get_relationships(node_id=a["id"], direction="both")
    assert {edge["id"] for edge in outgoing} == {first["id"], second["id"]}
    assert {edge["id"] for edge in incoming} == {first["id"], second["id"]}
    assert len(both) == 2
    assert all(edge["start_node_id"] == a["id"] for edge in both)

    assert subject.delete_relationship(first["id"])
    assert len(subject.get_relationships(node_id=a["id"])) == 1


def test_relationship_batch_and_update_preserve_endpoints_and_properties():
    subject, _ = store()
    a = subject.create_node(["Actor"], {"name": "A"}, vertex_id="actor:a")
    b = subject.create_node(["Actor"], {"name": "B"}, vertex_id="actor:b")
    c = subject.create_node(["Actor"], {"name": "C"}, vertex_id="actor:c")

    created = subject.create_relationships(
        [
            {
                "start_id": a["id"],
                "end_id": b["id"],
                "type": "KNOWS",
                "properties": {"source": "s1"},
            },
            {
                "start_id": b["id"],
                "end_id": c["id"],
                "type": "KNOWS",
                "properties": {"source": "s2"},
            },
        ]
    )
    assert [(edge["start_node_id"], edge["end_node_id"]) for edge in created] == [
        ("actor:a", "actor:b"),
        ("actor:b", "actor:c"),
    ]

    merged = subject.update_relationship(created[0]["id"], {"quote": "A knows B"})
    assert merged["properties"] == {"source": "s1", "quote": "A knows B"}
    replaced = subject.update_relationship(
        created[0]["id"], {"source": "replacement"}, merge=False
    )
    assert replaced["properties"] == {"source": "replacement"}
    assert subject.get_relationship(created[0]["id"])["start_node_id"] == "actor:a"


def test_bounded_neighbors_and_filtered_shortest_path_preserve_path():
    subject, _ = store()
    a = subject.create_node(["Actor"], {"name": "A"})
    b = subject.create_node(["Actor"], {"name": "B"})
    c = subject.create_node(["Actor"], {"name": "C"})
    subject.create_relationship(a["id"], b["id"], "KNOWS")
    subject.create_relationship(b["id"], c["id"], "KNOWS")

    neighbors = subject.get_neighbors(
        a["id"],
        rel_type="KNOWS",
        direction="out",
        depth=2,
    )
    assert [(node["id"], node["distance"]) for node in neighbors] == [
        (b["id"], 1),
        (c["id"], 2),
    ]
    assert neighbors[-1]["path"] == [a["id"], b["id"], c["id"]]
    assert [edge["type"] for edge in neighbors[-1]["path_relationships"]] == [
        "KNOWS",
        "KNOWS",
    ]

    path = subject.shortest_path(
        a["id"], c["id"], rel_type="KNOWS", direction="out", max_depth=3
    )
    assert path["length"] == 2
    assert [node["id"] for node in path["nodes"]] == [a["id"], b["id"], c["id"]]
    assert len(path["relationships"]) == 2


def test_native_shortest_path_and_algorithm_extension_are_normalized():
    subject, client = store()
    a = subject.create_node(["Actor"], {"name": "A"})
    b = subject.create_node(["Actor"], {"name": "B"})
    subject.create_relationship(a["id"], b["id"], "KNOWS")

    path = subject.shortest_path(a["id"], b["id"], max_depth=4)
    assert path["length"] == 1
    assert [node["id"] for node in path["nodes"]] == [a["id"], b["id"]]
    assert client.traverser_manager.calls[0] == ("shortest_path", a["id"], b["id"], 4)

    result = subject.execute_native_algorithm("k_out", a["id"], 2)
    assert result == {"vertices": [a["id"]], "max_depth": 2}
    with pytest.raises(ValidationError, match="Unsupported"):
        subject.execute_native_algorithm("invented")


def test_parameterized_gremlin_returns_common_query_envelope():
    calls = []

    def execute(query, parameters):
        calls.append((query, parameters))
        return {
            "status": {"code": 200},
            "result": {"data": [{"name": "Alice"}, {"name": "Bob", "score": 0.8}]},
        }

    subject, _ = store(gremlin_executor=execute)
    result = subject.execute_query(
        "g.V().has('name', name).valueMap()",
        {"name": "Alice"},
    )
    assert calls == [("g.V().has('name', name).valueMap()", {"name": "Alice"})]
    assert result["success"] is True
    assert result["keys"] == ["name", "score"]
    assert result["metadata"]["query_language"] == "gremlin-groovy"


def test_search_nodes_is_bounded_and_handles_chinese_without_cypher_translation():
    subject, _ = store(search_candidate_limit=10)
    subject.create_node(
        ["Claim"],
        {
            "name": "事故预警机制",
            "content": "中美双方提议建立人工智能事故预警通报机制",
        },
    )
    subject.create_node(
        ["Claim"],
        {"name": "无关记录", "content": "体育比赛结果"},
    )

    results = subject.search_nodes(
        "人工智能事故预警",
        labels=["Claim"],
        limit=5,
    )
    assert len(results) == 1
    assert results[0]["type"] == "Claim"
    assert results[0]["score"] > 0.5
    assert results[0]["metadata"]["name"] == "事故预警机制"


def test_apply_schema_rejects_incompatible_existing_definition():
    subject, client = store()
    client.schema_manager.current["propertykeys"] = [
        {"name": "name", "data_type": "INT", "cardinality": "SINGLE"}
    ]
    with pytest.raises(ValidationError, match="incompatible"):
        subject.apply_schema(
            {
                "propertykeys": [
                    {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"}
                ]
            }
        )


def test_apply_schema_and_create_index_use_explicit_hugegraph_builders():
    subject, client = store()
    counts = subject.apply_schema(
        {
            "propertykeys": [
                {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"}
            ],
            "vertexlabels": [
                {
                    "name": "Actor",
                    "id_strategy": "CUSTOMIZE_STRING",
                    "properties": ["name"],
                    "nullable_keys": ["name"],
                }
            ],
            "edgelabels": [
                {
                    "name": "KNOWS",
                    "source_label": "Actor",
                    "target_label": "Actor",
                    "properties": ["name"],
                    "frequency": "MULTIPLE",
                    "sort_keys": ["name"],
                }
            ],
            "indexlabels": [
                {
                    "name": "idx_actor_name",
                    "base_type": "VERTEX_LABEL",
                    "base_value": "Actor",
                    "index_type": "SEARCH",
                    "fields": ["name"],
                }
            ],
        }
    )
    assert counts == {
        "propertykeys": 1,
        "vertexlabels": 1,
        "edgelabels": 1,
        "indexlabels": 1,
    }
    calls = client.schema_manager.calls
    assert any(
        call[:2] == ("property", "name") and call[2] in {"as_text", "asText"}
        for call in calls
    )
    assert any(
        call[:2] == ("vertex", "Actor")
        and call[2] in {"use_customize_string_id", "useCustomizeStringId"}
        for call in calls
    )
    assert any(
        call[:2] == ("edge", "KNOWS") and call[2] in {"multi_times", "multiTimes"}
        for call in calls
    )
    assert ("index", "idx_actor_name", "search", ()) in calls


class CloseTrackingClient(Client):
    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


class FailingReplacementGraphManager(GraphManager):
    def __init__(self):
        super().__init__()
        self.fail_vertex_append = False
        self.fail_edge_append = False
        self.vertex_eliminate_calls = 0
        self.edge_eliminate_calls = 0

    def appendVertex(self, vertex_id, properties):
        if self.fail_vertex_append:
            raise RuntimeError("simulated vertex write failure")
        return super().appendVertex(vertex_id, properties)

    def eliminateVertex(self, vertex_id, properties):
        self.vertex_eliminate_calls += 1
        return super().eliminateVertex(vertex_id, properties)

    def appendEdge(self, edge_id, properties):
        if self.fail_edge_append:
            raise RuntimeError("simulated edge write failure")
        return super().appendEdge(edge_id, properties)

    def eliminateEdge(self, edge_id, properties):
        self.edge_eliminate_calls += 1
        return super().eliminateEdge(edge_id, properties)


class PathIdTraverser(TraverserManager):
    def shortest_path(self, source_id, target_id, max_depth):
        self.calls.append(("shortest_path", source_id, target_id, max_depth))
        return {"path": [source_id, target_id]}


def test_graph_store_factory_builds_hugegraph_backend():
    client = Client()
    facade = GraphStore(
        backend="hugegraph",
        client=client,
        read_only=True,
        uri="http://127.0.0.1:8080",
        graph="existing_graph",
    )

    assert isinstance(facade._store_backend, HugeGraphStore)
    assert facade._store_backend.read_only is True
    assert facade._store_backend.graph_name == "existing_graph"
    assert facade.connect() is True


def test_graph_store_factory_accepts_apache_hugegraph_alias():
    facade = GraphStore(backend="apache_hugegraph", client=Client())
    assert isinstance(facade._store_backend, HugeGraphStore)


def test_close_keeps_injected_client_open():
    client = CloseTrackingClient()
    HugeGraphStore(client=client).close()
    assert client.closed is False


def test_replace_write_failure_does_not_delete_existing_properties():
    client = Client()
    graph = FailingReplacementGraphManager()
    client.graph_manager = graph
    client.traverser_manager = TraverserManager(graph)
    subject = HugeGraphStore(client=client)
    vertex = subject.create_node(["Actor"], {"name": "Alice", "title": "Analyst"})
    other = subject.create_node(["Actor"], {"name": "Bob"})
    edge = subject.create_relationship(
        vertex["id"], other["id"], "KNOWS", {"source": "s1", "quote": "q1"}
    )

    graph.fail_vertex_append = True
    with pytest.raises(ProcessingError, match="simulated vertex write failure"):
        subject.update_node(vertex["id"], {"name": "Alicia"}, merge=False)
    assert graph.vertex_eliminate_calls == 0
    assert subject.get_node(vertex["id"])["properties"] == {
        "name": "Alice",
        "title": "Analyst",
    }

    graph.fail_edge_append = True
    with pytest.raises(ProcessingError, match="simulated edge write failure"):
        subject.update_relationship(edge["id"], {"source": "s2"}, merge=False)
    assert graph.edge_eliminate_calls == 0
    assert subject.get_relationship(edge["id"])["properties"] == {
        "source": "s1",
        "quote": "q1",
    }


def test_native_path_id_response_falls_back_to_relationship_evidence():
    client = Client()
    client.traverser_manager = PathIdTraverser(client.graph_manager)
    subject = HugeGraphStore(client=client)
    source = subject.create_node(["Actor"], {"name": "A"})
    target = subject.create_node(["Actor"], {"name": "B"})
    edge = subject.create_relationship(source["id"], target["id"], "KNOWS")

    path = subject.shortest_path(source["id"], target["id"], max_depth=2)

    assert [node["id"] for node in path["nodes"]] == [source["id"], target["id"]]
    assert [relationship["id"] for relationship in path["relationships"]] == [
        edge["id"]
    ]


def test_gremlin_rejects_unsupported_scheme_and_remote_plaintext_credentials():
    with pytest.raises(ValidationError, match="http:// or https://"):
        HugeGraphStore(uri="ftp://example.com", client=Client())

    with pytest.raises(ValidationError, match="plain HTTP"):
        HugeGraphStore(
            uri="http://example.com:8080",
            user="admin",
            password="secret",
            client=Client(),
        )


def test_gremlin_uses_client_when_no_bindings():
    class Gremlin:
        def exec(self, query):
            assert query == "g.V().count()"
            return {"data": [3]}

    class ClientWithGremlin(Client):
        def gremlin(self):
            return Gremlin()

    store = HugeGraphStore(client=ClientWithGremlin())
    assert store.execute_query("g.V().count()")["records"]

    calls = []
    store._http_gremlin = lambda query, params: calls.append((query, params)) or {}
    store.execute_query("g.V(x)", {"x": 1})
    assert calls == [("g.V(x)", {"x": 1})]

