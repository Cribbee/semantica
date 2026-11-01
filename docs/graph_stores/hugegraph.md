# HugeGraph graph-store backend

`HugeGraphStore` adapts Apache HugeGraph to Semantica's property-graph store
contract without translating Cypher. It uses `pyhugegraph` for schema and graph
CRUD, the native Gremlin REST endpoint for parameterized queries, and HugeGraph
traverser APIs for supported algorithms.

Unparameterized Gremlin uses the Python client's Gremlin manager. Parameterized
queries use the REST endpoint because the client's `exec()` does not expose
bindings; values are never interpolated into the query string. Server algorithm
submission and task options absent from the client use a small REST fallback.

## Install

```bash
pip install "hugegraph-python>=1.7.1"
```

The dependency is optional. Importing Semantica without `hugegraph-python`
continues to work; constructing or connecting a HugeGraph backend requires it.

## Read an existing graph

Use `read_only=True` for retrieval and GraphRAG experiments over an existing
business graph:

```python
from semantica.graph_store.hugegraph_store import HugeGraphStore

store = HugeGraphStore(
    uri="http://localhost:8080",
    graph="hugegraph",
    graphspace="DEFAULT",  # omit when graphspaces are disabled
    user="admin",
    password="hugegraph",
    read_only=True,
)
store.connect()

actors = store.get_nodes(labels=["Actor"], properties={"name": "Alice"})
local_context = store.get_neighbors(actors[0]["id"], depth=2)
```

`read_only` rejects schema, index, node, and edge mutations. It does not rely on
server-side permissions, so production deployments should still use a
least-privilege HugeGraph account.

## Strong schema and writes

HugeGraph has a strong schema and one physical label per vertex. Write-enabled
code must apply an explicit schema and map any semantic multi-label model to one
physical HugeGraph vertex label:

```python
store = HugeGraphStore(
    uri="http://localhost:8080",
    graph="semantica_test",
    user="admin",
    password="hugegraph",
)
store.connect()
store.apply_schema(
    {
        "propertykeys": [
            {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
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
                "frequency": "MULTIPLE",
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
```

Before creating missing definitions, `apply_schema()` compares any same-named
existing definitions and rejects incompatible data types, ID strategies, label
links, multiplicity, or index fields. This prevents `ifNotExist()` from silently
accepting a different physical schema.

Batch vertex creation uses HugeGraph's native batch endpoint when all IDs are
automatic. If any vertex has an explicit `id`, the adapter preserves it by using
the single-vertex endpoint. Batch edges resolve or accept endpoint labels because
HugeGraph's batch edge API requires `outVLabel` and `inVLabel`.

## Query and traversal semantics

- `execute_query(query, parameters)` executes Gremlin-Groovy with bindings and
  returns Semantica's common `{success, records, keys, metadata}` envelope.
- `get_neighbors()` performs bounded 1-N hop expansion and returns each node with
  `distance`, `path`, and `path_relationships`; it never loads the full graph.
- `shortest_path()` prefers HugeGraph's native traverser, then falls back to a
  bounded direction/edge-label-aware BFS when filtering is required.
- `execute_native_algorithm()` exposes a small allowlist of HugeGraph traversers,
  including K-out, K-neighbor, shortest paths, and PersonalRank. It is an
  extension point, not a second algorithm framework.
- `search_nodes()` is deterministic bounded lexical entry-node retrieval for the
  ContextRetriever adapter. It is deliberately separate from raw Gremlin.

Nodes are normalized as `{id, labels, properties}` and relationships as
`{id, type, start_node_id, end_node_id, properties}`. HugeGraph edge IDs are kept
opaque, so multiple edges and sort-key-generated IDs are not rewritten.

## ContextRetriever and hybrid GraphRAG

`HugeGraphContextAdapter` gives `ContextRetriever` a semantic `query(question)`
operation while keeping raw Gremlin under `execute_query()`. It expands only the
selected local subgraph and preserves path, direction, source, time, quote, and
other relationship evidence fields.

```python
from semantica.context.context_retriever import ContextRetriever
from semantica.context.hugegraph_context import HugeGraphContextAdapter

knowledge_graph = HugeGraphContextAdapter(
    store,
    node_labels=["Claim", "Actor", "Event"],
    relationship_types=["SUPPORTED_BY", "INVOLVES", "MENTIONS"],
    max_hops=3,
)

pure_graph = ContextRetriever(
    knowledge_graph=knowledge_graph,
    hybrid_alpha=1.0,
    max_expansion_hops=3,
)
hybrid = ContextRetriever(
    knowledge_graph=knowledge_graph,
    vector_store=vector_store,
    hybrid_alpha=0.6,
    max_expansion_hops=3,
)
```

The adapter is a read-through view. It does not re-extract entities or create a
second business graph; a vector store or text index can be supplied separately
for Semantica's existing hybrid retrieval and ranking path.

## Test scope

The unit suite uses a behaviorally accurate fake `pyhugegraph` client and covers
strong-schema conflicts, custom IDs, node and relationship CRUD, native batches,
multiple directed edges, bounded multi-hop traversal, shortest paths,
parameterized Gremlin envelopes, native algorithm dispatch, read-only mode, and
Chinese lexical entry-node matching. A live HugeGraph container remains the
required integration gate for release certification.


## Unified GraphStore factory

Applications that only need Semantica's common graph-store API can construct the
backend through the public factory:

```python
from semantica.graph_store import GraphStore

store = GraphStore(
    backend="hugegraph",
    uri="https://hugegraph.example.com",
    graph="hugegraph",
    user="reader",
    password="secret",
    read_only=True,
)
```

Install the optional backend with `pip install "semantica[graph-hugegraph]"`.
Backend-specific entry-node search and native traverser extensions remain on
`HugeGraphStore`. Credentialed requests require HTTPS unless the target is a
loopback endpoint.

