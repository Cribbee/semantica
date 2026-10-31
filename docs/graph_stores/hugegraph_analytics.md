# HugeGraph GraphRAG, Traverser, and Server OLAP

This document supplements [the HugeGraph backend guide](hugegraph.md). It covers
Semantica retrieval/reasoning integration and HugeGraph Server's native
traverser and single-machine OLAP surfaces. HugeGraph Computer is not required.

## Architecture

```text
Semantica GraphStore factory
        |
        v
HugeGraphStore
  |-- schema / CRUD / batch / Gremlin
  |-- native traversers (pyhugegraph)
  |-- ContextRetriever adapter
  `-- Server Job/Task REST API
          |
          `-- AlgorithmPool single-machine OLAP
```

The backend does not translate Cypher. It uses `pyhugegraph`, Gremlin, and
HugeGraph's native REST APIs directly. HugeGraph-specific OLAP remains an
explicit extension instead of forcing every Semantica backend to implement
algorithms that do not have portable semantics.

## GraphRAG and reasoning

Use `HugeGraphContextAdapter` as the knowledge graph supplied to the existing
`ContextRetriever`. The adapter performs semantic entry-node search and bounded
local expansion without copying the entire graph into memory.

```python
from semantica.context.context_retriever import ContextRetriever
from semantica.context.hugegraph_context import HugeGraphContextAdapter
from semantica.graph_store import GraphStore

store = GraphStore(
    backend="hugegraph",
    uri="https://hugegraph.example",
    graph="knowledge",
    graphspace="DEFAULT",
    user="admin",
    password="secret",
)

adapter = HugeGraphContextAdapter(
    store._store_backend,
    node_labels=["Entity", "Claim", "Evidence"],
    relationship_types=["RELATED_TO", "SUPPORTED_BY"],
    max_hops=2,
)
retriever = ContextRetriever(
    knowledge_graph=adapter,
    vector_store=vector_store,
    hybrid_alpha=0.6,
)
contexts = retriever.hybrid_search("What evidence supports claim C1?")
```

Retrieved nodes retain business IDs, labels, source and time properties,
evidence attributes, path vertices, edge direction, and edge properties. To use
the existing `GraphReasoner`, convert the retrieved contexts without rebuilding
the business graph:

```python
from semantica.context.hugegraph_reasoning import (
    reason_over_hugegraph_contexts,
)
from semantica.reasoning.graph_reasoner import GraphReasoner

answer = reason_over_hugegraph_contexts(
    GraphReasoner(provider="openai"),
    contexts,
    "What evidence supports claim C1?",
)
```

## Online traversers

`HugeGraphStore.execute_native_algorithm()` exposes the traversers already
provided by HugeGraph, including:

- `shortest_path`, `all_shortest_paths`, `weighted_shortest_path`
- `single_source_shortest_path`, `paths`
- `k_out`, `k_neighbor`
- `same_neighbors`, `jaccard_similarity`
- `personal_rank`

`store.analytics.shortest_path(...)` and
`store.analytics.traverser(name, ...)` provide a small analytics facade while
preserving each traverser's native parameters and result semantics.

## Server single-machine OLAP

The backend calls the current HugeGraph Server routes directly:

```text
POST   /graphspaces/{graphspace}/graphs/{graph}/jobs/algorithm/{name}
GET    /graphspaces/{graphspace}/graphs/{graph}/tasks/{id}?with_result=true
PUT    /graphspaces/{graphspace}/graphs/{graph}/tasks/{id}?action=cancel
DELETE /graphspaces/{graphspace}/graphs/{graph}/tasks/{id}?force=false
```

Example:

```python
backend = store._store_backend
result = backend.analytics.run(
    "degree_centrality",
    {"direction": "BOTH", "top": 20},
    timeout=300,
)
print(result["result"])
```

Supported capability groups:

| Category | Algorithms |
| --- | --- |
| Statistics | `count_vertex`, `count_edge`, `subgraph_stat` |
| Ranking | `page_rank` |
| Community | `lpa`, `louvain`, `weak_connected_component` |
| Centrality | degree, stress, betweenness, closeness, eigenvector |
| Structure | `k_core`, `triangle_count`, `cluster_coefficient` |

Aliases such as `pagerank`, `degree`, `wcc`, `label_propagation`, and
`clustering_coefficient` are normalized to Server algorithm names. Algorithm
parameters are passed through unchanged so Server semantics are not hidden.

### Mutation isolation

PageRank and the community/component algorithms can create schema or write
result properties. They are blocked in read-only mode and require explicit
confirmation:

```python
result = backend.analytics.louvain(
    direction="BOTH",
    times=20,
    allow_mutation=True,
)
```

Run mutating jobs only on an isolated test or analysis graph. This prevents an
OLAP validation run from changing an existing business graph.

## Compatibility and transport

- `graph_api_root` can override the default graphspace route for older or
  proxied deployments. It may use `{uri}`, `{graphspace}`, and `{graph}`.
- Basic credentials are rejected over non-loopback plain HTTP by default. Use
  HTTPS. `allow_insecure_auth=True` is an explicit development-only override.
- A caller-injected `pyhugegraph` client remains caller-owned and is not closed
  by `HugeGraphStore`.
- `merge=False` property replacement uses write-then-eliminate ordering and
  best-effort rollback, preserving stable vertex and edge IDs.

## Validation scope

The contract tests cover:

- current Algorithm Job and Task route shapes;
- alias and result normalization;
- task polling, cancellation, deletion, and server-error propagation;
- mutation gating and read-only behavior;
- recoverable node/edge property replacement;
- real `ContextRetriever` graph-only and vector/graph hybrid paths;
- evidence-preserving conversion into the existing `GraphReasoner`.

A release candidate should additionally run the integration suite against the
exact HugeGraph Docker image and Server version intended for deployment. This
is a compatibility gate, not a reason to substitute mocked tests for the
Semantica contract suite.
