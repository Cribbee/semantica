# Shared native query plans

`ContextRetriever.plan_query(query, **options)` runs the existing DRIFT thematic framing and facet generation without traversing the graph or synthesizing an answer. `QueryPlan` and `QueryPlanItem` are immutable, JSON-serializable information needs, not claims or verified evidence.

```python
from semantica.context import ContextRetriever, QueryPlan

retriever = ContextRetriever(knowledge_graph=graph, llm=provider)
plan = retriever.plan_query(question, max_facets=5)
wire = plan.to_dict()
restored = QueryPlan.from_dict(wire)
assert restored.content_hash == plan.content_hash

local = retriever.retrieve_plan(restored, mode="local")
drift = retriever.retrieve_plan(restored, mode="drift", item_ids=[plan.items[0].item_id])
```

Plan generation reports `generation_method` (`model`, `extractive_fallback`, or caller `provided`) and query/framing/item truncation flags. Strict planning expects canonical facet objects with `sub_query`; empty lists and invalid list entries are rejected instead of silently dropped. The existing facet field normalizers (seed string/list conversion and score clamping) are retained; strict here describes list membership and provider error propagation, not a new field-normalization policy. Legacy `search` still accepts its historical shorthand/aliases. A provider exception or cancellation in this new planning path propagates without trying another provider method. No provider yields an explicitly marked deterministic fallback. An application requiring successful model planning must check these receipts.

`retrieve_plan` returns one receipt for every selected item, including empty results: plan hash, item ID, original sub-query, actual query, seed entities/mode, contexts, empty/truncation flags and metrics. IDs must exist and be unique.

- Local uses the existing public local ranking/expansion path, adding target entities as **query hints**. They are not a strict ID filter or a guarantee of resolved graph seeds. The original `query` and effective `actual_query` are separate. Native local retrieval has no budget-stop receipt; its `budget_reached` remains null.
- DRIFT consumes the supplied plan without generating facets or thematic framing again. Each item uses its own sub-query, embedding, and graph seed entities, including when two items share the same seed. `budget_reached`, resolved `seed_count`, context token count/budget and empty results are reported per item.
- Retrieval-only consumption skips answer synthesis. A graph fact may belong to multiple items; program-assigned `plan_item_id` cannot be overridden by edge attributes. Relevance and trust filters still apply. Truth-maintenance filtering remains local-only and is rejected for DRIFT.

For lower-level composition, `DriftSearchEngine.plan()` and `search(query, plan=plan, item_ids=..., synthesize=False)` are public. Calling `search` without a plan retains the existing pipeline. The legacy path keeps its existing provider fallback behavior; the new plan path is strict about provider failures.

Limits are **per item**. Callers own the shared execution budget, cancellation, immutable source/scope binding, model identity, review/version state and dependency invalidation. No per-item checkpoint or evidence-truth guarantee is implied. `max_results` truncation is distinct from graph budget exhaustion or semantic completeness.

`content_hash` is an opaque native SHA256 identity. Carry it across the wire; do not compute it using a different JSON serializer. Score values normalize to finite floats so JavaScript integer spelling does not change the identity on native rehydration. Product metadata should wrap the native plan, not add unknown fields to its versioned DTO.
