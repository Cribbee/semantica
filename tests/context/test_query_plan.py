"""Native shared planning: one plan, query-specific retrieval, explicit limits."""

from dataclasses import FrozenInstanceError
from concurrent.futures import CancelledError
from unittest.mock import Mock

import pytest

from semantica.context import ContextRetriever, QueryPlan, QueryPlanItem
from semantica.context.context_retriever import RetrievedContext
from semantica.context.drift_search import DriftSearchEngine
from semantica.context.drift_search import DriftSearchResult
from semantica.context.tiers import TrustTier


def plan_fixture():
    return QueryPlan(
        "Compare accounts",
        (
            QueryPlanItem("first", "What did A report?", ("A",)),
            QueryPlanItem("second", "When did A act?", ("A",)),
        ),
    )


def test_plan_roundtrip_is_immutable_and_selection_is_strict():
    entities = ["A"]
    item = QueryPlanItem("first", "Question", entities)
    plan = QueryPlan("Root question", [item])
    entities.append("B")
    assert item.target_entities == ("A",)
    wire = plan.to_dict()
    assert QueryPlan.from_dict(wire) == plan
    assert QueryPlan.from_dict(wire).content_hash == plan.content_hash
    wire["items"][0]["target_entities"].append("C")
    assert item.target_entities == ("A",)
    with pytest.raises(FrozenInstanceError):
        item.sub_query = "Changed"
    for selection in ([], ["first", "first"], ["missing"]):
        with pytest.raises(ValueError):
            plan.select(selection)
    with pytest.raises(ValueError):
        QueryPlan("Root", (item, item))
    with pytest.raises(ValueError):
        QueryPlanItem("empty", " ")
    with pytest.raises(ValueError):
        QueryPlan.from_dict(dict(plan.to_dict(), query_truncated="false"))


def test_plan_only_reuses_native_generation_without_traversal_or_synthesis():
    llm = Mock(spec=["generate_structured"])
    llm.generate_structured.return_value = {
        "facets": [
            {"sub_query": "Check original record", "target_entities": ["A"]},
        ]
    }
    engine = DriftSearchEngine(llm=llm)
    engine._traverse_and_prune = Mock(side_effect=AssertionError("Traversal"))
    engine._synthesize_hybrid = Mock(side_effect=AssertionError("Synthesis"))
    plan = engine.plan("Compare accounts")
    assert plan.generation_method == "model"
    assert plan.items[0].sub_query == "Check original record"
    llm.generate_structured.assert_called_once()


def test_fallback_and_input_output_truncation_are_disclosed():
    fallback = DriftSearchEngine().plan("Question")
    assert fallback.generation_method == "extractive_fallback"
    llm = Mock(spec=["generate_structured"])
    llm.generate_structured.return_value = {
        "facets": [
            {"sub_query": "First"},
            {"sub_query": "Second"},
        ]
    }
    engine = DriftSearchEngine(llm=llm, max_facets=1, max_context_tokens=100)
    plan = engine.plan("Question " * 1000)
    assert plan.query == "Question " * 1000
    assert plan.query_truncated is True
    assert plan.items_truncated is True
    assert len(plan.items) == 1


def test_same_seed_different_queries_have_distinct_alignment_and_ownership():
    plan = plan_fixture()
    engine = DriftSearchEngine()
    engine._generate_facets = Mock(side_effect=AssertionError("Replanning"))
    engine._extract_thematic_framing = Mock(side_effect=AssertionError("Reframing"))
    engine._synthesize_hybrid = Mock(side_effect=AssertionError("Synthesis"))
    queries = []

    def traverse(facets, query, framing, embedding, **options):
        queries.append(query)
        assert len(facets) == 1
        assert facets[0].target_entities == ["A"]
        options["receipt"].update(budget_reached=False)
        if "When" in query:
            return [], 0, 0
        return [{"source": "A", "target": "B", "relation": "REPORTED"}], 1, 0

    engine._traverse_and_prune = traverse
    result = engine.search(plan.query, plan=plan, synthesize=False)
    assert queries == [item.sub_query for item in plan.items]
    assert result.answer == ""
    assert len(result.to_retrieved_contexts()) == 1
    assert result.to_retrieved_contexts()[0].metadata["plan_item_id"] == "first"
    assert [item["empty"] for item in result.metrics["items"]] == [False, True]
    assert result.metrics["plan_hash"] == plan.content_hash


def test_plan_mismatch_and_unknown_id_fail_before_embedding():
    embedder = Mock(side_effect=AssertionError("Should not embed"))
    engine = DriftSearchEngine(embedder=embedder)
    with pytest.raises(ValueError):
        engine.search("Wrong question", plan=plan_fixture())
    with pytest.raises(ValueError):
        engine.search(plan_fixture().query, plan=plan_fixture(), item_ids=["missing"])


def test_traversal_budget_and_empty_seed_receipts():
    graph = {
        "nodes": [{"id": "A"}, {"id": "B"}],
        "edges": [
            {
                "source": "A",
                "target": "B",
                "type": "REPORTED",
                "description": "long " * 500,
            },
        ],
    }
    engine = DriftSearchEngine(knowledge_graph=graph, max_context_tokens=100)
    engine._compute_alignment_score = Mock(return_value=1)
    result = engine.search(plan_fixture().query, plan=plan_fixture(), synthesize=False)
    assert result.metrics["items"][0]["budget_reached"] is True
    assert result.metrics["items"][0]["empty"] is True
    missing = QueryPlan("Root", (QueryPlanItem("missing", "Unknown", ("Absent",)),))
    result = engine.search("Root", plan=missing, synthesize=False)
    assert result.metrics["items"][0]["seed_count"] == 0
    assert result.metrics["items"][0]["budget_reached"] is False


def test_local_plan_facade_uses_subqueries_without_mutating_cached_context():
    retriever = ContextRetriever()
    cached = RetrievedContext(content="Record", score=1, source="test")
    retriever.retrieve = Mock(return_value=[cached])
    results = retriever.retrieve_plan(plan_fixture())
    assert [call.args[0] for call in retriever.retrieve.call_args_list] == [
        item.sub_query + "\nTarget entities: A" for item in plan_fixture().items
    ]
    assert [result["item_id"] for result in results] == ["first", "second"]
    assert results[0]["contexts"][0].metadata["plan_item_id"] == "first"
    assert results[1]["contexts"][0].metadata["plan_item_id"] == "second"
    assert cached.metadata == {}
    assert results[0]["metrics"]["budget_reached"] is None


def test_control_flow_baseexception_is_not_swallowed_by_planning():
    class Cancelled(BaseException):
        pass

    llm = Mock(spec=["generate_structured"])
    llm.generate_structured.side_effect = Cancelled()
    with pytest.raises(Cancelled):
        DriftSearchEngine(llm=llm).plan("Question")


@pytest.mark.parametrize(
    "error",
    [CancelledError(), RuntimeError("budget_closed"), TypeError("provider_failed")],
)
def test_plan_provider_errors_are_terminal_and_never_redispatched(error):
    llm = Mock(spec=["generate_typed", "generate"])
    llm.generate_typed.side_effect = error
    with pytest.raises(type(error)):
        DriftSearchEngine(llm=llm).plan("Question")
    llm.generate_typed.assert_called_once()
    llm.generate.assert_not_called()


def test_typed_provider_without_schema_argument_is_dispatched_once():
    class Provider:
        calls = 0

        def generate_typed(self, prompt):
            self.calls += 1
            return {"facets": [{"sub_query": "Check record"}]}

    provider = Provider()
    assert DriftSearchEngine(llm=provider).plan("Root").generation_method == "model"
    assert provider.calls == 1


def test_plan_rejects_empty_facet_instead_of_silently_omitting_it():
    llm = Mock(spec=["generate_structured"])
    llm.generate_structured.return_value = {
        "facets": [
            {"sub_query": "Valid"},
            {"sub_query": " "},
        ]
    }
    with pytest.raises(ValueError, match="empty_sub_query"):
        DriftSearchEngine(llm=llm).plan("Root")


def test_integer_float_wire_roundtrip_keeps_native_plan_hash():
    # JavaScript JSON.stringify writes 1.0 as 1. The native hash is opaque to
    # callers; consuming either wire spelling must restore the same identity.
    plan = plan_fixture()
    wire = plan.to_dict()
    wire["items"][0]["relevance_score"] = 1
    wire["items"][1]["relevance_score"] = 1
    assert QueryPlan.from_dict(wire).content_hash == plan.content_hash
    negative_zero = QueryPlan(
        "Root", (QueryPlanItem("a", "Query", relevance_score=-0.0),)
    )
    js_wire = negative_zero.to_dict()
    js_wire["items"][0]["relevance_score"] = 0
    assert QueryPlan.from_dict(js_wire).content_hash == negative_zero.content_hash


def test_seed_hints_change_actual_local_query_but_not_original_subquery():
    plan = QueryPlan(
        "Root",
        (
            QueryPlanItem("a", "When was it done?", ("Organization-A",)),
            QueryPlanItem("b", "When was it done?", ("opaque/id:B",)),
            QueryPlanItem("c", "When was it done?"),
        ),
    )
    retriever = ContextRetriever()
    retriever.retrieve = Mock(return_value=[])
    result = retriever.retrieve_plan(plan)
    assert len({item["actual_query"] for item in result}) == 3
    assert result[2]["actual_query"] == "When was it done?"
    assert all(item["query"] == "When was it done?" for item in result)
    assert all(item["seed_mode"] == "query_hints" for item in result)


def test_drift_context_attribution_cannot_be_overridden_by_graph_attributes():
    result = DriftSearchResult(
        query="Root",
        answer="",
        verified_local_contexts=[
            {
                "source": "A",
                "target": "B",
                "relation": "X",
                "plan_item_id": "real",
                "attributes": {"plan_item_id": "forged"},
            }
        ],
    )
    assert result.to_retrieved_contexts()[0].metadata["plan_item_id"] == "real"
    result.verified_local_contexts[0].pop("plan_item_id")
    result.verified_local_contexts[0]["attributes"] = {}
    assert "plan_item_id" not in result.to_retrieved_contexts()[0].metadata


def test_drift_plan_applies_relevance_and_trust_filters_and_rejects_truth_filter():
    graph = {
        "nodes": [{"id": "A"}, {"id": "B"}],
        "edges": [
            {"source": "A", "target": "B", "relation": "REPORTED"},
        ],
    }
    retriever = ContextRetriever(knowledge_graph=graph)
    plan = QueryPlan("Root", (QueryPlanItem("a", "REPORTED", ("A",)),))
    unfiltered = retriever.retrieve_plan(plan, mode="drift", drift_threshold=0)
    assert len(unfiltered[0]["contexts"]) == 1
    trusted = retriever.retrieve_plan(
        plan, mode="drift", drift_threshold=0, min_trust_tier=TrustTier.SILVER
    )
    assert trusted[0]["empty"] is True
    assert trusted[0]["metrics"]["contexts_before_filters"] == 1
    assert trusted[0]["metrics"]["contexts_after_filters"] == 0
    relevant = retriever.retrieve_plan(
        plan, mode="drift", drift_threshold=0, min_relevance_score=1
    )
    assert relevant[0]["empty"] is True
    with pytest.raises(Exception, match="truth_filter"):
        retriever.retrieve_plan(plan, mode="drift", truth_filter=object())


@pytest.mark.parametrize("invalid", [None, 42, False])
def test_strict_plan_never_silently_drops_invalid_facet_entries(invalid):
    llm = Mock(spec=["generate_structured"])
    llm.generate_structured.return_value = {
        "facets": [
            {"sub_query": "Valid"},
            invalid,
        ]
    }
    with pytest.raises(ValueError):
        DriftSearchEngine(llm=llm).plan("Root")
    llm.generate_structured.assert_called_once()


@pytest.mark.parametrize("error", [CancelledError(), RuntimeError("budget_closed")])
def test_fact_embedding_failure_during_supplied_plan_does_not_fallback(error):
    graph = {
        "nodes": [{"id": "A"}, {"id": "B"}],
        "edges": [
            {"source": "A", "target": "B", "relation": "REPORTED"},
        ],
    }
    # First embedding is the item's query; the second is its graph fact.
    embedder = Mock(side_effect=[[1.0], error])
    engine = DriftSearchEngine(knowledge_graph=graph, embedder=embedder)
    with pytest.raises(type(error)):
        engine.search(plan_fixture().query, plan=plan_fixture(), synthesize=False)
    assert embedder.call_count == 2
    assert embedder.call_args_list[0].args[0] == plan_fixture().items[0].sub_query


def test_supplied_plan_synthesis_error_propagates_without_legacy_fallback():
    llm = Mock(spec=["generate"])
    llm.generate.side_effect = RuntimeError("budget_closed")
    engine = DriftSearchEngine(llm=llm)
    with pytest.raises(RuntimeError, match="budget_closed"):
        engine.search(plan_fixture().query, plan=plan_fixture())
    llm.generate.assert_called_once()


def test_fallback_entity_limit_is_disclosed_and_model_empty_list_is_rejected():
    graph = {
        "nodes": [{"id": "Alpha"}, {"id": "Beta"}, {"id": "Gamma"}],
        "edges": [
            {"source": "Alpha", "target": "Beta", "relation": "X"},
            {"source": "Beta", "target": "Gamma", "relation": "X"},
        ],
    }
    plan = DriftSearchEngine(knowledge_graph=graph, max_facets=1).plan(
        "Alpha Beta Gamma"
    )
    assert plan.generation_method == "extractive_fallback"
    assert plan.items_truncated is True
    llm = Mock(spec=["generate_structured"])
    llm.generate_structured.return_value = {"facets": []}
    with pytest.raises(ValueError):
        DriftSearchEngine(llm=llm).plan("Root")


def test_wire_required_fields_report_value_errors():
    for value in [{}, {"schema_version": "query-plan-v1", "items": []}]:
        with pytest.raises(ValueError):
            QueryPlan.from_dict(value)
    for value in [{}, {"item_id": "a"}, {"sub_query": "Question"}]:
        with pytest.raises(ValueError):
            QueryPlanItem.from_dict(value)


def test_real_lexical_alignment_separates_questions_sharing_the_same_graph_seed():
    graph = {
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "edges": [
            {
                "source": "A",
                "target": "B",
                "relation": "COST",
                "description": "cash invoice amount",
            },
            {
                "source": "A",
                "target": "C",
                "relation": "DATE",
                "description": "date timeline schedule",
            },
        ],
    }
    plan = QueryPlan(
        "Compare records",
        (
            QueryPlanItem("cost", "cash invoice amount", ("A",)),
            QueryPlanItem("date", "date timeline schedule", ("A",)),
        ),
    )
    retriever = ContextRetriever(knowledge_graph=graph)
    native_call = Mock(wraps=retriever.retrieve_drift)
    retriever.retrieve_drift = native_call
    receipts = retriever.retrieve_plan(plan, mode="drift", max_depth=1)
    native_call.assert_called_once()
    assert [
        [ctx.metadata["relation"] for ctx in row["contexts"]] for row in receipts
    ] == [["COST"], ["DATE"]]
