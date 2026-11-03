"""Incremental identity preserves native cache eligibility, not stale evidence."""

import copy
import json

import networkx as nx
import pytest

from semantica.kg import (
    CommunityHierarchy,
    CommunityHierarchyBuilder,
    CommunitySummarizer,
    HierarchicalCommunity,
    build_community_hierarchy,
)


def graph(*pairs):
    result = nx.Graph()
    for left, right in pairs:
        result.add_edge(left, right, relation="recorded")
        result.nodes[left]["quote"] = left
        result.nodes[right]["quote"] = right
    return result


def view(g):
    """Source-backed dictionary projection used by GraphRAG callers."""
    return {
        "entities": [{"id": n, **dict(data)} for n, data in g.nodes(data=True)],
        "relationships": [
            {"source": u, "target": v, **dict(data)}
            for u, v, data in g.edges(data=True)
        ],
    }


def members(hierarchy):
    return {(c.level, tuple(c.entity_ids)): c for c in hierarchy.communities.values()}


class Model:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, **options):
        self.prompts.append(prompt)
        return json.dumps(
            {
                "title": "Recorded sources",
                "summary": "Source-backed summary",
                "findings": [],
                "impact_rating": 5.0,
            }
        )


def test_unrelated_insertion_reuses_legacy_disk_cache_and_exact_generation_inputs(
    tmp_path,
):
    builder = CommunityHierarchyBuilder(max_levels=1)
    before_graph = graph(("b1", "b2"), ("z1", "z2"))
    previous = builder.build(before_graph)
    frozen = previous.to_dict()
    model = Model()
    summarizer = CommunitySummarizer(llm=model, cache_dir=tmp_path)
    old_reports = summarizer.summarize_hierarchy(previous, graph=view(before_graph))
    assert len(model.prompts) == 2
    old_prompts = list(model.prompts)
    after_graph = graph(("a1", "a2"), ("b1", "b2"), ("z1", "z2"))
    baseline = builder.build(after_graph)
    updated = builder.build(after_graph, previous_hierarchy=previous)
    assert set(members(updated)) == set(members(baseline))
    assert previous.to_dict() == frozen
    for key, old in members(previous).items():
        current = members(updated)[key]
        assert (current.id, current.index, current.content_hash) == (
            old.id,
            old.index,
            old.content_hash,
        )
    # New instance proves disk reuse, not only in-process memory reuse.
    reports = CommunitySummarizer(llm=model, cache_dir=tmp_path).summarize_hierarchy(
        updated, graph=view(after_graph)
    )
    assert len(model.prompts) == 3
    for identifier, report in old_reports.items():
        assert reports[identifier].to_dict() == report.to_dict()
    # Cache-disabled recomputation receives precisely the previous prompts.
    reference = Model()
    CommunitySummarizer(llm=reference, cache_enabled=False).summarize_hierarchy(
        updated, graph=view(after_graph)
    )
    assert all(prompt in reference.prompts for prompt in old_prompts)


@pytest.mark.parametrize(
    "change", ["quote", "edge", "chunks", "tokens", "system_prompt"]
)
def test_reuse_never_hides_report_affecting_changes(tmp_path, change):
    builder = CommunityHierarchyBuilder(max_levels=1)
    g = graph(("b1", "b2"))
    previous = builder.build(g)
    model = Model()
    CommunitySummarizer(llm=model, cache_dir=tmp_path).summarize_hierarchy(
        previous, graph=view(g)
    )
    options = {}
    kwargs = {}
    changed = g.copy()
    if change == "quote":
        changed.nodes["b1"]["quote"] = "Corrected original source"
    elif change == "edge":
        changed["b1"]["b2"]["relation"] = "corrected"
    elif change == "chunks":
        kwargs["text_chunks"] = ["Additional original source"]
    elif change == "tokens":
        kwargs["max_tokens"] = 3000
    else:
        options["system_prompt"] = "Different generation instructions"
    updated = builder.build(changed, previous_hierarchy=previous)
    CommunitySummarizer(llm=model, cache_dir=tmp_path, **options).summarize_hierarchy(
        updated, graph=view(changed), **kwargs
    )
    assert len(model.prompts) == 2


def test_default_api_and_serialized_previous_hierarchy_remain_compatible():
    builder = CommunityHierarchyBuilder(max_levels=1)
    g = graph(("b1", "b2"))
    first = builder.build(g)
    assert first.to_dict() == builder.build(g, previous_hierarchy=None).to_dict()
    previous = CommunityHierarchy.from_dict(first.to_dict())
    current = build_community_hierarchy(g, max_levels=1, previous_hierarchy=previous)
    assert (
        members(current)[(0, ("b1", "b2"))].id == members(first)[(0, ("b1", "b2"))].id
    )
    assert "identity_policy" not in first.metadata
    with pytest.raises(TypeError, match="previous_hierarchy"):
        builder.build(g, previous_hierarchy={})


def test_parent_child_identity_dependencies_are_remapped_without_mutating_previous(
    monkeypatch, tmp_path
):
    builder = CommunityHierarchyBuilder()
    old_graph = graph(("b1", "b2"), ("z1", "z2"))
    monkeypatch.setattr(
        builder,
        "_build_louvain_partitions",
        lambda g: [[{"b1", "b2"}, {"z1", "z2"}], [{"b1", "b2", "z1", "z2"}]],
    )
    previous = builder.build(old_graph)
    before = copy.deepcopy(previous.to_dict())
    model = Model()
    CommunitySummarizer(llm=model, cache_dir=tmp_path).summarize_hierarchy(
        previous, graph=view(old_graph)
    )
    assert len(model.prompts) == 3
    new_graph = graph(("a1", "a2"), ("b1", "b2"), ("z1", "z2"))
    monkeypatch.setattr(
        builder,
        "_build_louvain_partitions",
        lambda g: [
            [{"a1", "a2"}, {"b1", "b2"}, {"z1", "z2"}],
            [{"a1", "a2"}, {"b1", "b2", "z1", "z2"}],
        ],
    )
    current = builder.build(new_graph, previous_hierarchy=previous)
    CommunitySummarizer(llm=model, cache_dir=tmp_path).summarize_hierarchy(
        current, graph=view(new_graph)
    )
    assert len(model.prompts) == 5  # only the two communities of the added branch
    assert previous.to_dict() == before
    for c in current.communities.values():
        if c.parent_id:
            assert c.id in current.communities[c.parent_id].child_ids
        for child in c.child_ids:
            assert current.communities[child].parent_id == c.id


def test_changed_child_report_invalidates_parent_even_if_parent_membership_is_stable(
    tmp_path, monkeypatch
):
    builder = CommunityHierarchyBuilder()
    monkeypatch.setattr(
        builder, "_build_louvain_partitions", lambda g: [[{"b1", "b2"}], [{"b1", "b2"}]]
    )
    original = graph(("b1", "b2"))
    previous = builder.build(original)
    model = Model()
    summarizer = CommunitySummarizer(llm=model, cache_dir=tmp_path)
    summarizer.summarize_hierarchy(previous, graph=view(original))
    changed = original.copy()
    changed.nodes["b1"]["quote"] = "Changed evidence"
    updated = builder.build(changed, previous_hierarchy=previous)
    summarizer.summarize_hierarchy(updated, graph=view(changed))
    assert len(model.prompts) == 4


def test_cache_hit_refreshes_parent_and_metrics_without_mutating_stored_report(
    tmp_path,
):
    model = Model()
    summarizer = CommunitySummarizer(llm=model, cache_dir=tmp_path)
    original = HierarchicalCommunity(
        id="stable",
        level=0,
        index=0,
        entity_ids=["x"],
        parent_id="old-parent",
        metrics={"external_edges": 1, "old_metric": 2},
    )
    report = summarizer.summarize_community(original)
    frozen = copy.deepcopy(report.to_dict())
    updated = HierarchicalCommunity(
        id="stable",
        level=0,
        index=0,
        entity_ids=["x"],
        parent_id="new-parent",
        metrics={"external_edges": 3},
    )
    current = summarizer.summarize_community(updated)
    assert len(model.prompts) == 1
    assert current.parent_id == "new-parent"
    assert current.metadata == {"size": 1, "directed": False, "external_edges": 3}
    assert report.to_dict() == frozen
    assert summarizer.get_cached_report(original.content_hash).to_dict() == frozen


def test_split_merge_and_removed_groups_never_recycle_positional_identity(monkeypatch):
    builder = CommunityHierarchyBuilder(max_levels=1)
    g = graph(("a", "b"), ("c", "d"))
    monkeypatch.setattr(
        builder, "_build_louvain_partitions", lambda g: [[{"a", "b"}, {"c", "d"}]]
    )
    previous = builder.build(g)
    monkeypatch.setattr(
        builder, "_build_louvain_partitions", lambda g: [[{"a", "b", "c", "d"}]]
    )
    merged = builder.build(g, previous_hierarchy=previous)
    assert not set(merged.communities) & set(previous.communities)
    monkeypatch.setattr(
        builder, "_build_louvain_partitions", lambda g: [[{"a", "b"}, {"c", "d"}]]
    )
    split = builder.build(g, previous_hierarchy=merged)
    assert not set(split.communities) & set(merged.communities)
    assert builder.build(g, previous_hierarchy=merged).to_dict() == split.to_dict()


def test_ambiguous_previous_membership_is_rejected():
    duplicate = CommunityHierarchy(
        [
            HierarchicalCommunity(id="first", level=0, index=0, entity_ids=["a", "b"]),
            HierarchicalCommunity(id="second", level=0, index=1, entity_ids=["a", "b"]),
        ]
    )
    with pytest.raises(ValueError, match="ambiguous"):
        CommunityHierarchyBuilder(max_levels=1).build(
            graph(("a", "b")), previous_hierarchy=duplicate
        )


def test_child_summary_change_invalidates_parent_generation_without_graph_change(
    tmp_path,
):
    from semantica.kg import CommunityReport

    parent = HierarchicalCommunity(
        id="p", level=1, index=0, entity_ids=["a"], child_ids=["c"]
    )
    child = CommunityReport(
        community_id="c",
        level=0,
        title="First",
        summary="Original account",
        content_hash="stable-structure",
    )
    model = Model()
    s = CommunitySummarizer(llm=model, cache_dir=tmp_path)
    s.summarize_community(parent, child_reports=[child])
    corrected = copy.deepcopy(child)
    corrected.summary = "Corrected account"
    s.summarize_community(parent, child_reports=[corrected])
    assert len(model.prompts) == 2


def test_graph_direction_change_cannot_collide_with_a_previous_generated_identity():
    builder = CommunityHierarchyBuilder(max_levels=1)
    empty_previous = CommunityHierarchy()
    undirected = builder.build(graph(("a", "b")), previous_hierarchy=empty_previous)
    directed = nx.DiGraph(graph(("a", "b")))
    updated = builder.build(directed, previous_hierarchy=undirected)
    assert not set(updated.communities) & set(undirected.communities)


def test_removing_unrelated_group_preserves_remaining_identity():
    builder = CommunityHierarchyBuilder(max_levels=1)
    previous = builder.build(graph(("a", "b"), ("x", "y")))
    updated = builder.build(graph(("x", "y")), previous_hierarchy=previous)
    old = members(previous)[(0, ("x", "y"))]
    assert list(updated.communities) == [old.id]
    assert updated.communities[old.id].content_hash == old.content_hash


def test_inconsistent_previous_dictionary_ids_are_rejected():
    previous = CommunityHierarchy(
        [HierarchicalCommunity(id="actual", level=0, index=0, entity_ids=["a", "b"])]
    )
    previous.communities["wrong"] = previous.communities.pop("actual")
    with pytest.raises(ValueError, match="inconsistent"):
        CommunityHierarchyBuilder(max_levels=1).build(
            graph(("a", "b")), previous_hierarchy=previous
        )
