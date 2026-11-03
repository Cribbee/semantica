"""Bounded report concurrency preserves dependency and output ordering."""

import hashlib
import json
import re
import threading
import time

import pytest

from semantica.kg import (
    CommunityHierarchy,
    CommunitySummarizer,
    HierarchicalCommunity,
    summarize_hierarchy,
)


def hierarchy():
    leaves = [
        HierarchicalCommunity(
            id=f"leaf-{i}",
            level=0,
            index=i,
            entity_ids=[f"entity-{i}"],
            parent_id="parent",
        )
        for i in range(8)
    ]
    parent = HierarchicalCommunity(
        id="parent",
        level=1,
        index=0,
        entity_ids=[f"entity-{i}" for i in range(8)],
        child_ids=[c.id for c in leaves],
    )
    return CommunityHierarchy([*leaves, parent])


class FixedModel:
    def __init__(self, workers=1, fail=False):
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(workers) if workers > 1 else None
        self.active = self.maximum = 0
        self.started = []
        self.finished = []
        self.fail = fail

    def __call__(self, prompt, **options):
        identifier = re.search(r"Community ID: (\S+)", prompt).group(1)
        with self.lock:
            ordinal = len(self.started)
            self.started.append(identifier)
            self.active += 1
            self.maximum = max(self.active, self.maximum)
            if identifier == "parent":
                assert set(self.finished) == {f"leaf-{i}" for i in range(8)}
        try:
            if self.barrier is not None and ordinal < self.barrier.parties:
                self.barrier.wait(timeout=5)
            if self.fail and identifier == "leaf-0":
                raise Fatal("confirmed provider failure")
            # Deliberately finish siblings in a different order.
            time.sleep(
                0.005 * (8 - int(identifier[-1]))
                if identifier.startswith("leaf-")
                else 0
            )
            return json.dumps(
                {
                    "title": identifier,
                    "summary": hashlib.sha256(prompt.encode()).hexdigest(),
                    "findings": [],
                    "impact_rating": 5.0,
                }
            )
        finally:
            with self.lock:
                self.active -= 1
                self.finished.append(identifier)


class Fatal(BaseException):
    pass


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_reports_and_cache_bytes_match_serial_with_out_of_order_completion(
    tmp_path, workers
):
    h = hierarchy()
    serial_model, parallel_model = FixedModel(), FixedModel(workers)
    one, many = tmp_path / "one", tmp_path / "many"
    serial = CommunitySummarizer(llm=serial_model, cache_dir=one).summarize_hierarchy(h)
    parallel = CommunitySummarizer(
        llm=parallel_model, cache_dir=many
    ).summarize_hierarchy(h, max_workers=workers)
    assert list(parallel) == list(serial)
    assert {k: v.to_dict() for k, v in parallel.items()} == {
        k: v.to_dict() for k, v in serial.items()
    }
    assert parallel_model.maximum == workers
    assert parallel_model.finished != parallel_model.started
    assert len(parallel_model.started) == len(set(parallel_model.started)) == 9
    assert {p.name: p.read_bytes() for p in one.glob("*.json")} == {
        p.name: p.read_bytes() for p in many.glob("*.json")
    }
    cached = CommunitySummarizer(
        llm=lambda *a, **kw: pytest.fail("cached reports must not call the provider"),
        cache_dir=many,
    ).summarize_hierarchy(h, max_workers=workers)
    assert {k: v.to_dict() for k, v in cached.items()} == {
        k: v.to_dict() for k, v in serial.items()
    }


def test_default_and_explicit_one_worker_are_serial():
    for options in [{}, {"max_workers": 1}]:
        model = FixedModel()
        CommunitySummarizer(llm=model).summarize_hierarchy(hierarchy(), **options)
        assert model.maximum == 1
        assert model.started == [f"leaf-{i}" for i in range(8)] + ["parent"]


def test_fatal_failure_has_bounded_dispatch_no_parent_and_keeps_valid_cache(tmp_path):
    model = FixedModel(2, fail=True)
    s = CommunitySummarizer(llm=model, cache_dir=tmp_path)
    with pytest.raises(Fatal):
        s.summarize_hierarchy(hierarchy(), max_workers=2)
    assert len(model.started) <= 2
    assert "parent" not in model.started and model.active == 0
    assert "leaf-0" not in [
        json.loads(p.read_text())["community_id"] for p in tmp_path.glob("*.json")
    ]
    # Completed sibling work survives; the failed report itself was not synthesized.
    successful = set(model.started) - {"leaf-0"}
    model = FixedModel()
    model.finished = sorted(successful)
    reports = CommunitySummarizer(llm=model, cache_dir=tmp_path).summarize_hierarchy(
        hierarchy(), max_workers=2
    )
    assert len(reports) == 9
    assert successful.isdisjoint(model.started)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2", None])
def test_invalid_bounds_fail_before_empty_hierarchy_or_provider(workers):
    with pytest.raises(ValueError, match="positive integer"):
        CommunitySummarizer().summarize_hierarchy(
            CommunityHierarchy(), max_workers=workers
        )
    with pytest.raises(ValueError, match="positive integer"):
        summarize_hierarchy(CommunityHierarchy(), max_workers=workers)


@pytest.mark.parametrize("child", ["missing", "peer", "parent"])
def test_invalid_parallel_dependencies_fail_before_dispatch(child):
    p = HierarchicalCommunity(
        id="parent", level=0, index=0, entity_ids=["p"], child_ids=[child]
    )
    peer = HierarchicalCommunity(id="peer", level=0, index=1, entity_ids=["q"])
    with pytest.raises(ValueError, match="children|dependencies"):
        CommunitySummarizer(
            llm=lambda *a, **kw: pytest.fail("must not dispatch")
        ).summarize_hierarchy(CommunityHierarchy([p, peer]), max_workers=2)


def test_level_selection_still_builds_children_and_returns_only_requested_level():
    model = FixedModel(2)
    result = CommunitySummarizer(llm=model).summarize_hierarchy(
        hierarchy(), levels=[1], max_workers=2
    )
    assert list(result) == ["parent"]
    assert len(model.started) == 9


def test_convenience_function_forwards_worker_bound():
    model = FixedModel(2)
    result = summarize_hierarchy(hierarchy(), llm=model, max_workers=2)
    assert len(result) == 9 and model.maximum == 2


def test_shared_embedder_is_serialized_while_report_generation_is_parallel():
    active = maximum = 0
    lock = threading.Lock()

    def embed(text):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.003)
            assert active == 1
            return [float(len(text))]
        finally:
            with lock:
                active -= 1

    model = FixedModel(4)
    reports = CommunitySummarizer(llm=model, embedder=embed).summarize_hierarchy(
        hierarchy(), max_workers=4
    )
    assert model.maximum == 4 and maximum == 1
    assert all(report.embedding for report in reports.values())
