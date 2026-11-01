from __future__ import annotations

import json

import pytest

from semantica.graph_store.hugegraph_store import HugeGraphStore
from semantica.utils.exceptions import ProcessingError, ValidationError


class RestExecutor:
    def __init__(self):
        self.calls = []
        self.polls = 0

    def __call__(self, method, url, payload):
        self.calls.append((method, url, payload))
        if method == "POST":
            return {"task_id": 42}
        if method == "GET" and "with_result=false" in url:
            self.polls += 1
            status = "RUNNING" if self.polls == 1 else "SUCCESS"
            return {"task_id": 42, "task_status": status}
        if method == "GET" and "with_result=true" in url:
            return {
                "task_id": 42,
                "task_status": "SUCCESS",
                "task_result": json.dumps({"v1": 3, "v2": 1}),
            }
        if method == "PUT":
            return {"task_id": 42, "task_status": "CANCELLING"}
        if method == "DELETE":
            return None
        raise AssertionError((method, url, payload))


def make_store(executor=None, **options):
    return HugeGraphStore(
        uri="http://localhost:8080",
        graph="semantica_test",
        graphspace="DEFAULT",
        user="",
        password="",
        client=object(),
        rest_executor=executor or RestExecutor(),
        **options,
    )


def test_olap_run_uses_current_algorithm_and_task_api():
    executor = RestExecutor()
    store = make_store(executor)

    result = store.run_olap_algorithm(
        "degree",
        {"direction": "BOTH", "top": 10},
        timeout=1,
        poll_interval=0,
    )

    assert result["algorithm"] == "degree_centrality"
    assert result["task_id"] == 42
    assert result["result"] == {"v1": 3, "v2": 1}
    assert executor.calls[0] == (
        "POST",
        "http://localhost:8080/graphspaces/DEFAULT/graphs/"
        "semantica_test/jobs/algorithm/degree_centrality",
        {"direction": "BOTH", "top": 10},
    )
    assert any("with_result=false" in call[1] for call in executor.calls)
    assert any("with_result=true" in call[1] for call in executor.calls)


def test_mutating_algorithms_require_isolated_graph_opt_in():
    executor = RestExecutor()
    store = make_store(executor)

    with pytest.raises(ValidationError, match="allow_mutation=True"):
        store.submit_olap_algorithm("pagerank", {"times": 10})

    task_id = store.submit_olap_algorithm(
        "pagerank",
        {"times": 10},
        allow_mutation=True,
    )
    assert task_id == 42
    assert executor.calls[-1][1].endswith("/jobs/algorithm/page_rank")

    read_only = make_store(RestExecutor(), read_only=True)
    with pytest.raises(ValidationError, match="read-only"):
        read_only.submit_olap_algorithm(
            "page_rank",
            {},
            allow_mutation=True,
        )


def test_task_lifecycle_and_capabilities_are_normalized():
    executor = RestExecutor()
    store = make_store(executor)

    cancelled = store.cancel_olap_task(42)
    assert cancelled["task_status"] == "CANCELLING"
    assert store.delete_olap_task(42, force=True) is None

    capabilities = store.analytics.capabilities()
    olap = {item["name"]: item for item in capabilities["olap"]}
    traversers = {item["name"] for item in capabilities["traversers"]}
    assert olap["page_rank"]["mutates_graph"] is True
    assert olap["degree_centrality"]["mutates_graph"] is False
    assert {"louvain", "triangle_count", "weak_connected_component"} <= set(olap)
    assert {"shortest_path", "k_out", "personal_rank"} <= traversers


def test_task_operations_reuse_client_when_options_are_supported():
    class TaskManager:
        def cancel_task(self, task_id):
            return {"task_id": task_id, "task_status": "CANCELLING"}

        def delete_task(self, task_id):
            return {"deleted": task_id}

    class Client:
        def task(self):
            return TaskManager()

    store = HugeGraphStore(client=Client())
    assert store.cancel_olap_task(42)["task_status"] == "CANCELLING"
    assert store.delete_olap_task(42) == {"deleted": 42}


def test_failed_task_surfaces_server_error():
    def executor(method, url, payload):
        del url, payload
        if method == "POST":
            return {"task_id": 9}
        return {
            "task_id": 9,
            "task_status": "FAILED",
            "task_error": "invalid edge label",
        }

    store = make_store(executor)
    with pytest.raises(ProcessingError, match="invalid edge label"):
        store.run_olap_algorithm(
            "degree_centrality",
            {},
            timeout=1,
            poll_interval=0,
        )


def test_transport_validation_and_injected_client_ownership():
    with pytest.raises(ValidationError, match="plain HTTP"):
        HugeGraphStore(
            uri="http://hugegraph.example:8080",
            user="admin",
            password="secret",
            client=object(),
        )

    store = HugeGraphStore(
        uri="http://hugegraph.example:8080",
        user="admin",
        password="secret",
        allow_insecure_auth=True,
        client=object(),
    )
    assert store.uri == "http://hugegraph.example:8080"

    class SharedClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    client = SharedClient()
    HugeGraphStore(client=client).close()
    assert client.closed is False

