"""HugeGraph Server Algorithm Job and Task REST client."""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable, Mapping
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ...utils.exceptions import ProcessingError, ValidationError
from .catalog import (
    OLAP_ALGORITHMS,
    TaskId,
    _ALIASES,
    _FAILURE,
    _SUCCESS,
    _decode_json,
)

RestExecutor = Callable[[str, str, Optional[Dict[str, Any]]], Any]


class HugeGraphOlapMixin:
    """Call HugeGraph Server's native Algorithm Job and Task APIs."""

    @staticmethod
    def _normalize_algorithm_name(name: str) -> str:
        normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        return _ALIASES.get(normalized, normalized)

    def list_olap_algorithms(self) -> List[Dict[str, Any]]:
        """Return the explicitly supported Server OLAP capability matrix."""
        return [OLAP_ALGORITHMS[name].as_dict() for name in sorted(OLAP_ALGORITHMS)]

    def get_olap_algorithm(self, name: str) -> Dict[str, Any]:
        normalized = self._normalize_algorithm_name(name)
        capability = OLAP_ALGORITHMS.get(normalized)
        if capability is None:
            supported = ", ".join(sorted(OLAP_ALGORITHMS))
            raise ValidationError(
                f"Unsupported HugeGraph OLAP algorithm {name!r}; supported: {supported}"
            )
        return capability.as_dict()

    def _graph_api_root(self) -> str:
        configured = self.config.get("graph_api_root")
        graph = quote(str(self.graph_name), safe="")
        graphspace = quote(str(self.graphspace or "DEFAULT"), safe="")
        if configured:
            return str(configured).rstrip("/").format(
                uri=self.uri,
                graph=graph,
                graphspace=graphspace,
            )
        return f"{self.uri}/graphspaces/{graphspace}/graphs/{graph}"

    def _rest_request(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        executor = getattr(self, "_rest_executor", None)
        if executor is not None:
            return _decode_json(executor(method, url, payload))

        headers = {"Accept": "application/json", "Accept-Encoding": "gzip"}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        authorization = self._authorization_header()
        if authorization:
            headers["Authorization"] = authorization

        request = Request(url, data=body, method=method, headers=headers)
        try:
            with urlopen(  # nosec B310 - validated HugeGraph endpoint
                request,
                timeout=self._timeout_tuple()[1],
            ) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                if not raw:
                    return None
                return _decode_json(raw)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProcessingError(
                f"HugeGraph REST {method} {url} failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise ProcessingError(
                f"HugeGraph REST {method} {url} failed: {exc.reason}"
            ) from exc

    @staticmethod
    def _extract_task_id(response: Any) -> TaskId:
        response = _decode_json(response)
        if isinstance(response, Mapping):
            for key in ("task_id", "taskId", "id"):
                if response.get(key) is not None:
                    return response[key]
            for key in ("task", "data"):
                nested = response.get(key)
                if isinstance(nested, Mapping):
                    try:
                        return HugeGraphOlapMixin._extract_task_id(nested)
                    except ProcessingError:
                        pass
        if isinstance(response, (int, str)) and str(response).strip():
            return response
        raise ProcessingError(f"HugeGraph did not return a task id: {response!r}")

    @staticmethod
    def _task_record(response: Any) -> Dict[str, Any]:
        response = _decode_json(response)
        if not isinstance(response, Mapping):
            raise ProcessingError(f"Invalid HugeGraph task response: {response!r}")
        for key in ("task", "data"):
            nested = response.get(key)
            if isinstance(nested, Mapping):
                return dict(nested)
        return dict(response)

    @staticmethod
    def _task_status(task: Mapping[str, Any]) -> str:
        for key in ("task_status", "status", "state"):
            if task.get(key) is not None:
                return str(task[key]).strip().lower()
        return ""

    @staticmethod
    def _extract_task_result(task: Mapping[str, Any]) -> Any:
        for key in ("task_result", "result", "output"):
            if key in task:
                return _decode_json(task[key])
        return dict(task)

    def submit_olap_algorithm(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        allow_mutation: bool = False,
    ) -> TaskId:
        normalized = self._normalize_algorithm_name(name)
        capability = OLAP_ALGORITHMS.get(normalized)
        if capability is None:
            self.get_olap_algorithm(name)  # raises a detailed error
            raise AssertionError("unreachable")
        if capability.mutates_graph:
            if self.read_only:
                raise ValidationError(
                    f"HugeGraph algorithm {normalized!r} cannot run in read-only mode"
                )
            if not allow_mutation:
                raise ValidationError(
                    f"HugeGraph algorithm {normalized!r} may write schema or "
                    "properties; run it on an isolated graph and pass "
                    "allow_mutation=True"
                )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ValidationError("HugeGraph algorithm parameters must be a mapping")

        encoded = quote(normalized, safe="")
        response = self._rest_request(
            "POST",
            f"{self._graph_api_root()}/jobs/algorithm/{encoded}",
            dict(parameters or {}),
        )
        return self._extract_task_id(response)

    def get_olap_task(
        self,
        task_id: TaskId,
        *,
        with_result: bool = True,
    ) -> Dict[str, Any]:
        encoded = quote(str(task_id), safe="")
        query = urlencode({"with_result": str(bool(with_result)).lower()})
        response = self._rest_request(
            "GET", f"{self._graph_api_root()}/tasks/{encoded}?{query}"
        )
        return self._task_record(response)

    def get_olap_task_result(self, task_id: TaskId) -> Any:
        return self._extract_task_result(
            self.get_olap_task(task_id, with_result=True)
        )

    def cancel_olap_task(self, task_id: TaskId) -> Dict[str, Any]:
        if self._rest_executor is None and not self.config.get("graph_api_root"):
            return self._task_record(self._manager("task").cancel_task(task_id))
        encoded = quote(str(task_id), safe="")
        response = self._rest_request(
            "PUT",
            f"{self._graph_api_root()}/tasks/{encoded}?"
            f"{urlencode({'action': 'cancel'})}",
        )
        return self._task_record(response)

    def delete_olap_task(self, task_id: TaskId, *, force: bool = False) -> Any:
        if (
            not force
            and self._rest_executor is None
            and not self.config.get("graph_api_root")
        ):
            return self._manager("task").delete_task(task_id)
        encoded = quote(str(task_id), safe="")
        query = urlencode({"force": str(bool(force)).lower()})
        return self._rest_request(
            "DELETE", f"{self._graph_api_root()}/tasks/{encoded}?{query}"
        )

    def wait_olap_task(
        self,
        task_id: TaskId,
        *,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> Dict[str, Any]:
        timeout = float(timeout)
        if timeout <= 0:
            raise ValidationError("timeout must be greater than zero")
        poll_interval = max(0.0, float(poll_interval))
        deadline = time.monotonic() + timeout
        last_task: Dict[str, Any] = {}

        while time.monotonic() < deadline:
            last_task = self.get_olap_task(task_id, with_result=False)
            status = self._task_status(last_task)
            if status in _SUCCESS:
                return self.get_olap_task(task_id, with_result=True)
            if status in _FAILURE:
                detail = (
                    last_task.get("task_error")
                    or last_task.get("error")
                    or last_task.get("message")
                    or last_task.get("task_result")
                    or last_task
                )
                raise ProcessingError(
                    f"HugeGraph task {task_id!r} ended with status {status!r}: {detail}"
                )
            time.sleep(poll_interval)

        status = self._task_status(last_task) or "unknown"
        raise ProcessingError(
            f"Timed out waiting for HugeGraph task {task_id!r}; last status: {status}"
        )

    def run_olap_algorithm(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        allow_mutation: bool = False,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> Dict[str, Any]:
        normalized = self._normalize_algorithm_name(name)
        task_id = self.submit_olap_algorithm(
            normalized,
            parameters,
            allow_mutation=allow_mutation,
        )
        task = self.wait_olap_task(
            task_id,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        return {
            "algorithm": normalized,
            "task_id": task_id,
            "task": task,
            "result": self._extract_task_result(task),
            "capability": self.get_olap_algorithm(normalized),
        }

