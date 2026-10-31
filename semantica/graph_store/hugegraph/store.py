"""Public HugeGraph graph-store backend."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

from ...utils.exceptions import ProcessingError, ValidationError
from ...utils.logging import get_logger
from .crud import HugeGraphCrudMixin
from .query import HugeGraphQueryMixin
from .schema import HugeGraphSchemaMixin
from .utils import GremlinExecutor, _as_bool, _call, _positive_int

try:  # Optional dependency.
    from pyhugegraph.client import PyHugeClient

    HUGEGRAPH_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - dependency test.
    PyHugeClient = None  # type: ignore[assignment,misc]
    HUGEGRAPH_AVAILABLE = False


class HugeGraphStore(
    HugeGraphSchemaMixin,
    HugeGraphCrudMixin,
    HugeGraphQueryMixin,
):
    """Implement Semantica's graph-store backend contract for HugeGraph."""

    _NATIVE_ALGORITHMS = {
        "k_out": ("k_out", "kOut"),
        "k_neighbor": ("k_neighbor", "kNeighbor"),
        "same_neighbors": ("same_neighbors", "sameNeighbors"),
        "jaccard_similarity": ("jaccard_similarity", "jaccardSimilarity"),
        "shortest_path": ("shortest_path", "shortestPath"),
        "all_shortest_paths": ("all_shortest_paths", "allShortestPaths"),
        "weighted_shortest_path": (
            "weighted_shortest_path",
            "weightedShortestPath",
        ),
        "single_source_shortest_path": (
            "single_source_shortest_path",
            "singleSourceShortestPath",
        ),
        "paths": ("paths",),
        "personal_rank": ("personal_rank", "personalRank"),
    }

    def __init__(
        self,
        uri: Optional[str] = None,
        graph: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        graphspace: Optional[str] = None,
        timeout: Optional[Union[int, float, Tuple[float, float]]] = None,
        read_only: bool = False,
        client: Optional[Any] = None,
        gremlin_executor: Optional[GremlinExecutor] = None,
        **config: Any,
    ) -> None:
        self.logger = get_logger("hugegraph_store")
        self.config = config

        configured_uri = (
            uri
            or config.get("url")
            or config.get("uri")
            or "http://localhost:8080"
        )
        self.uri = configured_uri.rstrip("/")
        self.graph_name = graph or config.get("graph") or "hugegraph"
        self.user = user if user is not None else config.get("user", "admin")
        self.password = (
            password
            if password is not None
            else config.get("pwd", config.get("password", "hugegraph"))
        )
        self.graphspace = (
            graphspace if graphspace is not None else config.get("graphspace")
        )
        self.timeout = (
            timeout if timeout is not None else config.get("timeout", (0.5, 15.0))
        )
        self.read_only = _as_bool(config.get("read_only", read_only))

        configured_properties = config.get(
            "search_properties",
            (
                "content",
                "text",
                "name",
                "statement",
                "summary",
                "description",
                "title",
            ),
        )
        if isinstance(configured_properties, str):
            configured_properties = [
                item.strip()
                for item in configured_properties.split(",")
                if item.strip()
            ]
        self.search_properties = tuple(str(item) for item in configured_properties)
        self.search_candidate_limit = _positive_int(
            config.get("search_candidate_limit", 500),
            "search_candidate_limit",
        )
        self._client = client
        self._owns_client = False
        self._gremlin_executor = gremlin_executor

    def _timeout_tuple(self) -> Tuple[float, float]:
        if isinstance(self.timeout, tuple) and len(self.timeout) == 2:
            return (float(self.timeout[0]), float(self.timeout[1]))
        seconds = float(self.timeout)
        return (min(1.0, seconds), seconds)

    def connect(self, **options: Any) -> bool:
        if self._client is None:
            if not HUGEGRAPH_AVAILABLE or PyHugeClient is None:
                raise ProcessingError(
                    "HugeGraph support requires the optional "
                    "'hugegraph-python' package"
                )
            try:
                self._client = PyHugeClient(
                    self.uri,
                    self.graph_name,
                    self.user,
                    self.password,
                    graphspace=self.graphspace,
                    timeout=self._timeout_tuple(),
                )
                self._owns_client = True
            except Exception as exc:
                raise ProcessingError(
                    f"Failed to create HugeGraph client: {exc}"
                ) from exc

        if options.get("verify", True):
            try:
                version_manager = _call(self._client, ("version",))
                _call(version_manager, ("version",))
            except ProcessingError:
                self.get_schema()
            except Exception as exc:
                raise ProcessingError(
                    f"Failed to connect to HugeGraph at {self.uri}: {exc}"
                ) from exc
        return True

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception as exc:
                self.logger.warning(
                    "Failed to close HugeGraph client cleanly: %s",
                    exc,
                )
        if self._owns_client:
            self._client = None
            self._owns_client = False

    def _client_instance(self) -> Any:
        if self._client is None:
            self.connect(verify=False)
        return self._client

    def _manager(self, name: str) -> Any:
        return _call(self._client_instance(), (name,))

    def _ensure_write(self, operation: str) -> None:
        if self.read_only:
            raise ValidationError(
                f"HugeGraphStore is read-only; {operation} is allowed only "
                "on an isolated test graph"
            )
