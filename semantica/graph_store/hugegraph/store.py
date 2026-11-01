"""Public HugeGraph graph-store backend."""

from __future__ import annotations

import base64
from ipaddress import ip_address
from typing import Any, Optional, Tuple, Union
from urllib.parse import urlparse

from ...utils.exceptions import ProcessingError, ValidationError
from ...utils.logging import get_logger
from .analytics import HugeGraphAnalytics, HugeGraphOlapMixin, RestExecutor
from .crud import HugeGraphCrudMixin
from .query import HugeGraphQueryMixin
from .safety import HugeGraphSafetyMixin
from .schema import HugeGraphSchemaMixin
from .utils import GremlinExecutor, _as_bool, _call, _positive_int

try:  # Optional dependency.
    from pyhugegraph.client import PyHugeClient

    HUGEGRAPH_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - dependency test.
    PyHugeClient = None  # type: ignore[assignment,misc]
    HUGEGRAPH_AVAILABLE = False


class HugeGraphStore(
    HugeGraphSafetyMixin,
    HugeGraphOlapMixin,
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
        allow_insecure_auth: bool = False,
        client: Optional[Any] = None,
        gremlin_executor: Optional[GremlinExecutor] = None,
        rest_executor: Optional[RestExecutor] = None,
        **config: Any,
    ) -> None:
        self.logger = get_logger("hugegraph_store")
        self.config = config

        configured_uri = (
            uri or config.get("url") or config.get("uri") or "http://localhost:8080"
        )
        self.uri = configured_uri.rstrip("/")
        parsed_uri = urlparse(self.uri)
        if parsed_uri.scheme not in {"http", "https"} or not parsed_uri.netloc:
            raise ValidationError(
                "HugeGraph uri must be an absolute http:// or https:// URL"
            )

        self.graph_name = graph or config.get("graph") or "hugegraph"
        self.user = user if user is not None else config.get("user", "admin")
        self.password = (
            password
            if password is not None
            else config.get("pwd", config.get("password", "hugegraph"))
        )
        if self.password and not self.user:
            raise ValidationError(
                "HugeGraph password was provided without a user name"
            )

        self.allow_insecure_auth = _as_bool(
            config.get("allow_insecure_auth", allow_insecure_auth)
        )
        if (
            parsed_uri.scheme == "http"
            and self.user
            and not self.allow_insecure_auth
            and not self._is_loopback_host(parsed_uri.hostname)
        ):
            raise ValidationError(
                "Refusing to send HugeGraph credentials over plain HTTP to a "
                "non-loopback host; use HTTPS or set allow_insecure_auth=True"
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
        self._rest_executor = rest_executor
        self.analytics = HugeGraphAnalytics(self)

    @staticmethod
    def _is_loopback_host(hostname: Optional[str]) -> bool:
        if not hostname:
            return False
        normalized = hostname.strip().lower()
        if normalized == "localhost":
            return True
        try:
            return ip_address(normalized).is_loopback
        except ValueError:
            return False

    def _authorization_header(self) -> Optional[str]:
        if not self.user:
            return None
        credentials = f"{self.user}:{self.password}".encode("utf-8")
        token = base64.b64encode(credentials).decode("ascii")
        return f"Basic {token}"

    def _timeout_tuple(self) -> Tuple[float, float]:
        if isinstance(self.timeout, tuple) and len(self.timeout) == 2:
            return (float(self.timeout[0]), float(self.timeout[1]))
        seconds = float(self.timeout)
        return (min(1.0, seconds), seconds)

    def _validated_http_uri(self) -> str:
        """Validate scheme and protect Basic credentials on remote endpoints."""
        parsed = urlparse(self.uri)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValidationError("HugeGraph URI must be an absolute http or https URL")
        is_loopback = self._is_loopback_host(parsed.hostname)
        if (
            parsed.scheme == "http"
            and self.user
            and not is_loopback
            and not self.allow_insecure_auth
        ):
            raise ValidationError(
                "Refusing to send HugeGraph Basic credentials over remote HTTP; "
                "use HTTPS or a loopback endpoint"
            )
        return self.uri

    def connect(self, **options: Any) -> bool:
        if self._client is None:
            self._validated_http_uri()
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
        # A caller-injected client can be shared with other components; only
        # close resources created by this store instance.
        if not self._owns_client or self._client is None:
            return
        if hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception as exc:
                self.logger.warning(
                    "Failed to close HugeGraph client cleanly: %s",
                    exc,
                )
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

