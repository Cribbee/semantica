"""Shared normalization and validation helpers for the HugeGraph backend."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

from ...utils.exceptions import ProcessingError, ValidationError

NodeId = Union[int, str]
GremlinExecutor = Callable[[str, Optional[Dict[str, Any]]], Any]
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CJK = re.compile(r"[\u3400-\u9fff]")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _positive_int(
    value: Any,
    name: str,
    *,
    allow_zero: bool = False,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid {name}: {value!r}") from exc
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        raise ValidationError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _identifier(value: str, kind: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValidationError(f"Invalid HugeGraph {kind}: {value!r}")
    return value


def _find_method(target: Any, *names: str) -> Callable[..., Any]:
    for name in names:
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    raise ProcessingError(
        f"{type(target).__name__} does not provide any of: " f"{', '.join(names)}"
    )


def _call(
    target: Any,
    names: Sequence[str],
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _find_method(target, *names)(*args, **kwargs)


def _chain(
    target: Any,
    names: Sequence[str],
    *args: Any,
) -> Any:
    result = _call(target, names, *args)
    return target if result is None else result


def _plain(value: Any) -> Any:
    """Convert pyhugegraph structures and containers to plain data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    for fields in (
        (
            "id",
            "label",
            "type",
            "outV",
            "outVLabel",
            "inV",
            "inVLabel",
            "properties",
        ),
        ("id", "label", "type", "properties"),
    ):
        if any(hasattr(value, field) for field in fields):
            return {
                field: _plain(getattr(value, field))
                for field in fields
                if hasattr(value, field)
            }
    if hasattr(value, "__dict__"):
        return {
            key.lstrip("_"): _plain(item)
            for key, item in vars(value).items()
            if not key.startswith("__")
        }
    return value


def _node_dict(
    value: Any,
    *,
    labels: Optional[Sequence[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw = _plain(value)
    if not isinstance(raw, dict):
        raw = {"id": raw}
    label = raw.get("label")
    raw_labels = raw.get("labels")
    if isinstance(raw_labels, str):
        normalized_labels = [raw_labels]
    elif isinstance(raw_labels, list):
        normalized_labels = [str(item) for item in raw_labels]
    elif label is not None:
        normalized_labels = [str(label)]
    else:
        normalized_labels = [str(item) for item in (labels or [])]
    raw_properties = raw.get("properties")
    normalized_properties = (
        dict(raw_properties)
        if isinstance(raw_properties, dict)
        else dict(properties or {})
    )
    return {
        "id": raw.get("id"),
        "labels": normalized_labels,
        "properties": normalized_properties,
    }


def _edge_dict(
    value: Any,
    *,
    rel_type: Optional[str] = None,
    start_node_id: Optional[NodeId] = None,
    end_node_id: Optional[NodeId] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw = _plain(value)
    if not isinstance(raw, dict):
        raw = {"id": raw}
    raw_properties = raw.get("properties")
    return {
        "id": raw.get("id"),
        "type": raw.get("label") or raw.get("type") or rel_type,
        "start_node_id": raw.get(
            "outV",
            raw.get("start_node_id", start_node_id),
        ),
        "end_node_id": raw.get(
            "inV",
            raw.get("end_node_id", end_node_id),
        ),
        "properties": (
            dict(raw_properties)
            if isinstance(raw_properties, dict)
            else dict(properties or {})
        ),
    }


def _records(value: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize Gremlin data to Semantica's query-record envelope."""
    data = _plain(value)
    if data is None:
        items: List[Any] = []
    elif isinstance(data, list):
        items = data
    else:
        items = [data]

    records: List[Dict[str, Any]] = []
    keys: List[str] = []
    for item in items:
        record = dict(item) if isinstance(item, dict) else {"value": item}
        records.append(record)
        for key in record:
            if key not in keys:
                keys.append(key)
    return records, keys


def _result_data(value: Any) -> Any:
    """Extract ``result.data`` from a raw Gremlin response."""
    if not isinstance(value, dict):
        return value
    result = value.get("result")
    if isinstance(result, dict) and "data" in result:
        return result.get("data")
    response_keys = {"status", "requestId", "request_id"}
    if "data" in value and set(value).intersection(response_keys):
        return value.get("data")
    return value


def _schema_specs(
    schema: Mapping[str, Any],
    collection: str,
) -> List[Dict[str, Any]]:
    value = schema.get(collection, []) or []
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise ValidationError(
            f"HugeGraph schema {collection!r} must be a list of objects"
        )
    return [dict(item) for item in value]


def _schema_value(key: str, value: Any) -> Any:
    if isinstance(value, list):
        normalized = [str(item) for item in value]
        ordered_keys = {"primary_keys", "sort_keys", "fields"}
        return normalized if key in ordered_keys else sorted(normalized)
    if key == "enable_label_index":
        return bool(value)
    if isinstance(value, str) and key in {
        "data_type",
        "cardinality",
        "id_strategy",
        "frequency",
        "base_type",
        "index_type",
    }:
        return value.upper()
    return value


def _validate_existing_schema(
    current: Mapping[str, Any],
    requested: Mapping[str, Any],
) -> None:
    """Reject name collisions whose strong-schema fields differ."""
    comparable = {
        "propertykeys": ("data_type", "cardinality"),
        "vertexlabels": (
            "id_strategy",
            "properties",
            "nullable_keys",
            "primary_keys",
            "enable_label_index",
        ),
        "edgelabels": (
            "source_label",
            "target_label",
            "properties",
            "nullable_keys",
            "frequency",
            "sort_keys",
            "enable_label_index",
        ),
        "indexlabels": (
            "base_type",
            "base_value",
            "index_type",
            "fields",
        ),
    }
    for collection, keys in comparable.items():
        actual_by_name = {
            str(item.get("name")): item
            for item in _schema_specs(current, collection)
            if item.get("name") is not None
        }
        for expected in _schema_specs(requested, collection):
            name = str(expected.get("name", ""))
            actual = actual_by_name.get(name)
            if actual is None:
                continue
            mismatches = []
            for key in keys:
                if key not in expected:
                    continue
                actual_value = _schema_value(key, actual.get(key))
                expected_value = _schema_value(key, expected.get(key))
                if actual_value != expected_value:
                    mismatches.append(
                        f"{key}: existing={actual.get(key)!r}, "
                        f"requested={expected.get(key)!r}"
                    )
            if mismatches:
                raise ValidationError(
                    f"Existing HugeGraph {collection[:-1]} {name!r} "
                    "is incompatible: " + "; ".join(mismatches)
                )


def _text_fragments(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _text_fragments(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _text_fragments(item)
    elif value is not None and isinstance(value, (int, float, bool)):
        yield str(value)


def _query_terms(query: str) -> List[str]:
    normalized = query.casefold().strip()
    if not normalized:
        return []
    terms = [
        term
        for term in re.findall(
            r"[\w\u3400-\u9fff]+",
            normalized,
        )
        if term
    ]
    # Chinese questions are often one token. Character bigrams preserve
    # deterministic matching without adding a tokenizer dependency.
    if _CJK.search(normalized):
        cjk = "".join(char for char in normalized if _CJK.fullmatch(char))
        terms.extend(cjk[index : index + 2] for index in range(max(0, len(cjk) - 1)))
        terms.extend(char for char in cjk if len(cjk) <= 8)
    return list(dict.fromkeys(term for term in terms if term))


def _lexical_score(query: str, values: Iterable[str]) -> float:
    haystack = "\n".join(values).casefold()
    normalized = query.casefold().strip()
    if not normalized or not haystack:
        return 0.0
    if normalized in haystack:
        return 1.0
    terms = _query_terms(normalized)
    if not terms:
        return 0.0
    matched = [term for term in terms if term in haystack]
    if not matched:
        return 0.0
    coverage = len(matched) / len(terms)
    density = min(
        1.0,
        sum(len(term) for term in matched) / max(1, len(normalized)),
    )
    return min(0.99, 0.75 * coverage + 0.25 * density)

