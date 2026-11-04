"""Serializable, immutable query plans shared by multiple retrieval consumers.

Plans describe information needs, not established facts or evidence. Product
approval, requiredness and execution versions belong to the caller.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class QueryPlanItem:
    """One stable information need; relevance is retrieval priority, not truth."""

    item_id: str
    sub_query: str
    target_entities: Tuple[str, ...] = ()
    rationale: str = ""
    relevance_score: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise ValueError("Plan item_id must be nonempty")
        if not isinstance(self.sub_query, str) or not self.sub_query.strip():
            raise ValueError("Plan sub_query must be nonempty")
        if not isinstance(self.target_entities, (list, tuple)) or not all(
            isinstance(entity, str) and entity.strip()
            for entity in self.target_entities
        ):
            raise ValueError("Plan target_entities must contain nonempty strings")
        if not isinstance(self.rationale, str):
            raise ValueError("Plan rationale must be a string")
        if (
            isinstance(self.relevance_score, bool)
            or not isinstance(self.relevance_score, (float, int))
            or not math.isfinite(self.relevance_score)
            or not 0 <= self.relevance_score <= 1
        ):
            raise ValueError("Plan relevance_score must be between zero and one")
        # Normalize JSON integer/float spelling across JavaScript round trips.
        object.__setattr__(self, "relevance_score", float(self.relevance_score) or 0.0)
        object.__setattr__(self, "target_entities", tuple(self.target_entities))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "sub_query": self.sub_query,
            "target_entities": list(self.target_entities),
            "rationale": self.rationale,
            "relevance_score": self.relevance_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryPlanItem":
        if not isinstance(data, dict) or set(data) - {
            "item_id",
            "sub_query",
            "target_entities",
            "rationale",
            "relevance_score",
        }:
            raise ValueError("Invalid query plan item fields")
        if not {"item_id", "sub_query"}.issubset(data):
            raise ValueError("Missing query plan item fields")
        entities = data.get("target_entities", [])
        if not isinstance(entities, (tuple, list)):
            raise ValueError("Plan target_entities must be a list")
        return cls(
            item_id=data["item_id"],
            sub_query=data["sub_query"],
            target_entities=tuple(entities),
            rationale=data.get("rationale", ""),
            relevance_score=data.get("relevance_score", 1.0),
        )


@dataclass(frozen=True)
class QueryPlan:
    """A frozen plan and its generation receipt, without caller state."""

    query: str
    items: Tuple[QueryPlanItem, ...]
    thematic_framing: str = ""
    global_reports_used: Tuple[str, ...] = ()
    generation_method: str = "provided"
    query_truncated: bool = False
    framing_truncated: bool = False
    items_truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("Plan query must be nonempty")
        if not isinstance(self.items, (list, tuple)):
            raise ValueError("Plan items must be a list or tuple")
        items = tuple(self.items)
        if not items or not all(isinstance(item, QueryPlanItem) for item in items):
            raise ValueError("Plan must contain QueryPlanItem objects")
        if len({item.item_id for item in items}) != len(items):
            raise ValueError("Plan item IDs must be unique")
        if not isinstance(self.thematic_framing, str):
            raise ValueError("Plan thematic_framing must be a string")
        if self.generation_method not in ("model", "extractive_fallback", "provided"):
            raise ValueError("Unknown plan generation_method")
        if not all(
            isinstance(value, bool)
            for value in (
                self.query_truncated,
                self.framing_truncated,
                self.items_truncated,
            )
        ):
            raise ValueError("Plan truncation flags must be boolean")
        if not isinstance(self.global_reports_used, (list, tuple)) or not all(
            isinstance(value, str) for value in self.global_reports_used
        ):
            raise ValueError("Plan report IDs must be strings")
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "global_reports_used", tuple(self.global_reports_used))

    def select(
        self, item_ids: Optional[Sequence[str]] = None
    ) -> Tuple[QueryPlanItem, ...]:
        if item_ids is None:
            return self.items
        if not item_ids or len(set(item_ids)) != len(item_ids):
            raise ValueError("Plan selection must be nonempty and unique")
        indexed = {item.item_id: item for item in self.items}
        if any(item_id not in indexed for item_id in item_ids):
            raise ValueError("Unknown plan item ID")
        return tuple(indexed[item_id] for item_id in item_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "query-plan-v1",
            "query": self.query,
            "items": [item.to_dict() for item in self.items],
            "thematic_framing": self.thematic_framing,
            "global_reports_used": list(self.global_reports_used),
            "generation_method": self.generation_method,
            "query_truncated": self.query_truncated,
            "framing_truncated": self.framing_truncated,
            "items_truncated": self.items_truncated,
        }

    @property
    def content_hash(self) -> str:
        """Opaque native identity, stable after JSON/JavaScript round trips.

        Callers should carry this value rather than reimplement Python JSON
        encoding as a cross-language canonicalization standard.
        """
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryPlan":
        if not isinstance(data, dict) or set(data) - {
            "schema_version",
            "query",
            "items",
            "thematic_framing",
            "global_reports_used",
            "generation_method",
            "query_truncated",
            "framing_truncated",
            "items_truncated",
        }:
            raise ValueError("Invalid query plan fields")
        if not {"schema_version", "query", "items"}.issubset(data):
            raise ValueError("Missing query plan fields")
        if data.get("schema_version") != "query-plan-v1":
            raise ValueError("Unsupported query plan schema")
        if not isinstance(data.get("items"), list):
            raise ValueError("Plan items must be a list")
        reports = data.get("global_reports_used", [])
        if not isinstance(reports, list):
            raise ValueError("Plan reports must be a list")
        return cls(
            query=data["query"],
            items=tuple(QueryPlanItem.from_dict(item) for item in data["items"]),
            thematic_framing=data.get("thematic_framing", ""),
            global_reports_used=tuple(reports),
            generation_method=data.get("generation_method", "provided"),
            query_truncated=data.get("query_truncated", False),
            framing_truncated=data.get("framing_truncated", False),
            items_truncated=data.get("items_truncated", False),
        )
