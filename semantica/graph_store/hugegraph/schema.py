"""Strong-schema and index operations for HugeGraph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

from ...utils.exceptions import ProcessingError, ValidationError
from .utils import (
    _call,
    _chain,
    _identifier,
    _plain,
    _schema_specs,
    _validate_existing_schema,
)


class HugeGraphSchemaMixin:
    """Implement explicit HugeGraph schema and index operations."""

    def get_schema(self) -> Dict[str, Any]:
        try:
            value = _call(
                self._manager("schema"),
                ("get_schema", "getSchema"),
            )
        except Exception as exc:
            raise ProcessingError(f"Failed to read HugeGraph schema: {exc}") from exc
        plain = _plain(value)
        return dict(plain) if isinstance(plain, dict) else {"schema": plain}

    def apply_schema(
        self,
        schema: Dict[str, Any],
        **options: Any,
    ) -> Dict[str, int]:
        """Create an explicit HugeGraph schema without weakening its contract.

        The accepted shape mirrors HugeGraph REST schema objects and supports
        ``propertykeys``, ``vertexlabels``, ``edgelabels`` and ``indexlabels``.
        Existing definitions are retained only when their material fields are
        compatible with the requested definition.
        """
        self._ensure_write("apply_schema")
        if not isinstance(schema, Mapping):
            raise ValidationError("HugeGraph schema must be an object")
        if options.get("validate_existing", True):
            _validate_existing_schema(self.get_schema(), schema)

        manager = self._manager("schema")
        counts = {
            "propertykeys": 0,
            "vertexlabels": 0,
            "edgelabels": 0,
            "indexlabels": 0,
        }
        type_methods = {
            "OBJECT": ("as_object", "asObject"),
            "BOOLEAN": ("as_bool", "asBool"),
            "BYTE": ("as_byte", "asByte"),
            "INT": ("as_int", "asInt"),
            "LONG": ("as_long", "asLong"),
            "FLOAT": ("as_float", "asFloat"),
            "DOUBLE": ("as_double", "asDouble"),
            "TEXT": ("as_text", "asText"),
            "BLOB": ("as_blob", "asBlob"),
            "DATE": ("as_date", "asDate"),
        }
        cardinality_methods = {
            "SINGLE": ("value_single", "valueSingle"),
            "LIST": ("value_list", "valueList"),
            "SET": ("value_set", "valueSet"),
        }
        id_methods = {
            "AUTOMATIC": ("use_automatic_id", "useAutomaticId"),
            "CUSTOMIZE_STRING": (
                "use_customize_string_id",
                "useCustomizeStringId",
            ),
            "CUSTOMIZE_NUMBER": (
                "use_customize_number_id",
                "useCustomizeNumberId",
            ),
            "PRIMARY_KEY": ("use_primary_key_id", "usePrimaryKeyId"),
        }

        try:
            for spec in _schema_specs(schema, "propertykeys"):
                name = _identifier(str(spec["name"]), "property key")
                builder = _call(
                    manager,
                    ("property_key", "propertyKey"),
                    name,
                )
                data_type = str(spec.get("data_type", "TEXT")).upper()
                cardinality = str(spec.get("cardinality", "SINGLE")).upper()
                if (
                    data_type not in type_methods
                    or cardinality not in cardinality_methods
                ):
                    raise ValidationError(
                        "Unsupported property key definition for "
                        f"{name}: {data_type}/{cardinality}"
                    )
                builder = _chain(builder, type_methods[data_type])
                builder = _chain(builder, cardinality_methods[cardinality])
                builder = _chain(builder, ("if_not_exist", "ifNotExist"))
                _call(builder, ("create",))
                counts["propertykeys"] += 1

            for spec in _schema_specs(schema, "vertexlabels"):
                name = _identifier(str(spec["name"]), "vertex label")
                builder = _call(
                    manager,
                    ("vertex_label", "vertexLabel"),
                    name,
                )
                strategy = str(spec.get("id_strategy", "AUTOMATIC")).upper()
                if strategy not in id_methods:
                    raise ValidationError(
                        f"Unsupported id_strategy for {name}: {strategy}"
                    )
                builder = _chain(builder, id_methods[strategy])
                properties = [str(item) for item in spec.get("properties", []) or []]
                nullable = [str(item) for item in spec.get("nullable_keys", []) or []]
                primary = [str(item) for item in spec.get("primary_keys", []) or []]
                if properties:
                    builder = _chain(builder, ("properties",), *properties)
                if nullable:
                    builder = _chain(
                        builder,
                        ("nullable_keys", "nullableKeys"),
                        *nullable,
                    )
                if primary:
                    builder = _chain(
                        builder,
                        ("primary_keys", "primaryKeys"),
                        *primary,
                    )
                builder = _chain(
                    builder,
                    ("enable_label_index", "enableLabelIndex"),
                    bool(spec.get("enable_label_index", True)),
                )
                builder = _chain(builder, ("if_not_exist", "ifNotExist"))
                _call(builder, ("create",))
                counts["vertexlabels"] += 1

            for spec in _schema_specs(schema, "edgelabels"):
                name = _identifier(str(spec["name"]), "edge label")
                source = _identifier(
                    str(spec["source_label"]),
                    "source vertex label",
                )
                target = _identifier(
                    str(spec["target_label"]),
                    "target vertex label",
                )
                builder = _call(
                    manager,
                    ("edge_label", "edgeLabel"),
                    name,
                )
                builder = _chain(builder, ("link",), source, target)
                properties = [str(item) for item in spec.get("properties", []) or []]
                nullable = [str(item) for item in spec.get("nullable_keys", []) or []]
                sort_keys = [str(item) for item in spec.get("sort_keys", []) or []]
                if properties:
                    builder = _chain(builder, ("properties",), *properties)
                if nullable:
                    builder = _chain(
                        builder,
                        ("nullable_keys", "nullableKeys"),
                        *nullable,
                    )
                if sort_keys:
                    builder = _chain(
                        builder,
                        ("sort_keys", "sortKeys"),
                        *sort_keys,
                    )
                frequency = str(spec.get("frequency", "SINGLE")).upper()
                frequency_methods = (
                    ("multi_times", "multiTimes")
                    if frequency == "MULTIPLE"
                    else ("single_time", "singleTime")
                )
                builder = _chain(builder, frequency_methods)
                builder = _chain(
                    builder,
                    ("enable_label_index", "enableLabelIndex"),
                    bool(spec.get("enable_label_index", True)),
                )
                builder = _chain(builder, ("if_not_exist", "ifNotExist"))
                _call(builder, ("create",))
                counts["edgelabels"] += 1

            for spec in _schema_specs(schema, "indexlabels"):
                fields = list(spec.get("fields") or [])
                first_field = str(fields[0]) if fields else ""
                self.create_index(
                    str(spec["base_value"]),
                    first_field,
                    str(spec.get("index_type", "SECONDARY")),
                    index_name=str(spec["name"]),
                    base_type=str(spec.get("base_type", "VERTEX_LABEL")),
                    fields=fields,
                )
                counts["indexlabels"] += 1
        except (ValidationError, ProcessingError):
            raise
        except Exception as exc:
            raise ProcessingError(f"Failed to apply HugeGraph schema: {exc}") from exc
        return counts

    def create_index(
        self,
        label: str,
        property_name: str,
        index_type: str = "btree",
        **options: Any,
    ) -> bool:
        self._ensure_write("create_index")
        label = _identifier(label, "label")
        fields = [str(item) for item in options.get("fields", []) or [property_name]]
        if not fields or any(not item or item == "None" for item in fields):
            raise ValidationError(
                "HugeGraph index requires at least one property field"
            )
        for field in fields:
            _identifier(field, "property key")

        default_name = f"idx_{label}_{'_'.join(fields)}"
        index_name = _identifier(
            str(options.get("index_name", default_name)),
            "index label",
        )
        normalized_type = str(index_type).upper()
        type_methods = {
            "BTREE": ("secondary",),
            "SECONDARY": ("secondary",),
            "FULLTEXT": ("search",),
            "SEARCH": ("search",),
            "RANGE": ("range",),
            "SHARD": ("shard",),
            "UNIQUE": ("unique",),
        }
        if normalized_type not in type_methods:
            raise ValidationError(f"Unsupported HugeGraph index type: {index_type}")

        try:
            builder = _call(
                self._manager("schema"),
                ("index_label", "indexLabel"),
                index_name,
            )
            base_type = str(options.get("base_type", "VERTEX_LABEL")).upper()
            if base_type == "EDGE_LABEL":
                builder = _chain(builder, ("on_e", "onE"), label)
            else:
                builder = _chain(builder, ("on_v", "onV"), label)
            builder = _chain(builder, ("by",), *fields)
            builder = _chain(builder, type_methods[normalized_type])
            builder = _chain(builder, ("if_not_exist", "ifNotExist"))
            _call(builder, ("create",))
            return True
        except (ValidationError, ProcessingError):
            raise
        except Exception as exc:
            raise ProcessingError(
                f"Failed to create HugeGraph index {index_name}: {exc}"
            ) from exc

