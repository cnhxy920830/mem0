import json
import logging
import uuid
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import UUID

import pyarrow as pa
from pydantic import BaseModel

try:
    import lancedb
    from lancedb.rerankers import RRFReranker
except ImportError as exc:
    raise ImportError("LanceDB vector store requires lancedb. Please install it using 'pip install lancedb'.") from exc

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)

RESERVED_COLUMNS = {"id", "vector", "payload_json", "search_text"}
CORE_STRING_COLUMNS = (
    "data",
    "user_id",
    "agent_id",
    "run_id",
    "actor_id",
    "role",
    "hash",
    "created_at",
    "updated_at",
    "memory_type",
)
SEARCH_TEXT_COLUMN = "search_text"
VECTOR_COLUMN = "vector"
PAYLOAD_JSON_COLUMN = "payload_json"
VECTOR_INDEX_NAME = "vector_idx"
FTS_INDEX_NAME = "search_text_idx"
VECTOR_INDEX_MIN_ROWS = 256
SCALAR_INDEX_COLUMNS = (
    ("user_id", "user_id_idx"),
    ("agent_id", "agent_id_idx"),
    ("run_id", "run_id_idx"),
    ("actor_id", "actor_id_idx"),
    ("memory_type", "memory_type_idx"),
    ("role", "role_idx"),
)
SEARCH_TEXT_PRIORITY_FIELDS = (
    "data",
    "memory",
    "text",
    "tags",
    "experience_label",
    "topic",
    "category",
    "memory_type",
    "role",
    "actor_id",
)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    distance: Optional[float] = None
    source: Optional[str] = None
    payload: Optional[Dict[str, Any]]


class LanceDB(VectorStoreBase):
    def __init__(
        self,
        uri: str = "./lancedb",
        collection_name: str = "mem0",
        embedding_model_dims: int = 1536,
        table_name: Optional[str] = None,
        distance_metric: str = "cosine",
        storage_options: Optional[Dict[str, str]] = None,
    ):
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.table_name = table_name or collection_name
        self.distance_metric = distance_metric
        self.storage_options = storage_options

        connect_kwargs: Dict[str, Any] = {}
        if storage_options:
            connect_kwargs["storage_options"] = storage_options

        self.db = lancedb.connect(uri, **connect_kwargs)
        self.table = self.create_col(
            name=self.table_name,
            vector_size=self.embedding_model_dims,
            distance=self.distance_metric,
        )

    def create_col(self, name: Optional[str] = None, vector_size: Optional[int] = None, distance: str = "cosine"):
        table_name = name or self.table_name
        dims = vector_size or self.embedding_model_dims

        if table_name in self._list_tables():
            table = self.db.open_table(table_name)
            self._validate_vector_dimension(table, dims)
            self._validate_required_columns(table)
        else:
            table = self.db.create_table(table_name, schema=self._build_schema(dims))

        if table_name == self.table_name:
            self.table = table
            self.embedding_model_dims = dims
            self.distance_metric = distance

        return table

    def insert(self, vectors: List[list], payloads: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        if payloads is None:
            payloads = [{} for _ in range(len(vectors))]
        if len(vectors) != len(payloads) or len(vectors) != len(ids):
            raise ValueError("Vectors, payloads, and ids must have the same length.")

        self._ensure_payload_columns(payloads)
        had_rows = self.table.count_rows() > 0
        rows = [
            self._build_row(vector_id=vector_id, vector=vector, payload=payload)
            for vector_id, vector, payload in zip(ids, vectors, payloads)
        ]
        self.table.add(rows)
        self._ensure_indexes(rebuild_fts=not had_rows)

    def search(
        self, query: str, vectors: List[float], limit: int = 5, filters: Optional[Dict] = None
    ) -> List[OutputData]:
        return self.search_semantic(query=query, vectors=vectors, limit=limit, filters=filters)

    def search_semantic(
        self,
        query: str,
        vectors: List[float],
        limit: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        if limit <= 0 or self.table.count_rows() <= 0:
            return []

        query_vector = self._normalize_query_vector(vectors)
        query_builder = self.table.search(query_vector).distance_type(self.distance_metric).limit(limit)

        if filters:
            filter_expression = self._build_filter_expression(filters)
            query_builder = query_builder.where(filter_expression, prefilter=True)

        return [self._row_to_output(row, include_score=True, source="semantic") for row in query_builder.to_list()]

    def search_keyword(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        normalized_query = str(query or "").strip()
        if limit <= 0 or not normalized_query or self.table.count_rows() <= 0:
            return []
        self._ensure_fts_index(replace=False)
        query_builder = self.table.search(
            normalized_query,
            query_type="fts",
            fts_columns=SEARCH_TEXT_COLUMN,
        ).limit(limit)
        if filters:
            filter_expression = self._build_filter_expression(filters)
            query_builder = query_builder.where(filter_expression, prefilter=True)
        return [self._row_to_output(row, include_score=True, source="keyword") for row in query_builder.to_list()]

    def search_hybrid(
        self,
        query: str,
        vectors: List[float],
        *,
        limit: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        normalized_query = str(query or "").strip()
        if limit <= 0 or self.table.count_rows() <= 0:
            return []
        if not normalized_query:
            return self.search_semantic(query=query, vectors=vectors, limit=limit, filters=filters)

        self._ensure_fts_index(replace=False)
        query_vector = self._normalize_query_vector(vectors)
        query_builder = (
            self.table.search(
                query_type="hybrid",
                vector_column_name=VECTOR_COLUMN,
                fts_columns=SEARCH_TEXT_COLUMN,
            )
            .vector(query_vector)
            .text(normalized_query)
            .distance_type(self.distance_metric)
            .rerank(RRFReranker(return_score="all"))
            .limit(limit)
        )
        if filters:
            filter_expression = self._build_filter_expression(filters)
            query_builder = query_builder.where(filter_expression, prefilter=True)
        return [self._row_to_output(row, include_score=True, source="hybrid") for row in query_builder.to_list()]

    def search_hybrid_candidates(
        self,
        *,
        query: str,
        vectors: List[float],
        semantic_limit: int,
        keyword_limit: int,
        filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        return self.search_hybrid(
            query=query,
            vectors=vectors,
            limit=max(semantic_limit, keyword_limit),
            filters=filters,
        )

    def delete(self, vector_id: str):
        self.table.delete(self._eq_expression("id", vector_id))

    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict] = None):
        existing = self.get(vector_id)
        if existing is None:
            raise ValueError(f"Vector {vector_id} not found in collection {self.table_name}.")

        next_payload = payload.copy() if payload is not None else existing.payload.copy()
        self._ensure_payload_columns([next_payload])

        update_values: Dict[str, Any] = {
            PAYLOAD_JSON_COLUMN: self._serialize_payload(next_payload),
            SEARCH_TEXT_COLUMN: self._build_search_text(next_payload),
        }
        if vector is not None:
            update_values[VECTOR_COLUMN] = self._normalize_stored_vector(vector)

        for column_name in self._metadata_column_names():
            if column_name in next_payload:
                normalized_value = self._normalize_value_for_field(
                    next_payload[column_name],
                    self.table.schema.field(column_name).type,
                )
                update_values[column_name] = normalized_value
            else:
                update_values[column_name] = None

        self.table.update(where=self._eq_expression("id", vector_id), values=update_values)

    def get(self, vector_id: str) -> Optional[OutputData]:
        rows = self.table.search().where(self._eq_expression("id", vector_id)).limit(1).to_list()
        if not rows:
            return None
        return self._row_to_output(rows[0], include_score=False)

    def list_cols(self) -> List[str]:
        return self._list_tables()

    def delete_col(self):
        if self.table_name in self._list_tables():
            self.db.drop_table(self.table_name)
        self.table = None

    def col_info(self) -> Dict[str, Any]:
        return {
            "name": self.table_name,
            "count": self.table.count_rows(),
            "dimension": self.embedding_model_dims,
            "distance_metric": self.distance_metric,
            "uri": self.uri,
            "columns": self.table.schema.names,
        }

    def list(self, filters: Optional[Dict] = None, limit: int = 100) -> List[List[OutputData]]:
        if limit <= 0:
            return [[]]

        query_builder = self.table.search().limit(limit)
        if filters:
            query_builder = query_builder.where(self._build_filter_expression(filters))
        return [[self._row_to_output(row, include_score=False) for row in query_builder.to_list()]]

    def count(self, filters: Optional[Dict] = None) -> int:
        if filters:
            return int(self.table.count_rows(filter=self._build_filter_expression(filters)))
        return int(self.table.count_rows())

    def delete_many(self, filters: Dict[str, Any]) -> int:
        filter_expression = self._build_filter_expression(filters)
        deleted_count = int(self.table.count_rows(filter=filter_expression))
        if deleted_count <= 0:
            return 0
        delete_result = self.table.delete(filter_expression)
        reported_deleted = getattr(delete_result, "num_deleted_rows", None)
        if reported_deleted is not None:
            deleted_count = int(reported_deleted)
        if deleted_count > 0 and self.table.count_rows() > 0:
            self.optimize()
        return deleted_count

    def optimize(self) -> None:
        self.table.optimize()

    def reset(self):
        logger.warning("Resetting collection %s", self.table_name)
        self.delete_col()
        self.table = self.create_col(self.table_name, self.embedding_model_dims, self.distance_metric)

    def _build_schema(self, dims: int) -> pa.Schema:
        fields = [
            pa.field("id", pa.string(), nullable=False),
            pa.field(VECTOR_COLUMN, pa.list_(pa.float32(), dims), nullable=False),
            pa.field(PAYLOAD_JSON_COLUMN, pa.string(), nullable=False),
            pa.field(SEARCH_TEXT_COLUMN, pa.string(), nullable=False),
        ]
        fields.extend(pa.field(column_name, pa.string()) for column_name in CORE_STRING_COLUMNS)
        return pa.schema(fields)

    def _validate_vector_dimension(self, table, expected_dims: int):
        vector_field = table.schema.field(VECTOR_COLUMN)
        vector_type = vector_field.type
        if not pa.types.is_fixed_size_list(vector_type):
            raise ValueError(f"Table {self.table_name} does not use a fixed-size vector column.")

        actual_dims = vector_type.list_size
        if actual_dims != expected_dims:
            raise ValueError(
                f"Embedding dimension mismatch for LanceDB table {self.table_name}: "
                f"expected {expected_dims}, found {actual_dims}."
            )

    def _validate_required_columns(self, table) -> None:
        required_columns = {PAYLOAD_JSON_COLUMN, SEARCH_TEXT_COLUMN}
        missing = required_columns.difference(set(table.schema.names))
        if missing:
            raise ValueError(
                f"LanceDB table {self.table_name} is missing required columns: {', '.join(sorted(missing))}."
            )

    def _ensure_payload_columns(self, payloads: Iterable[Optional[Dict[str, Any]]]):
        existing_columns = set(self.table.schema.names)
        column_kinds: Dict[str, str] = {}

        for payload in payloads:
            if not payload:
                continue
            for key, value in payload.items():
                if key in RESERVED_COLUMNS or key.startswith("_") or key in existing_columns:
                    continue
                kind = self._infer_column_kind(value)
                if kind is None:
                    continue
                if key in column_kinds:
                    column_kinds[key] = self._merge_column_kind(column_kinds[key], kind)
                else:
                    column_kinds[key] = kind

        if not column_kinds:
            return

        self.table.add_columns(
            [pa.field(column_name, self._arrow_type_for_kind(kind)) for column_name, kind in column_kinds.items()]
        )
        self.table = self.db.open_table(self.table_name)

    def _build_row(self, vector_id: str, vector: List[float], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload_dict = payload.copy() if payload else {}
        row: Dict[str, Any] = {
            "id": vector_id,
            VECTOR_COLUMN: self._normalize_stored_vector(vector),
            PAYLOAD_JSON_COLUMN: self._serialize_payload(payload_dict),
            SEARCH_TEXT_COLUMN: self._build_search_text(payload_dict),
        }

        for column_name in self._metadata_column_names():
            if column_name in payload_dict:
                normalized_value = self._normalize_value_for_field(
                    payload_dict[column_name],
                    self.table.schema.field(column_name).type,
                )
                row[column_name] = normalized_value
            else:
                row[column_name] = None

        return row

    def _metadata_column_names(self) -> List[str]:
        return [column_name for column_name in self.table.schema.names if column_name not in RESERVED_COLUMNS]

    def _normalize_stored_vector(self, vector: List[float]) -> List[float]:
        normalized_vector = [float(item) for item in vector]
        if len(normalized_vector) != self.embedding_model_dims:
            raise ValueError(
                f"Vector dimension mismatch for LanceDB table {self.table_name}: "
                f"expected {self.embedding_model_dims}, received {len(normalized_vector)}."
            )
        return normalized_vector

    def _normalize_query_vector(self, vector: List[float]) -> List[float]:
        if vector and isinstance(vector[0], (list, tuple)):
            return self._normalize_stored_vector(vector[0])
        return self._normalize_stored_vector(vector)

    def _row_to_output(self, row: Dict[str, Any], include_score: bool, source: Optional[str] = None) -> OutputData:
        payload = self._deserialize_payload(row.get(PAYLOAD_JSON_COLUMN))
        score = None
        distance = None
        if include_score and row.get("_relevance_score") is not None:
            score = float(row["_relevance_score"])
            if row.get("_distance") is not None:
                distance = float(row["_distance"])
        elif include_score and row.get("_distance") is not None:
            distance = float(row["_distance"])
            score = self._distance_to_score(distance)
        elif include_score and row.get("_score") is not None:
            score = float(row["_score"])
        return OutputData(id=row.get("id"), score=score, distance=distance, source=source, payload=payload)

    def _distance_to_score(self, distance: float) -> float:
        if self.distance_metric in {"cosine", "dot"}:
            return 1.0 - distance
        return 1.0 / (1.0 + distance)

    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        expression = self._build_nested_filter_expression(filters)
        return expression or "TRUE"

    def _build_nested_filter_expression(self, filters: Dict[str, Any]) -> str:
        clauses: List[str] = []

        for key, value in filters.items():
            if key in {"$or", "OR"}:
                or_clauses = [
                    self._build_nested_filter_expression(condition)
                    for condition in value
                    if isinstance(condition, dict)
                ]
                or_clauses = [clause for clause in or_clauses if clause]
                if or_clauses:
                    clauses.append("(" + " OR ".join(f"({clause})" for clause in or_clauses) + ")")
            elif key in {"$and", "AND"}:
                and_clauses = [
                    self._build_nested_filter_expression(condition)
                    for condition in value
                    if isinstance(condition, dict)
                ]
                and_clauses = [clause for clause in and_clauses if clause]
                if and_clauses:
                    clauses.append("(" + " AND ".join(f"({clause})" for clause in and_clauses) + ")")
            elif key in {"$not", "NOT"}:
                not_clauses = [
                    self._build_nested_filter_expression(condition)
                    for condition in value
                    if isinstance(condition, dict)
                ]
                not_clauses = [clause for clause in not_clauses if clause]
                if not_clauses:
                    clauses.extend(f"NOT ({clause})" for clause in not_clauses)
            else:
                clause = self._build_field_filter_expression(key, value)
                if clause:
                    clauses.append(clause)

        return " AND ".join(f"({clause})" for clause in clauses)

    def _build_field_filter_expression(self, key: str, value: Any) -> str:
        if key not in self.table.schema.names:
            return "FALSE"

        column_sql = self._sql_identifier(key)
        field_type = self.table.schema.field(key).type

        if value == "*":
            return f"{column_sql} IS NOT NULL"

        if not isinstance(value, dict):
            return self._comparison_expression(column_sql, "eq", value, field_type)

        expressions = [
            self._comparison_expression(column_sql, operator, operand, field_type)
            for operator, operand in value.items()
        ]
        expressions = [expression for expression in expressions if expression]
        return " AND ".join(f"({expression})" for expression in expressions)

    def _comparison_expression(self, column_sql: str, operator: str, operand: Any, field_type: pa.DataType) -> str:
        if operator == "eq":
            if operand is None:
                return f"{column_sql} IS NULL"
            return f"{column_sql} IS NOT NULL AND {column_sql} = {self._sql_literal(operand)}"
        if operator == "ne":
            if operand is None:
                return f"{column_sql} IS NOT NULL"
            return f"{column_sql} IS NOT NULL AND {column_sql} != {self._sql_literal(operand)}"
        if operator == "gt":
            return f"{column_sql} IS NOT NULL AND {column_sql} > {self._sql_literal(operand)}"
        if operator == "gte":
            return f"{column_sql} IS NOT NULL AND {column_sql} >= {self._sql_literal(operand)}"
        if operator == "lt":
            return f"{column_sql} IS NOT NULL AND {column_sql} < {self._sql_literal(operand)}"
        if operator == "lte":
            return f"{column_sql} IS NOT NULL AND {column_sql} <= {self._sql_literal(operand)}"
        if operator == "in":
            values = [self._sql_literal(item) for item in operand]
            if not values:
                return "FALSE"
            return f"{column_sql} IS NOT NULL AND {column_sql} IN ({', '.join(values)})"
        if operator == "nin":
            values = [self._sql_literal(item) for item in operand]
            if not values:
                return "TRUE"
            return f"{column_sql} IS NOT NULL AND {column_sql} NOT IN ({', '.join(values)})"
        if operator == "contains":
            if not pa.types.is_string(field_type):
                return "FALSE"
            return f"{column_sql} IS NOT NULL AND {column_sql} LIKE {self._sql_literal(f'%{operand}%')}"
        if operator == "icontains":
            if not pa.types.is_string(field_type):
                return "FALSE"
            return (
                f"{column_sql} IS NOT NULL AND "
                f"LOWER({column_sql}) LIKE {self._sql_literal(f'%{str(operand).lower()}%')}"
            )
        raise ValueError(f"Unsupported LanceDB filter operator: {operator}")

    def _eq_expression(self, column_name: str, value: Any) -> str:
        return self._comparison_expression(self._sql_identifier(column_name), "eq", value, pa.string())

    def _in_expression(self, column_name: str, values: Sequence[Any]) -> str:
        return self._comparison_expression(self._sql_identifier(column_name), "in", list(values), pa.string())

    def _sql_identifier(self, column_name: str) -> str:
        escaped = column_name.replace("`", "``")
        return f"`{escaped}`"

    def _list_tables(self) -> List[str]:
        if hasattr(self.db, "list_tables"):
            tables = self.db.list_tables()
            if hasattr(tables, "tables"):
                return list(tables.tables)
            if isinstance(tables, dict):
                return list(tables.get("tables", []))
            return list(tables)
        return list(self.db.table_names())

    def _sql_literal(self, value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return "'" + str(self._json_safe_scalar(value)).replace("'", "''") + "'"

    def _serialize_payload(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=self._json_default)

    def _deserialize_payload(self, payload_json: Optional[str]) -> Dict[str, Any]:
        if not payload_json:
            return {}
        return json.loads(payload_json)

    def _infer_column_kind(self, value: Any) -> Optional[str]:
        value = self._json_safe_scalar(value)
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "string"
        return None

    def _merge_column_kind(self, current_kind: str, next_kind: str) -> str:
        if current_kind == next_kind:
            return current_kind
        if {current_kind, next_kind} == {"int", "float"}:
            return "float"
        return "string"

    def _arrow_type_for_kind(self, kind: str) -> pa.DataType:
        if kind == "bool":
            return pa.bool_()
        if kind == "int":
            return pa.int64()
        if kind == "float":
            return pa.float64()
        return pa.string()

    def _normalize_value_for_field(self, value: Any, field_type: pa.DataType) -> Any:
        if value is None:
            return None

        value = self._json_safe_scalar(value)

        if pa.types.is_boolean(field_type):
            return value if isinstance(value, bool) else None
        if pa.types.is_integer(field_type):
            return value if isinstance(value, int) and not isinstance(value, bool) else None
        if pa.types.is_floating(field_type):
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return float(value)
            return None
        if pa.types.is_string(field_type):
            return str(value)

        return None

    def _json_safe_scalar(self, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, (UUID, Path, Enum)):
            return str(value)
        return value

    def _json_default(self, value: Any) -> str:
        return str(self._json_safe_scalar(value))

    def _build_search_text(self, payload: Dict[str, Any]) -> str:
        parts: List[str] = []
        seen: set[str] = set()

        def add_value(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    add_value(item)
                return
            normalized = str(self._json_safe_scalar(value) or "").strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            parts.append(normalized)

        for field_name in SEARCH_TEXT_PRIORITY_FIELDS:
            add_value(payload.get(field_name))
        for key, value in payload.items():
            if key in SEARCH_TEXT_PRIORITY_FIELDS:
                continue
            if key in {"user_id", "agent_id", "run_id", "created_at", "updated_at", "hash"}:
                continue
            add_value(value)
        return "\n".join(parts)

    def _ensure_indexes(self, *, rebuild_fts: bool = False) -> None:
        self._ensure_scalar_indexes()
        self._ensure_fts_index(replace=rebuild_fts)
        self._ensure_vector_index()

    def _ensure_scalar_indexes(self) -> None:
        if self.table.count_rows() <= 0:
            return
        for column_name, index_name in SCALAR_INDEX_COLUMNS:
            if column_name not in self.table.schema.names:
                continue
            if self._has_index(index_name):
                continue
            self.table.create_scalar_index(
                column_name,
                replace=False,
                name=index_name,
            )

    def _ensure_fts_index(self, *, replace: bool = False) -> None:
        if self.table.count_rows() <= 0:
            return
        if self._has_index(FTS_INDEX_NAME) and not replace:
            return
        self.table.create_fts_index(
            SEARCH_TEXT_COLUMN,
            replace=replace,
            name=FTS_INDEX_NAME,
            base_tokenizer="simple",
            lower_case=True,
            with_position=True,
            stem=True,
            remove_stop_words=False,
            ascii_folding=True,
        )

    def _ensure_vector_index(self) -> None:
        if self.table.count_rows() < VECTOR_INDEX_MIN_ROWS:
            return
        if self._has_index(VECTOR_INDEX_NAME):
            return
        self.table.create_index(
            metric=self.distance_metric,
            vector_column_name=VECTOR_COLUMN,
            replace=False,
            name=VECTOR_INDEX_NAME,
        )

    def _has_index(self, index_name: str) -> bool:
        return any(
            str(getattr(index, "name", "") or "").strip() == index_name
            for index in list(self.table.list_indices())
        )
