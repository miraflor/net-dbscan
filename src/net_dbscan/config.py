"""Parameter objects and per-group overrides for net-dbscan."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .network import DEFAULT_MAX_NEIGHBOR_PAIRS
from .sparse_dbscan import BORDER_POLICIES, validate_parameters


@dataclass(frozen=True)
class DBSCANConfig:
    """DBSCAN parameters; distances use the projected network CRS units.

    ``eps`` is both the DBSCAN neighbourhood radius and the exact maximum
    network distance that must be searched. It is part of the clustering
    model, not merely a computational truncation horizon.
    """

    eps: float
    min_samples: int = 5
    noise_policy: str = "exclude"
    border_policy: str = "expansion"
    max_snap_distance: float | None = None
    max_neighbor_pairs: int = DEFAULT_MAX_NEIGHBOR_PAIRS

    def __post_init__(self) -> None:
        validate_parameters(eps=self.eps, min_samples=self.min_samples, border_policy=self.border_policy)
        object.__setattr__(self, "eps", float(self.eps))
        object.__setattr__(self, "min_samples", int(self.min_samples))
        if self.noise_policy not in {"exclude", "singleton"}:
            raise ValueError("noise_policy must be 'exclude' or 'singleton'")
        if self.max_snap_distance is not None:
            value = float(self.max_snap_distance)
            if math.isnan(value) or value < 0:
                raise ValueError("max_snap_distance must be None or a number >= 0")
            object.__setattr__(self, "max_snap_distance", value)
        if (
            isinstance(self.max_neighbor_pairs, bool)
            or not isinstance(self.max_neighbor_pairs, (int, np.integer))
            or self.max_neighbor_pairs < 1
        ):
            raise ValueError("max_neighbor_pairs must be an integer >= 1")
        object.__setattr__(self, "max_neighbor_pairs", int(self.max_neighbor_pairs))

    def replace(self, **changes: Any) -> "DBSCANConfig":
        return dataclasses.replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# Kept as a source-compatibility alias for releases before 0.4.0.
NetDBSCANConfig = DBSCANConfig


# Internal keys cannot collide with ordinary group strings. User-facing tokens
# remain readable and backward compatible with the 0.3.0 file convention.
NULL_GROUP_TOKEN = "__null__"
BLANK_GROUP_TOKEN = "__blank__"
NULL_GROUP_KEY = "\0net_dbscan:null"
BLANK_GROUP_KEY = "\0net_dbscan:blank"

_INT_FIELDS = {"min_samples"}
_FLOAT_FIELDS = {"eps"}
_TEXT_FIELDS = {"noise_policy", "border_policy"}
GROUP_OVERRIDE_FIELDS = tuple(sorted(_INT_FIELDS | _FLOAT_FIELDS | _TEXT_FIELDS))


def group_key(value) -> str:
    """Internal key for a group value, keeping missing values distinct from text."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return NULL_GROUP_KEY
    text = str(value)
    return BLANK_GROUP_KEY if text.strip() == "" else text


def group_display(key: str) -> str:
    """Unambiguous user-facing text for an internal group key."""
    if key == NULL_GROUP_KEY:
        return NULL_GROUP_TOKEN
    if key == BLANK_GROUP_KEY:
        return BLANK_GROUP_TOKEN
    if key in {NULL_GROUP_TOKEN, BLANK_GROUP_TOKEN}:
        return f"literal:{key}"
    return str(key)


def _group_param_key(text: str) -> str:
    if text == NULL_GROUP_TOKEN:
        return NULL_GROUP_KEY
    if text == BLANK_GROUP_TOKEN:
        return BLANK_GROUP_KEY
    if text.startswith("literal:"):
        return text[len("literal:") :]
    return text


def _convert(field: str, text: str):
    if field in _INT_FIELDS:
        number = float(text)
        if not number.is_integer():
            raise ValueError(f"{field} must be an integer, got {text!r}")
        return int(number)
    if field in _FLOAT_FIELDS:
        return float(text)
    return text.strip()


def read_group_params(path: str | Path, group_col: str) -> dict[str, dict[str, Any]]:
    """Read per-group DBSCAN overrides from CSV.

    The first column is ``group_col`` or ``group``. Supported overrides are
    ``eps``, ``min_samples``, ``noise_policy`` and ``border_policy``. Empty
    cells inherit the run-wide configuration. Use ``__null__`` and
    ``__blank__`` for actual missing groups; prefix ``literal:`` to target an
    ordinary string with either reserved spelling.
    """
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if len(table.columns) == 0:
        raise ValueError("group parameter file has no columns")
    key_col = table.columns[0]
    if key_col not in {group_col, "group"}:
        raise ValueError(
            f"first column of the group parameter file must be {group_col!r} or 'group', got {key_col!r}"
        )
    unknown = sorted(set(table.columns[1:]) - set(GROUP_OVERRIDE_FIELDS))
    if unknown:
        raise ValueError(
            f"unknown columns in group parameter file: {unknown}; allowed: {list(GROUP_OVERRIDE_FIELDS)}"
        )

    out: dict[str, dict[str, Any]] = {}
    semantic_keys: set[str] = set()
    for row_index, (_, row) in enumerate(table.iterrows(), start=2):
        raw_key = str(row[key_col])
        if raw_key == "":
            raise ValueError("empty group value in parameter file; write __blank__ for empty text")
        key = _group_param_key(raw_key)
        if key in semantic_keys:
            raise ValueError(f"group parameter file lists group {group_display(key)!r} more than once")
        semantic_keys.add(key)
        changes: dict[str, Any] = {}
        for field in table.columns[1:]:
            text = str(row[field]).strip()
            if text == "":
                continue
            try:
                changes[field] = _convert(field, text)
            except Exception as exc:
                raise ValueError(f"invalid {field!r} at row {row_index}: {text!r}") from exc
        # Preserve the public spelling; the grouped pipeline normalizes it.
        out[raw_key] = changes
    return out


def normalize_group_params(overrides: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Normalize public override keys to the pipeline's internal group keys."""
    if not overrides:
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, values in overrides.items():
        key = raw_key if raw_key in {NULL_GROUP_KEY, BLANK_GROUP_KEY} else _group_param_key(str(raw_key))
        if key in normalized:
            raise ValueError(f"group parameter overrides list group {group_display(key)!r} more than once")
        normalized[key] = dict(values)
    return normalized


def resolve_group_config(
    base: DBSCANConfig,
    overrides: Mapping[str, Mapping[str, Any]] | None,
    key: str,
) -> DBSCANConfig:
    """Return the run-wide config with this group's explicit overrides."""
    if not overrides or key not in overrides:
        return base
    changes = dict(overrides[key])
    unknown = sorted(set(changes) - set(GROUP_OVERRIDE_FIELDS))
    if unknown:
        raise ValueError(f"group {group_display(key)!r}: unknown override fields {unknown}")
    try:
        return base.replace(**changes)
    except ValueError as exc:
        raise ValueError(f"group {group_display(key)!r}: {exc}") from exc


def read_group_universe(path: str | Path, group_col: str) -> list[str]:
    """Read a one-column CSV of expected non-missing groups, preserving order."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if len(table.columns) != 1:
        raise ValueError("group universe must be a one-column CSV")
    column = table.columns[0]
    if column not in {group_col, "group"}:
        raise ValueError(f"group universe column must be {group_col!r} or 'group', got {column!r}")
    values = [group_key(v) for v in table[column].tolist()]
    if any(v in {NULL_GROUP_KEY, BLANK_GROUP_KEY} for v in values):
        raise ValueError("group universe cannot contain null/blank values")
    if len(values) != len(set(values)):
        raise ValueError("group universe lists a group more than once")
    return values


__all__ = [
    "BLANK_GROUP_KEY",
    "DBSCANConfig",
    "GROUP_OVERRIDE_FIELDS",
    "NetDBSCANConfig",
    "NULL_GROUP_KEY",
    "group_display",
    "group_key",
    "normalize_group_params",
    "read_group_params",
    "read_group_universe",
    "resolve_group_config",
]
