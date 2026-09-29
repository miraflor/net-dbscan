"""End-to-end network-space DBSCAN: points -> labels and diagnostics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import geopandas as gpd
import numpy as np
import pandas as pd

from ._version import __version__
from .config import (
    BLANK_GROUP_KEY,
    NULL_GROUP_KEY,
    DBSCANConfig,
    group_display,
    group_key,
    normalize_group_params,
    resolve_group_config,
)
from .io import (
    _require_pyarrow,
    canonical_order,
    prepare_network,
    prepare_points,
    read_vector,
    write_csv,
    write_geoparquet,
    write_json,
    write_parquet,
)
from .network import (
    DEFAULT_VERTEX_DIGITS,
    NeighborPairLimitError,
    NetworkGraph,
    Snaps,
    build_network_graph,
    distinct_positions,
    empty_snaps,
    neighbor_graph,
    snap_points,
)
from .sparse_dbscan import dbscan

SUMMARY_COLUMNS = [
    "group",
    "n_points_input",
    "n_points",
    "n_positions",
    "max_position_weight",
    "stacked_positions",
    "n_pairs",
    "network_components_used",
    "share_on_largest_component",
    "snap_distance_median",
    "snap_distance_p95",
    "snap_distance_max",
    "n_clusters",
    "noise_share",
    "core_share",
    "border_share",
    "largest_cluster_share",
    "neighborhood_weight_median",
    "neighborhood_weight_p95",
    "contested_share",
    "eps",
    "min_samples",
    "border_policy",
    "declared_in_universe",
    "missing_group_policy",
    "n_missing_group_input",
    "seconds",
]

CLUSTER_COLUMNS = ["cluster_id", "n_points", "n_positions", "n_core", "n_border"]


@dataclass
class ClusterOutputs:
    """Results for one DBSCAN unit (a whole run or one group)."""

    points: gpd.GeoDataFrame
    clusters: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)


def _cluster_ids(n_obs: int, position: np.ndarray, pos_label: np.ndarray, singleton_noise: bool) -> np.ndarray:
    """Stable public cluster IDs numbered by each unit's first canonical observation."""
    obs_label = pos_label[position]
    unit = np.where(obs_label >= 0, obs_label, -1).astype(np.int64)
    if singleton_noise:
        offset = int(pos_label.max()) + 1 if len(pos_label) and pos_label.max() >= 0 else 0
        noise = obs_label < 0
        unit[noise] = offset + position[noise]
    ids = np.full(n_obs, None, dtype=object)
    valid = unit >= 0
    if not valid.any():
        return ids
    units = unit[valid]
    unique_units, first = np.unique(units, return_index=True)
    rank = np.empty(len(unique_units), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(unique_units))
    names = np.array([f"C{k + 1:06d}" for k in range(len(unique_units))], dtype=object)
    ids[valid] = names[rank[np.searchsorted(unique_units, units)]]
    return ids


def _empty_clusters() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": pd.Series(dtype=object),
            "n_points": pd.Series(dtype="int64"),
            "n_positions": pd.Series(dtype="int64"),
            "n_core": pd.Series(dtype="int64"),
            "n_border": pd.Series(dtype="int64"),
        }
    )


def _empty_points(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = points.copy()
    out["cluster_id"] = pd.Series(dtype=object)
    out["is_noise"] = pd.Series(dtype=bool)
    out["is_core"] = pd.Series(dtype=bool)
    out["is_border"] = pd.Series(dtype=bool)
    out["n_candidate_clusters"] = pd.Series(dtype="int64")
    out["snap_distance"] = pd.Series(dtype=float)
    out["snapped_x"] = pd.Series(dtype=float)
    out["snapped_y"] = pd.Series(dtype=float)
    return out


def _cluster_unit(
    points: gpd.GeoDataFrame,
    snaps: Snaps,
    graph: NetworkGraph | None,
    config: DBSCANConfig,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]:
    """Cluster points in canonical order."""
    n = len(points)
    stats: dict[str, Any] = {"n_points": int(n)}
    if n == 0:
        return _empty_points(points), _empty_clusters(), stats
    if graph is None:
        raise RuntimeError("non-empty clustering unit has no network graph")

    position, representative = distinct_positions(snaps)
    n_pos = len(representative)
    weights = np.bincount(position, minlength=n_pos).astype(np.int64)
    pos_snaps = snaps.subset(representative)
    distances = neighbor_graph(graph, pos_snaps, max_distance=config.eps, max_pairs=config.max_neighbor_pairs)
    result = dbscan(
        distances,
        weights,
        eps=config.eps,
        min_samples=config.min_samples,
        border_policy=config.border_policy,
        check_symmetry=False,
    )

    ids = _cluster_ids(n, position, result.labels, config.noise_policy == "singleton")
    obs_label = result.labels[position]
    obs_core = result.is_core[position]
    obs_border = result.is_border[position]
    out = points.copy()
    out["cluster_id"] = pd.Series(ids, index=out.index, dtype=object)
    out["is_noise"] = obs_label < 0
    out["is_core"] = obs_core
    out["is_border"] = obs_border
    out["n_candidate_clusters"] = result.n_candidate_clusters[position]
    out["snap_distance"] = snaps.snap_distance
    out["snapped_x"] = snaps.snapped_xy[:, 0]
    out["snapped_y"] = snaps.snapped_xy[:, 1]

    clustered = (~out["is_noise"].to_numpy()) & out["cluster_id"].notna().to_numpy()
    if clustered.any():
        table = pd.DataFrame({"cluster_id": sorted(out.loc[clustered, "cluster_id"].unique().tolist())})
        cluster_points = out.loc[clustered].groupby("cluster_id").size()
        position_series = pd.Series(position[clustered], index=out.loc[clustered, "cluster_id"].to_numpy())
        cluster_positions = position_series.groupby(level=0).nunique()
        core_points = out.loc[clustered & out["is_core"].to_numpy()].groupby("cluster_id").size()
        border_points = out.loc[clustered & out["is_border"].to_numpy()].groupby("cluster_id").size()
        table["n_points"] = table["cluster_id"].map(cluster_points).astype("int64")
        table["n_positions"] = table["cluster_id"].map(cluster_positions).astype("int64")
        table["n_core"] = table["cluster_id"].map(core_points).fillna(0).astype("int64")
        table["n_border"] = table["cluster_id"].map(border_points).fillna(0).astype("int64")
        table = table[CLUSTER_COLUMNS]
    else:
        table = _empty_clusters()

    pos_component = graph.component[pos_snaps.u]
    obs_component = pos_component[position]
    component_counts = np.unique(obs_component, return_counts=True)[1]
    cluster_counts = np.bincount(obs_label[obs_label >= 0]) if (obs_label >= 0).any() else np.zeros(0, dtype=np.int64)
    noise = obs_label < 0
    stats.update(
        n_positions=int(n_pos),
        max_position_weight=int(weights.max()),
        stacked_positions=int((weights >= config.min_samples).sum()),
        n_pairs=int(distances.nnz // 2),
        network_components_used=int(len(component_counts)),
        share_on_largest_component=float(component_counts.max() / n),
        snap_distance_median=float(np.median(snaps.snap_distance)),
        snap_distance_p95=float(np.quantile(snaps.snap_distance, 0.95)),
        snap_distance_max=float(snaps.snap_distance.max()),
        n_clusters=int(result.n_clusters),
        noise_share=float(noise.mean()),
        core_share=float(obs_core.mean()),
        border_share=float(obs_border.mean()),
        largest_cluster_share=float(cluster_counts.max() / n) if len(cluster_counts) else 0.0,
        neighborhood_weight_median=float(np.median(result.neighborhood_weight[position])),
        neighborhood_weight_p95=float(np.quantile(result.neighborhood_weight[position], 0.95)),
        contested_share=float((obs_border & (result.n_candidate_clusters[position] >= 2)).mean()),
        eps=float(config.eps),
        min_samples=int(config.min_samples),
        border_policy=config.border_policy,
    )
    return out, table, stats


def _check_snap_distance(ids, distances: np.ndarray, limit: float | None) -> None:
    if limit is None or math.isinf(limit) or len(distances) == 0:
        return
    too_far = distances > limit
    if too_far.any():
        k = int(np.flatnonzero(too_far)[0])
        raise ValueError(
            f"point {ids[k]!r} lies {distances[k]:g} units from the nearest network arc; "
            f"max_snap_distance={limit:g} ({int(too_far.sum())} point(s) exceed it)"
        )


def _snap_points_to_network(network: gpd.GeoDataFrame, points: gpd.GeoDataFrame, vertex_digits: int | None):
    if len(points) == 0:
        return None, empty_snaps()
    graph = build_network_graph(network.geometry.to_numpy(), vertex_digits=vertex_digits)
    xy = np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])
    return graph, snap_points(graph, xy)


def _run_unit(
    points: gpd.GeoDataFrame,
    snaps: Snaps,
    graph: NetworkGraph | None,
    config: DBSCANConfig,
    point_id_col: str,
) -> ClusterOutputs:
    start = time.perf_counter()
    order = canonical_order(points[point_id_col].tolist()) if len(points) else np.empty(0, dtype=np.int64)
    points = points.iloc[order].reset_index(drop=True)
    snaps = snaps.subset(order)
    _check_snap_distance(points[point_id_col].tolist(), snaps.snap_distance, config.max_snap_distance)
    clustered_points, clusters, stats = _cluster_unit(points, snaps, graph, config)
    stats["seconds"] = round(time.perf_counter() - start, 3)
    return ClusterOutputs(points=clustered_points, clusters=clusters, summary=stats)


def _analysis_metadata(crs, graph: NetworkGraph | None) -> dict[str, Any]:
    try:
        unit = crs.axis_info[0].unit_name or None
    except (AttributeError, IndexError):
        unit = None
    return {
        "crs": str(crs),
        "distance_unit": unit,
        "graph_built": graph is not None,
        "network_vertices": None if graph is None else int(graph.n_vertices),
        "network_arcs": None if graph is None else int(graph.n_arcs),
        "network_components": None if graph is None else int(graph.n_components),
    }


def cluster_geodataframes(
    points: gpd.GeoDataFrame,
    network: gpd.GeoDataFrame,
    config: DBSCANConfig,
    *,
    point_id_col: str = "point_id",
    vertex_digits: int | None = DEFAULT_VERTEX_DIGITS,
) -> ClusterOutputs:
    """Cluster all supplied points as one network-distance DBSCAN unit.

    ``vertex_digits`` controls significant-digit rounding used to identify
    shared network vertices. Set it to ``None`` to disable rounding.
    """
    if not isinstance(config, DBSCANConfig):
        raise TypeError("config must be a DBSCANConfig")
    network = prepare_network(network)
    crs = network.crs
    work = prepare_points(points, crs, point_id_col)
    graph, snaps = _snap_points_to_network(network, work, vertex_digits)
    outputs = _run_unit(work, snaps, graph, config, point_id_col)
    outputs.summary = {"group": None, "n_points_input": int(len(points)), **outputs.summary}
    outputs.analysis = _analysis_metadata(crs, graph)
    return outputs


def _summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[SUMMARY_COLUMNS]


def _dependency_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _manifest(kind: str, inputs: dict[str, Any], config: DBSCANConfig, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "net_dbscan_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": kind,
        "inputs": inputs,
        "clustering": config.as_dict(),
        "network_backend": {
            "construction_and_snapping": "shapely",
            "shapely_version": _dependency_version("shapely"),
            "bounded_shortest_paths": "scipy",
            "scipy_version": _dependency_version("scipy"),
            "dbscan": "scikit-learn",
            "scikit_learn_version": _dependency_version("scikit-learn"),
        },
        **extra,
    }


def _refuse_existing(paths, force: bool) -> None:
    if force:
        return
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f"output already exists: {path}; pass force=True or --force to replace it")


def cluster_files(
    *,
    points_path: str | Path,
    network_path: str | Path,
    output_dir: str | Path,
    config: DBSCANConfig,
    point_id_col: str = "point_id",
    points_layer: str | None = None,
    network_layer: str | None = None,
    vertex_digits: int | None = DEFAULT_VERTEX_DIGITS,
    force: bool = False,
) -> ClusterOutputs:
    """Cluster one point file and write points, cluster table and diagnostics."""
    output_dir = Path(output_dir)
    targets = {
        "points": output_dir / "clustered_points.parquet",
        "clusters": output_dir / "cluster_table.parquet",
        "summary": output_dir / "summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    _refuse_existing(targets.values(), force)
    # Every file run writes Parquet; fail before expensive input processing.
    _require_pyarrow()
    outputs = cluster_geodataframes(
        read_vector(points_path, name="points", layer=points_layer),
        read_vector(network_path, name="network", layer=network_layer),
        config,
        point_id_col=point_id_col,
        vertex_digits=vertex_digits,
    )
    write_geoparquet(targets["points"], outputs.points, force=force)
    write_parquet(targets["clusters"], outputs.clusters, force=force)
    write_csv(targets["summary"], _summary_frame([outputs.summary]), force=force)
    inputs = {
        "points": str(points_path),
        "points_layer": points_layer,
        "network": str(network_path),
        "network_layer": network_layer,
        "point_id_col": point_id_col,
    }
    write_json(
        targets["manifest"],
        _manifest(
            "single",
            inputs,
            config,
            {"vertex_digits": vertex_digits, "analysis": outputs.analysis, "summary": outputs.summary},
        ),
        force=force,
    )
    return outputs


def group_output_token(value) -> str:
    """Filesystem-safe, unambiguous token for a group value or internal key."""
    return quote(group_display(group_key(value)), safe="-_.~")


def cluster_files_by_column(
    *,
    points_path: str | Path,
    network_path: str | Path,
    output_dir: str | Path,
    group_col: str,
    config: DBSCANConfig,
    group_params: Mapping[str, Mapping[str, Any]] | None = None,
    group_universe: list[str] | None = None,
    missing_group_policy: str = "exclude",
    point_id_col: str = "point_id",
    points_layer: str | None = None,
    network_layer: str | None = None,
    vertex_digits: int | None = DEFAULT_VERTEX_DIGITS,
    force: bool = False,
) -> pd.DataFrame:
    """Cluster each group independently while sharing preparation and snapping."""
    if not isinstance(config, DBSCANConfig):
        raise TypeError("config must be a DBSCANConfig")
    _require_pyarrow()
    points = read_vector(points_path, name="points", layer=points_layer)
    network = read_vector(network_path, name="network", layer=network_layer)
    if group_col not in points.columns:
        raise ValueError(f"group column {group_col!r} not found")
    if group_col == point_id_col:
        raise ValueError("group column and point id column must differ")
    if missing_group_policy not in {"exclude", "include", "error"}:
        raise ValueError("missing_group_policy must be 'exclude', 'include', or 'error'")

    raw = points[group_col]
    keys = raw.map(group_key)
    missing_mask = keys.isin({NULL_GROUP_KEY, BLANK_GROUP_KEY})
    n_missing_input = int(missing_mask.sum())
    if missing_group_policy == "error" and n_missing_input:
        null_n = int((keys == NULL_GROUP_KEY).sum())
        blank_n = int((keys == BLANK_GROUP_KEY).sum())
        raise ValueError(
            f"group column {group_col!r} has {n_missing_input:,} missing/blank value(s) "
            f"({null_n:,} null, {blank_n:,} blank); choose missing_group_policy='exclude' or 'include'"
        )

    observed = keys if missing_group_policy == "include" else keys.loc[~missing_mask]

    def _group_sort_key(k):
        if k == BLANK_GROUP_KEY:
            return (0, "")
        if k == NULL_GROUP_KEY:
            return (2, "")
        return (1, str(k))

    observed_values = sorted(observed.unique().tolist(), key=_group_sort_key)
    if group_universe is not None:
        universe = [group_key(v) for v in group_universe]
        if any(v in {NULL_GROUP_KEY, BLANK_GROUP_KEY} for v in universe):
            raise ValueError("group_universe cannot contain null/blank values")
        if len(universe) != len(set(universe)):
            raise ValueError("group_universe lists a group more than once")
        universe_set = set(universe)
        values = universe + [v for v in observed_values if v not in universe_set]
    else:
        universe = None
        universe_set = set()
        values = observed_values

    normalized_group_params = normalize_group_params(group_params)
    if normalized_group_params:
        unknown = sorted(set(normalized_group_params) - set(values))
        if unknown:
            shown = [group_display(k) for k in unknown[:10]]
            raise ValueError(f"group parameter keys match no group in {group_col!r}: {shown}")
    configs = {key: resolve_group_config(config, normalized_group_params, key) for key in values}

    output_dir = Path(output_dir)
    planned: dict[str, dict[str, Path]] = {}
    seen: dict[str, str] = {}
    for key in values:
        token = group_output_token(key)
        name = f"group_{token}.parquet"
        collision_key = name.casefold()
        if collision_key in seen:
            raise ValueError(
                f"group values {group_display(seen[collision_key])!r} and {group_display(key)!r} "
                f"map to colliding file names on case-insensitive filesystems ({name!r})"
            )
        seen[collision_key] = key
        planned[key] = {
            "points": output_dir / "points" / name,
            "clusters": output_dir / "clusters" / name,
        }
    run_files = [output_dir / "summary.csv", output_dir / "manifest.json"]
    _refuse_existing([p for paths in planned.values() for p in paths.values()] + run_files, force)

    network = prepare_network(network)
    crs = network.crs
    work = prepare_points(points, crs, point_id_col)
    work_keys = keys.to_numpy()

    # Missing groups excluded from the analysis are removed before snapping,
    # which is the expensive shared preprocessing stage.
    if missing_group_policy == "exclude" and n_missing_input:
        keep = ~np.isin(work_keys, [NULL_GROUP_KEY, BLANK_GROUP_KEY])
        work = work.loc[keep].reset_index(drop=True)
        work_keys = work_keys[keep]

    graph, snaps_all = _snap_points_to_network(network, work, vertex_digits)
    analysis = _analysis_metadata(crs, graph)

    outside_universe_keys = [v for v in values if universe is not None and v not in universe_set]
    outside_universe = [group_display(v) for v in outside_universe_keys]
    rows: list[dict[str, Any]] = []
    for key in values:
        rows_in = np.flatnonzero(work_keys == key)
        subset = work.iloc[rows_in].reset_index(drop=True)
        group_snaps = snaps_all.subset(rows_in)
        try:
            outputs = _run_unit(subset, group_snaps, graph, configs[key], point_id_col)
        except Exception as exc:
            n_positions = len(distinct_positions(group_snaps)[1]) if len(group_snaps) else 0
            context = (
                f"group {group_display(key)!r} failed "
                f"(n_points_input={int((keys == key).sum()):,}, "
                f"n_points={len(subset):,}, n_positions={n_positions:,})"
            )
            if hasattr(exc, "add_note"):
                exc.add_note(context)
                raise
            if isinstance(exc, ValueError):
                raise ValueError(f"{context}: {exc}") from exc
            raise RuntimeError(f"{context}: {exc}") from exc

        summary = {
            "group": group_display(key),
            "n_points_input": int((keys == key).sum()),
            **outputs.summary,
            "declared_in_universe": (key in universe_set) if universe is not None else None,
            "missing_group_policy": missing_group_policy,
            "n_missing_group_input": n_missing_input,
        }
        rows.append(summary)
        write_geoparquet(planned[key]["points"], outputs.points, force=force)
        write_parquet(planned[key]["clusters"], outputs.clusters, force=force)

    summary = _summary_frame(rows)
    summary.attrs["missing_group_policy"] = missing_group_policy
    summary.attrs["n_missing_group_input"] = n_missing_input
    summary.attrs["groups_outside_universe"] = outside_universe
    write_csv(output_dir / "summary.csv", summary, force=force)
    inputs = {
        "points": str(points_path),
        "points_layer": points_layer,
        "network": str(network_path),
        "network_layer": network_layer,
        "group_col": group_col,
        "point_id_col": point_id_col,
    }
    extra = {
        "vertex_digits": vertex_digits,
        "analysis": analysis,
        "group_params": {group_display(k): dict(v) for k, v in normalized_group_params.items()},
        "group_universe": list(group_universe) if group_universe is not None else None,
        "missing_group_policy": missing_group_policy,
        "n_missing_group_input": n_missing_input,
        "groups_outside_universe": outside_universe,
        "groups": {
            group_display(key): {"files": {k: str(p.relative_to(output_dir)) for k, p in planned[key].items()}}
            for key in values
        },
    }
    write_json(output_dir / "manifest.json", _manifest("grouped", inputs, config, extra), force=force)
    return summary


__all__ = [
    "ClusterOutputs",
    "cluster_geodataframes",
    "cluster_files",
    "cluster_files_by_column",
    "NeighborPairLimitError",
]
