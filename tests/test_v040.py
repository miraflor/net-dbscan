"""Hardening and backend tests introduced in net-dbscan 0.4.0."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from scipy.sparse import coo_matrix
from shapely.geometry import LineString, Point

import net_dbscan.network as network_module
import net_dbscan.pipeline as pipeline
from net_dbscan import DBSCANConfig, cluster_geodataframes
from net_dbscan.config import (
    BLANK_GROUP_KEY,
    NULL_GROUP_KEY,
    group_display,
    group_key,
    normalize_group_params,
)
from net_dbscan.network import build_network_graph, distinct_positions, neighbor_graph, snap_points
from net_dbscan.sparse_dbscan import dbscan

CRS = "EPSG:32651"


def _symmetric_graph(n, entries):
    rows, cols, data = [], [], []
    for i, j, d in entries:
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([d, d])
    return coo_matrix((data, (rows, cols)), shape=(n, n))


def test_duplicate_sparse_entries_are_rejected_before_csr_summing():
    graph = coo_matrix(([0.4, 0.4, 0.4], ([0, 0, 1], [1, 1, 0])), shape=(2, 2))
    with pytest.raises(ValueError, match="duplicate"):
        dbscan(graph, np.ones(2, dtype=np.int64), eps=1.0, min_samples=1)


def test_nonzero_stored_diagonal_is_rejected():
    graph = coo_matrix(([0.5, 0.2, 0.2], ([0, 0, 1], [0, 1, 0])), shape=(2, 2))
    with pytest.raises(ValueError, match="diagonal"):
        dbscan(graph, np.ones(2, dtype=np.int64), eps=1.0, min_samples=1)


def test_boolean_weights_are_rejected():
    with pytest.raises(ValueError, match="integer array"):
        dbscan(coo_matrix((2, 2)), np.array([True, True]), eps=1.0, min_samples=1)


def test_core_only_retains_intrinsic_border_class():
    # 0--1--2 with min_samples=3: only 1 is core, 0 and 2 are border points.
    graph = _symmetric_graph(3, [(0, 1, 1.0), (1, 2, 1.0)])
    result = dbscan(graph, np.ones(3, dtype=np.int64), eps=1.0, min_samples=3, border_policy="core_only")
    assert result.is_core.tolist() == [False, True, False]
    assert result.is_border.tolist() == [True, False, True]
    assert result.labels.tolist() == [-1, 0, -1]


def test_pipeline_reports_intrinsic_border_share_under_core_only():
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (20, 0)])], crs=CRS)
    points = gpd.GeoDataFrame(
        {"point_id": ["a", "b", "c"]},
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs=CRS,
    )
    out = cluster_geodataframes(points, None, roads, DBSCANConfig(eps=1.0, min_samples=3, border_policy="core_only"))
    assert out.points["is_border"].tolist() == [True, False, True]
    assert out.points["is_noise"].tolist() == [True, False, True]
    assert out.summary["border_share"] == pytest.approx(2 / 3)
    assert out.clusters["n_border"].sum() == 0


def test_group_sentinels_do_not_collide_with_literal_strings():
    assert group_key(None) == NULL_GROUP_KEY
    assert group_key("") == BLANK_GROUP_KEY
    assert group_key("__null__") == "__null__"
    assert group_key("__blank__") == "__blank__"
    assert group_display(NULL_GROUP_KEY) == "__null__"
    assert group_display("__null__") == "literal:__null__"
    normalized = normalize_group_params({"__null__": {"eps": 2}, "literal:__null__": {"eps": 3}})
    assert set(normalized) == {NULL_GROUP_KEY, "__null__"}


def test_case_insensitive_group_filename_collision_is_rejected(monkeypatch, tmp_path):
    points = gpd.GeoDataFrame(
        {"point_id": ["a", "b"], "kind": ["A", "a"]},
        geometry=[Point(0, 0), Point(1, 0)],
        crs=CRS,
    )
    roads = gpd.GeoDataFrame(geometry=[LineString([(-1, 0), (2, 0)])], crs=CRS)

    def fake_read(path, *, name, layer=None):
        return {"points": points, "network": roads}[name].copy()

    monkeypatch.setattr(pipeline, "_require_pyarrow", lambda: None)
    monkeypatch.setattr(pipeline, "read_vector", fake_read)
    with pytest.raises(ValueError, match="case-insensitive"):
        pipeline.cluster_files_by_column(
            points_path="p.parquet",
            network_path="r.parquet",
            output_dir=tmp_path,
            group_col="kind",
            config=DBSCANConfig(eps=10),
        )


def test_file_pipeline_checks_pyarrow_before_reading(monkeypatch, tmp_path):
    seen = []

    def missing():
        seen.append("pyarrow")
        raise ImportError("missing pyarrow")

    def should_not_read(*args, **kwargs):
        seen.append("read")
        raise AssertionError("read_vector should not be reached")

    monkeypatch.setattr(pipeline, "_require_pyarrow", missing)
    monkeypatch.setattr(pipeline, "read_vector", should_not_read)
    with pytest.raises(ImportError, match="missing pyarrow"):
        pipeline.cluster_files(
            points_path="points.gpkg",
            network_path="network.gpkg",
            output_dir=tmp_path / "out",
            config=DBSCANConfig(eps=100),
        )
    assert seen == ["pyarrow"]


def test_lines_join_only_at_shared_vertices():
    shared = build_network_graph([LineString([(0, 0), (10, 0)]), LineString([(10, 0), (10, 10)])])
    crossing = build_network_graph([LineString([(0, 5), (10, 5)]), LineString([(5, 0), (5, 10)])])
    assert shared.n_components == 1
    assert crossing.n_components == 2


def test_vertex_rounding_can_join_nearly_identical_endpoints():
    lines = [LineString([(100000.0, 0), (100010.0, 0)]), LineString([(100010.0 + 1e-9, 0), (100020.0, 0)])]
    assert build_network_graph(lines).n_components == 1
    assert build_network_graph(lines, vertex_digits=None).n_components == 2
    for bad in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="vertex_digits"):
            build_network_graph(lines, vertex_digits=bad)


def test_equal_nearest_arc_tie_is_deterministic_under_row_reordering():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(0, 2), (10, 2)])],
        crs=CRS,
    )
    xy = np.array([[5.0, 1.0]])
    first = snap_points(build_network_graph(roads), xy)
    second = snap_points(build_network_graph(roads.iloc[::-1].reset_index(drop=True)), xy)
    for field in ["u", "v", "offset", "arc_length", "snap_distance", "snapped_xy"]:
        assert np.array_equal(getattr(first, field), getattr(second, field))


def test_same_arc_batches_are_hard_bounded(monkeypatch):
    graph = build_network_graph([LineString([(0, 0), (100, 0)])])
    snaps = snap_points(graph, np.column_stack([np.arange(1.0, 13.0), np.zeros(12)]))
    monkeypatch.setattr(network_module, "_SAME_ARC_BATCH_PAIRS", 5)
    parts = list(network_module._pairs_on_same_arc(snaps, 1000.0, 1000))
    assert parts
    assert max(len(part[0]) for part in parts) <= 5
    pairs = {(int(i), int(j)) for a, b, _ in parts for i, j in zip(a, b)}
    assert len(pairs) == 12 * 11 // 2


def test_tiny_internal_chunks_preserve_neighbor_graph(monkeypatch):
    lines = []
    for k in range(8):
        angle = 2 * np.pi * k / 8
        lines.append(LineString([(0, 0), (100 * np.cos(angle), 100 * np.sin(angle))]))
    graph = build_network_graph(lines, vertex_digits=None)
    xy = []
    for k in range(8):
        angle = 2 * np.pi * k / 8
        for radius in [10, 20, 30, 40]:
            xy.append((radius * np.cos(angle), radius * np.sin(angle)))
    snaps = snap_points(graph, np.asarray(xy))
    _, representative = distinct_positions(snaps)
    positions = snaps.subset(representative)
    expected = neighbor_graph(graph, positions, max_distance=1000, max_pairs=100000)

    monkeypatch.setattr(network_module, "_CANDIDATE_CHUNK", 8)
    monkeypatch.setattr(network_module, "_SAME_ARC_BATCH_PAIRS", 7)
    got = neighbor_graph(graph, positions, max_distance=1000, max_pairs=100000)
    assert (expected != got).nnz == 0


def test_spaghetti_is_not_a_runtime_dependency():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    dependencies = text.split("[project.optional-dependencies]", 1)[0]
    assert "spaghetti" not in dependencies
