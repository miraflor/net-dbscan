"""Tests for the changes in net-dbscan 0.3.0."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

import net_dbscan
from conftest import contested_motifs, grid_roads, square_boundary, touching_blobs
from net_dbscan import DBSCANConfig, cluster_files, cluster_geodataframes
from net_dbscan.network import build_network_graph, distinct_positions, neighbor_graph, snap_points
from net_dbscan.sparse_dbscan import dbscan

CONFIG = DBSCANConfig(eps=60, min_samples=6)


def labels_by_id(points):
    return points.set_index("point_id").sort_index()


def same_partition(a, b):
    a, b = pd.factorize(a.fillna("-"))[0], pd.factorize(b.fillna("-"))[0]
    return len(set(zip(a, b))) == len(set(a)) == len(set(b))


# --- border policies on a hand-made example ---------------------------------


def line_graph(x, eps):
    d = np.abs(np.subtract.outer(x, x))
    d[d > eps] = 0
    return csr_matrix(d)


def test_border_policies_on_a_contested_point():
    # Cluster A at 0..3, cluster B at 10.5..13.5, and a non-core point at 6.9
    # within eps of A's point at 3 (3.9 away) and of B's point at 10.5 (3.6 away).
    x = np.array([0, 1, 2, 3, 10.5, 11.5, 12.5, 13.5, 6.9])
    w = np.ones(len(x), dtype=np.int64)
    graph = line_graph(x, 4.0)
    expansion = dbscan(graph, w, eps=4.0, min_samples=4)
    assert not expansion.is_core[8] and expansion.n_candidate_clusters[8] == 2
    assert expansion.labels[8] == expansion.labels[0]  # A is expanded first
    nearest = dbscan(graph, w, eps=4.0, min_samples=4, border_policy="nearest_core")
    assert nearest.labels[8] == nearest.labels[4]  # B's core point is nearer
    core_only = dbscan(graph, w, eps=4.0, min_samples=4, border_policy="core_only")
    assert core_only.labels[8] == -1
    for result in (nearest, core_only):
        assert np.array_equal(result.labels[:8], expansion.labels[:8]) and np.array_equal(result.is_core, expansion.is_core)
    # Reversing the order changes the standard rule but not the nearest-core rule.
    order = np.array([4, 5, 6, 7, 0, 1, 2, 3, 8])
    reversed_expansion = dbscan(line_graph(x[order], 4.0), w, eps=4.0, min_samples=4)
    reversed_nearest = dbscan(line_graph(x[order], 4.0), w, eps=4.0, min_samples=4, border_policy="nearest_core")
    assert reversed_expansion.labels[8] == reversed_expansion.labels[0]  # now B is expanded first
    assert reversed_nearest.labels[8] == reversed_nearest.labels[0]  # B again, as before


def test_invalid_border_policy():
    with pytest.raises(ValueError, match="border_policy"):
        DBSCANConfig(eps=10, border_policy="nearest")


# --- invariance through the pipeline -----------------------------------------


def run(points, roads, config=CONFIG):
    return cluster_geodataframes(points, square_boundary(), roads, config)


MOTIF_CONFIG = DBSCANConfig(eps=40, min_samples=4)


def test_road_row_order_changes_nothing():
    """0.2.0 numbered positions by spaghetti's vertex IDs, which follow the road rows."""
    roads, points = grid_roads(), contested_motifs()
    first = run(points, roads, MOTIF_CONFIG)
    assert first.summary["contested_share"] > 0  # the test needs contested border points
    for seed in [3, 7, 11]:
        shuffled = roads.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        second = run(points, shuffled, MOTIF_CONFIG)
        a, b = labels_by_id(first.points), labels_by_id(second.points)
        assert a["cluster_id"].fillna("-").equals(b["cluster_id"].fillna("-"))
        for column in ["is_core", "n_candidate_clusters", "snapped_x", "snapped_y"]:
            assert np.array_equal(a[column].to_numpy(), b[column].to_numpy())


def test_point_ids_and_row_order_change_no_partition():
    roads, points = grid_roads(), contested_motifs()
    first = run(points, roads, MOTIF_CONFIG)
    assert first.summary["contested_share"] > 0
    shuffled = points.sample(frac=1.0, random_state=2).reset_index(drop=True)
    renamed = shuffled.copy()
    renamed["point_id"] = [f"z{k:04d}" for k in range(len(renamed), 0, -1)]
    mapping = dict(zip(renamed["point_id"], shuffled["point_id"]))
    second = run(renamed, roads, MOTIF_CONFIG)
    b = second.points.assign(point_id=second.points["point_id"].map(mapping))
    assert same_partition(labels_by_id(first.points)["cluster_id"], labels_by_id(b)["cluster_id"])


@pytest.mark.parametrize("policy", ["expansion", "nearest_core", "core_only"])
def test_border_policy_through_the_pipeline(policy):
    roads, points = grid_roads(), contested_motifs()
    out = run(points, roads, MOTIF_CONFIG.replace(border_policy=policy))
    assert (out.points["n_candidate_clusters"] == 2).sum() == 5
    p = out.points
    assert out.summary["border_policy"] == policy
    border = (~p["is_core"]) & p["cluster_id"].notna()
    if policy == "core_only":
        assert not border.any() and out.clusters["n_border"].eq(0).all()
    base = run(points, roads, MOTIF_CONFIG)
    assert np.array_equal(p["is_core"].to_numpy(), base.points["is_core"].to_numpy())
    assert (p["n_candidate_clusters"] >= 0).all() and (p.loc[p["is_core"], "n_candidate_clusters"] == 1).all()


# --- network distances against brute force ------------------------------------


def brute_force(graph, pos):
    n_v = graph.n_vertices
    node = np.full(len(pos), -1, dtype=np.int64)
    at_u, at_v = pos.offset == 0, pos.offset == pos.arc_length
    node[at_u] = pos.u[at_u]
    node[at_v & ~at_u] = pos.v[at_v & ~at_u]
    interior = np.flatnonzero(node < 0)
    node[interior] = n_v + np.arange(len(interior))
    arc_key = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(graph.arc_u, graph.arc_v))}
    on_arc = {}
    for i in interior:
        on_arc.setdefault(arc_key[(int(pos.u[i]), int(pos.v[i]))], []).append(i)
    rows, cols, weights = [], [], []
    for k, (u, v, length) in enumerate(zip(graph.arc_u, graph.arc_v, graph.arc_length)):
        chain = sorted(on_arc.get(k, []), key=lambda i: pos.offset[i])
        stops = [int(u)] + [int(node[i]) for i in chain] + [int(v)]
        offsets = [0.0] + [float(pos.offset[i]) for i in chain] + [float(length)]
        for a, b, oa, ob in zip(stops[:-1], stops[1:], offsets[:-1], offsets[1:]):
            rows += [a, b]; cols += [b, a]; weights += [ob - oa, ob - oa]
    size = n_v + len(interior)
    full = dijkstra(coo_matrix((weights, (rows, cols)), shape=(size, size)).tocsr(), directed=False, indices=node)
    return full[:, node]


@pytest.mark.parametrize("eps", [40.0, 150.0])
def test_neighbour_distances_match_brute_force(eps):
    roads = grid_roads(n=9, jitter=10.0, seed=4)
    roads = roads.iloc[np.random.default_rng(1).random(len(roads)) > 0.15].reset_index(drop=True)  # dead ends, detours
    graph = build_network_graph(roads)
    rng = np.random.default_rng(2)
    points = touching_blobs().iloc[:0]
    xy = np.vstack([rng.uniform(0, 800, size=(120, 2)), graph.vertex_xy[:6] + 0.3])
    frame = points.__class__({"point_id": [f"b{i}" for i in range(len(xy))]}, geometry=shapely.points(xy), crs=roads.crs)
    snaps = snap_points(graph, frame)
    position, representative = distinct_positions(snaps)
    pos = snaps.subset(representative)
    found = neighbor_graph(graph, pos, max_distance=eps).toarray()
    expected = brute_force(graph, pos)
    np.fill_diagonal(expected, np.inf)
    within = expected <= eps
    assert np.array_equal(found != 0, within)
    assert np.allclose(found[within], expected[within], atol=1e-9)


# --- interface ---------------------------------------------------------------


def test_the_function_does_not_hide_the_module():
    module = importlib.import_module("net_dbscan.sparse_dbscan")
    assert net_dbscan.dbscan is module.dbscan
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("net_dbscan.dbscan")


def test_old_names_are_aliases():
    assert net_dbscan.RoadGraph is net_dbscan.NetworkGraph
    assert net_dbscan.build_road_graph is net_dbscan.build_network_graph
    assert net_dbscan.NetDBSCANConfig is net_dbscan.DBSCANConfig


def test_boundary_is_optional_and_cluster_table_file(tmp_path):
    roads, points = grid_roads(jitter=8.0), touching_blobs()
    without = cluster_geodataframes(points, None, roads, CONFIG)
    assert without.summary["n_points"] == len(points)
    pytest.importorskip("pyarrow")
    points.to_parquet(tmp_path / "p.parquet")
    roads.to_parquet(tmp_path / "r.parquet")
    cluster_files(points_path=tmp_path / "p.parquet", network_path=tmp_path / "r.parquet", output_dir=tmp_path / "out", config=CONFIG)
    assert (tmp_path / "out" / "cluster_table.parquet").exists()
    table = pd.read_parquet(tmp_path / "out" / "cluster_table.parquet")
    assert list(table.columns) == ["cluster_id", "n_points", "n_positions", "n_core", "n_border"]


def test_missing_extras_give_clear_messages(monkeypatch):
    from net_dbscan import _cli_entry, io

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match=r"net-dbscan\[files\]"):
        io._require_pyarrow()
    monkeypatch.setitem(sys.modules, "typer", None)
    monkeypatch.delitem(sys.modules, "net_dbscan.cli", raising=False)
    with pytest.raises(SystemExit, match=r"net-dbscan\[cli\]"):
        _cli_entry.main()


def test_python_dash_m_runs_the_command():
    src = str(Path(net_dbscan.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([sys.executable, "-m", "net_dbscan", "--version"], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and result.stdout.strip() == net_dbscan.__version__


def test_command_line_end_to_end(tmp_path):
    pytest.importorskip("pyarrow")
    from typer.testing import CliRunner

    from net_dbscan.cli import app

    roads, points = grid_roads(jitter=8.0), touching_blobs()
    points.to_parquet(tmp_path / "p.parquet")
    roads.to_parquet(tmp_path / "r.parquet")
    square_boundary().to_parquet(tmp_path / "b.parquet")
    args = ["cluster", "--points", str(tmp_path / "p.parquet"), "--network", str(tmp_path / "r.parquet"), "--boundary", str(tmp_path / "b.parquet"),
            "--output-dir", str(tmp_path / "out"), "--eps", "60", "--min-samples", "6", "--border-policy", "nearest_core"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert summary.loc[0, "border_policy"] == "nearest_core"


def test_snapped_positions_do_not_depend_on_road_order():
    """Canonical graph construction and snapping are invariant to network row order."""
    roads = grid_roads(jitter=9.0, seed=5)
    points = touching_blobs()
    first = snap_points(build_network_graph(roads), points)
    for seed in [1, 2, 3]:
        other = snap_points(build_network_graph(roads.sample(frac=1.0, random_state=seed).reset_index(drop=True)), points)
        for field in ["u", "v", "offset", "arc_length", "snap_distance", "snapped_xy"]:
            assert np.array_equal(getattr(first, field), getattr(other, field)), field
