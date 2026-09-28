"""DBSCAN on a sparse precomputed network-distance graph.

A border point is a non-core position within ``eps`` of at least one core
position. ``border_policy`` changes only how those border points are assigned;
core status and core connectivity are identical under every policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import DBSCAN as _SKDBSCAN
from sklearn.neighbors import sort_graph_by_row_values


@dataclass(frozen=True)
class DBSCANResult:
    """DBSCAN labels and intrinsic point classes on distinct snapped positions."""

    labels: np.ndarray
    is_core: np.ndarray
    is_border: np.ndarray
    neighborhood_weight: np.ndarray
    n_candidate_clusters: np.ndarray

    @property
    def n_clusters(self) -> int:
        labels = self.labels[self.labels >= 0]
        return int(len(np.unique(labels))) if len(labels) else 0

    @property
    def n_noise(self) -> int:
        return int((self.labels < 0).sum())


BORDER_POLICIES = ("expansion", "nearest_core", "core_only")


def validate_parameters(*, eps: float, min_samples: int, border_policy: str = "expansion") -> None:
    eps = float(eps)
    if not (np.isfinite(eps) and eps > 0):
        raise ValueError("eps must be a positive finite number")
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
        raise ValueError("min_samples must be an integer >= 1")
    if border_policy not in BORDER_POLICIES:
        raise ValueError(f"border_policy must be one of {BORDER_POLICIES}")


def _raw_coo(distances, n: int) -> coo_matrix:
    """Return the caller's stored sparse relation without summing duplicates."""
    try:
        graph = coo_matrix(distances, dtype=float, copy=True)
    except Exception as exc:
        raise ValueError("distance graph must be convertible to a numeric sparse matrix") from exc
    if graph.shape != (n, n):
        raise ValueError(f"distance graph must have shape {(n, n)}, got {graph.shape}")
    return graph


def _check_radius_graph(distances, n: int, eps: float, *, check_symmetry: bool = True) -> csr_matrix:
    raw = _raw_coo(distances, n)
    if not np.isfinite(raw.data).all() or (raw.data < 0).any():
        raise ValueError("stored network distances must be finite and >= 0")
    if len(raw.data) and (raw.data > eps * (1.0 + 1e-12)).any():
        raise ValueError("distance graph contains a stored pair beyond eps")

    # Sparse constructors may silently sum duplicate entries when converting to
    # CSR. Distances are observations, not additive weights, so duplicates are
    # malformed input and must be rejected before that conversion occurs.
    if len(raw.data):
        key = raw.row.astype(np.int64) * np.int64(n) + raw.col.astype(np.int64)
        order = np.argsort(key, kind="stable")
        if np.any(key[order][1:] == key[order][:-1]):
            raise ValueError("distance graph must not contain duplicate stored entries")
        diagonal = raw.row == raw.col
        if diagonal.any() and np.any(raw.data[diagonal] != 0.0):
            raise ValueError("stored diagonal distances must be zero")

    graph = raw.tocsr()
    graph.sort_indices()
    if not check_symmetry:
        return graph

    # Check the stored relation itself, including explicit zero entries. The
    # graph is small enough at this stage to sort its stored coordinates once;
    # the pipeline skips this check for graphs it constructs symmetrically.
    coo = graph.tocoo(copy=False)
    order_f = np.lexsort((coo.col, coo.row))
    order_b = np.lexsort((coo.row, coo.col))
    same = np.array_equal(coo.row[order_f], coo.col[order_b]) and np.array_equal(
        coo.col[order_f], coo.row[order_b]
    )
    if not same or not np.array_equal(coo.data[order_f], coo.data[order_b]):
        raise ValueError("distance graph must be symmetric, stored entries included")
    return graph


def _with_diagonal(graph: csr_matrix) -> csr_matrix:
    """Return a copy with every zero self-distance stored explicitly."""
    n = graph.shape[0]
    coo = graph.tocoo()
    off = coo.row != coo.col
    rows = np.concatenate([coo.row[off], np.arange(n)])
    cols = np.concatenate([coo.col[off], np.arange(n)])
    data = np.concatenate([coo.data[off], np.zeros(n)])
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _neighborhood_weight(graph: csr_matrix, weights: np.ndarray) -> np.ndarray:
    """Weighted number of observations within eps, including the point itself."""
    n = graph.shape[0]
    rows = np.repeat(np.arange(n), np.diff(graph.indptr))
    off = graph.indices != rows
    pattern = csr_matrix((off.astype(np.int64), graph.indices, graph.indptr), shape=graph.shape)
    return weights.astype(np.int64) + pattern @ weights.astype(np.int64)


def _border_candidates(graph: csr_matrix, is_core: np.ndarray):
    """(non-core position, core neighbour, distance) for all such stored pairs."""
    n = graph.shape[0]
    rows = np.repeat(np.arange(n), np.diff(graph.indptr))
    keep = ~is_core[rows] & is_core[graph.indices]
    return rows[keep], graph.indices[keep], graph.data[keep]


def _candidate_counts(is_core: np.ndarray, rows: np.ndarray, core_labels: np.ndarray) -> np.ndarray:
    """Number of distinct core clusters within eps of each position."""
    counts = is_core.astype(np.int64)
    if len(rows):
        base = int(core_labels.max()) + 1
        pairs = np.unique(rows.astype(np.int64) * base + core_labels)
        counts += np.bincount(pairs // base, minlength=len(is_core))
    return counts


def dbscan(
    distances,
    weights,
    *,
    eps: float,
    min_samples: int = 5,
    border_policy: str = "expansion",
    check_symmetry: bool = True,
) -> DBSCANResult:
    """Run DBSCAN on distinct snapped network positions.

    ``distances`` is a symmetric sparse matrix containing exactly the pairs
    whose shortest-path network distance is at most ``eps``. ``weights`` is
    the multiplicity of each distinct position, preserving the semantics of
    repeated co-located observations without materialising zero-distance
    duplicate pairs.

    ``border_policy`` chooses how intrinsic border points are labelled:

    * ``"expansion"`` (default): standard scikit-learn DBSCAN expansion.
    * ``"nearest_core"``: assign each border point to the cluster of its
      nearest core position by network distance; an exact inter-cluster tie
      goes to the smaller deterministic core-cluster label.
    * ``"core_only"``: leave every border point as noise (DBSCAN*).

    ``is_border`` in the returned result is intrinsic and therefore remains
    true under ``core_only`` even though those positions receive label ``-1``.
    Core positions and their connected components are unchanged by policy.

    ``check_symmetry=False`` skips the expensive stored-relation symmetry
    check. Use it only for graphs known to be symmetric by construction, such
    as :func:`net_dbscan.network.neighbor_graph`.
    """
    validate_parameters(eps=eps, min_samples=min_samples, border_policy=border_policy)
    weights = np.asarray(weights)
    if (
        weights.ndim != 1
        or np.issubdtype(weights.dtype, np.bool_)
        or not np.issubdtype(weights.dtype, np.integer)
    ):
        raise ValueError("weights must be a one-dimensional integer array")
    if (weights < 1).any():
        raise ValueError("every distinct position must have weight >= 1")

    n = len(weights)
    graph = _check_radius_graph(distances, n, float(eps), check_symmetry=check_symmetry)
    if n == 0:
        return DBSCANResult(
            labels=np.empty(0, dtype=np.int64),
            is_core=np.empty(0, dtype=bool),
            is_border=np.empty(0, dtype=bool),
            neighborhood_weight=np.empty(0, dtype=np.int64),
            n_candidate_clusters=np.empty(0, dtype=np.int64),
        )

    graph = _with_diagonal(graph)
    graph = sort_graph_by_row_values(graph, copy=False, warn_when_not_sorted=False)
    model = _SKDBSCAN(eps=float(eps), min_samples=int(min_samples), metric="precomputed").fit(
        graph,
        sample_weight=None if (weights == 1).all() else weights,
    )
    core = np.zeros(n, dtype=bool)
    core[model.core_sample_indices_] = True
    neighbourhood = _neighborhood_weight(graph, weights)
    expected_core = neighbourhood >= int(min_samples)
    if not np.array_equal(core, expected_core):
        raise RuntimeError("DBSCAN core flags disagree with weighted neighbourhood counts")

    labels = np.asarray(model.labels_, dtype=np.int64).copy()
    rows, cols, dist = _border_candidates(graph, core)
    candidates = _candidate_counts(core, rows, labels[cols])
    border = (~core) & (candidates >= 1)

    if border_policy == "core_only":
        labels[border] = -1
    elif border_policy == "nearest_core" and border.any():
        sel = border[rows]
        r, c, d = rows[sel], cols[sel], dist[sel]
        order = np.lexsort((labels[c], d, r))
        r, c = r[order], c[order]
        first = np.ones(len(r), dtype=bool)
        first[1:] = r[1:] != r[:-1]
        labels[r[first]] = labels[c[first]]

    return DBSCANResult(
        labels=labels,
        is_core=core,
        is_border=border,
        neighborhood_weight=neighbourhood,
        n_candidate_clusters=candidates,
    )
