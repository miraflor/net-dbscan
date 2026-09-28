import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from net_dbscan.sparse_dbscan import dbscan


def _graph(n, pairs):
    rows=[]; cols=[]; data=[]
    for i,j,d in pairs:
        rows += [i,j]; cols += [j,i]; data += [d,d]
    return coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr()


def test_two_clusters_and_noise():
    g = _graph(5, [(0,1,1.0),(2,3,1.0)])
    r = dbscan(g, np.ones(5,dtype=int), eps=2, min_samples=2)
    assert r.n_clusters == 2
    assert r.labels[4] == -1
    assert r.is_core.tolist() == [True, True, True, True, False]


def test_weighted_position_preserves_duplicate_semantics():
    # Position 0 stands for three co-located observations; it is core by itself.
    g = csr_matrix((2,2), dtype=float)
    r = dbscan(g, np.array([3,1]), eps=1, min_samples=3)
    assert r.labels[0] >= 0
    assert r.is_core.tolist() == [True, False]
    assert r.neighborhood_weight.tolist() == [3,1]


def test_border_point_is_not_core():
    # 0--1--2, eps relation already encoded; min_samples=3 makes 1 core only.
    g = _graph(3, [(0,1,1.0),(1,2,1.0)])
    r = dbscan(g, np.ones(3,dtype=int), eps=1, min_samples=3)
    assert r.is_core.tolist() == [False, True, False]
    assert (r.labels >= 0).all()


def test_rejects_asymmetric_sparse_relation():
    g = coo_matrix(([1.0], ([0],[1])), shape=(2,2)).tocsr()
    with pytest.raises(ValueError, match="symmetric"):
        dbscan(g, np.ones(2,dtype=int), eps=2, min_samples=1)


def test_empty():
    r = dbscan(csr_matrix((0,0)), np.empty(0,dtype=int), eps=1, min_samples=1)
    assert r.labels.size == 0
    assert r.n_clusters == 0
