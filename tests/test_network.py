import numpy as np
import shapely
from scipy.sparse import csr_matrix

from net_dbscan.network import RoadGraph, Snaps, distinct_positions, neighbor_graph, NeighborPairLimitError


def line_graph():
    xy=np.array([[0.,0.],[10.,0.],[20.,0.]])
    adj=csr_matrix(np.array([[0.,10.,0.],[10.,0.,10.],[0.,10.,0.]]))
    segments=shapely.linestrings(np.array([[[0.,0.],[10.,0.]],[[10.,0.],[20.,0.]]]))
    return RoadGraph(
        network=None,
        vertex_xy=xy,
        arc_u=np.array([0,1]),
        arc_v=np.array([1,2]),
        arc_length=np.array([10.,10.]),
        adjacency=adj,
        component=np.array([0,0,0]),
        arc_tree=shapely.STRtree(segments),
        vertex_tree=shapely.STRtree(shapely.points(xy)),
    )


def snaps(u,v,offset,length):
    u=np.asarray(u,dtype=np.int64); v=np.asarray(v,dtype=np.int64)
    offset=np.asarray(offset,dtype=float); length=np.asarray(length,dtype=float)
    xy=np.zeros((len(u),2),dtype=float)
    xy[:,0]=10.0*u + offset
    return Snaps(u,v,offset,length,np.zeros(len(u)),xy)


def test_same_arc_distance():
    g=line_graph(); s=snaps([0,0],[1,1],[2,8],[10,10])
    m=neighbor_graph(g,s,max_distance=6,max_pairs=10)
    assert m[0,1] == 6


def test_distance_across_adjacent_arcs():
    g=line_graph(); s=snaps([0,1],[1,2],[8,3],[10,10])
    m=neighbor_graph(g,s,max_distance=5,max_pairs=10)
    assert m[0,1] == 5


def test_pair_outside_eps_absent():
    g=line_graph(); s=snaps([0,1],[1,2],[2,8],[10,10])
    m=neighbor_graph(g,s,max_distance=5,max_pairs=10)
    assert m.nnz == 0


def test_endpoint_positions_on_incident_arcs_are_one_position():
    s=snaps([0,1],[1,2],[10,0],[10,10])
    pos,rep=distinct_positions(s)
    assert pos.tolist() == [0,0]
    assert rep.tolist() == [0]


def test_position_numbering_is_row_order_independent():
    s=snaps([0,1,0],[1,2,1],[2,3,8],[10,10,10])
    p1,_=distinct_positions(s)
    order=np.array([2,0,1])
    p2,_=distinct_positions(s.subset(order))
    # Compare by original observation after undoing the row permutation.
    restored=np.empty_like(p2); restored[order]=p2
    assert p1.tolist() == restored.tolist()


def test_pair_limit_has_context():
    g=line_graph(); s=snaps([0,0,0],[1,1,1],[1,2,3],[10,10,10])
    try:
        neighbor_graph(g,s,max_distance=10,max_pairs=1)
    except NeighborPairLimitError as exc:
        assert exc.max_pairs == 1
        assert exc.max_distance == 10
    else:
        raise AssertionError("expected pair limit")
