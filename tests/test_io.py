import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from net_dbscan.io import canonical_order, covered_by, prepare_boundary, prepare_network, prepare_points

CRS = "EPSG:32651"


def test_boundary_covers_edge_point():
    boundary = gpd.GeoDataFrame(geometry=[Polygon([(0,0),(10,0),(10,10),(0,10)])], crs=CRS)
    points = gpd.GeoDataFrame({"point_id":["edge","out"]}, geometry=[Point(0,5), Point(-1,5)], crs=CRS)
    p = prepare_points(points, CRS, "point_id")
    b = prepare_boundary(boundary, CRS)
    assert p.loc[covered_by(p,b),"point_id"].tolist() == ["edge"]


def test_network_requires_projected_crs():
    roads = gpd.GeoDataFrame(geometry=[LineString([(0,0),(1,0)])], crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected"):
        prepare_network(roads)


def test_multiline_is_exploded_before_graph_construction():
    geom = MultiLineString([[(0,0),(1,0)], [(10,0),(11,0)]])
    roads = gpd.GeoDataFrame(geometry=[geom], crs=CRS)
    got = prepare_network(roads)
    assert len(got) == 2
    assert set(got.geometry.geom_type) == {"LineString"}


def test_point_ids_checked_per_unit():
    assert canonical_order([2,10,1]).tolist() == [2,1,0]
    with pytest.raises(ValueError, match="duplicate"):
        canonical_order(["x","x"])


def test_reserved_output_column_refused():
    points = gpd.GeoDataFrame({"point_id":["x"], "cluster_id":["old"]}, geometry=[Point(0,0)], crs=CRS)
    with pytest.raises(ValueError, match="reserved"):
        prepare_points(points, CRS, "point_id")
