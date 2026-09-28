import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from net_dbscan import DBSCANConfig, cluster_geodataframes
from net_dbscan.pipeline import _cluster_ids

CRS = "EPSG:32651"


def test_cluster_ids_share_one_singleton_per_noise_position():
    pos=np.array([0,0,1,2])
    labels=np.array([-1,0,-1])
    ids=_cluster_ids(4,pos,labels,True)
    assert ids[0] == ids[1]
    assert ids[2] != ids[3]


def test_empty_after_boundary_returns_without_building_graph():
    network = gpd.GeoDataFrame(geometry=[LineString([(0,0),(100,0)])], crs=CRS)
    boundary = gpd.GeoDataFrame(geometry=[Polygon([(0,0),(10,0),(10,10),(0,10)])], crs=CRS)
    points = gpd.GeoDataFrame({"point_id":["x"]}, geometry=[Point(50,50)], crs=CRS)
    out = cluster_geodataframes(points,boundary,network,DBSCANConfig(eps=10))
    assert out.points.empty
    assert out.clusters.empty
    assert out.summary["n_points"] == 0


def test_small_end_to_end():
    network = gpd.GeoDataFrame(geometry=[LineString([(0,0),(100,0)])], crs=CRS)
    boundary = gpd.GeoDataFrame(geometry=[Polygon([(-10,-10),(110,-10),(110,10),(-10,10)])], crs=CRS)
    points = gpd.GeoDataFrame(
        {"point_id":["a","b","c"]},
        geometry=[Point(10,1),Point(15,-1),Point(90,1)],crs=CRS,
    )
    out=cluster_geodataframes(points,boundary,network,DBSCANConfig(eps=10,min_samples=2))
    assert out.points.loc[out.points.point_id.isin(["a","b"]),"cluster_id"].notna().all()
    assert out.points.loc[out.points.point_id.eq("c"),"is_noise"].item()
