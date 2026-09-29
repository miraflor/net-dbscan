from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from net_dbscan import DBSCANConfig
from net_dbscan.network import Snaps
from net_dbscan.pipeline import ClusterOutputs, cluster_files_by_column

CRS="EPSG:32651"


def _frames():
    points=gpd.GeoDataFrame(
        {"point_id":["a","b","c","d"],"kind":["A","",pd.NA,"A"]},
        geometry=[Point(0,0),Point(1,0),Point(2,0),Point(3,0)],crs=CRS,
    )
    network=gpd.GeoDataFrame(geometry=[LineString([(-1,0),(4,0)])],crs=CRS)
    return points,network


def _empty_snaps_for(frame):
    n=len(frame)
    return Snaps(
        np.zeros(n,dtype=int),np.ones(n,dtype=int),np.arange(n,dtype=float),
        np.full(n,10.0),np.zeros(n),np.column_stack([np.arange(n,dtype=float),np.zeros(n)])
    )


def test_missing_groups_excluded_and_universe_retained(monkeypatch,tmp_path):
    points,network=_frames()
    def fake_read(path,*,name,layer=None):
        return {"points":points,"network":network}[name].copy()
    monkeypatch.setattr("net_dbscan.pipeline._require_pyarrow", lambda: None)
    monkeypatch.setattr("net_dbscan.pipeline.read_vector",fake_read)
    snapped_counts=[]
    def fake_snap(n,p,vertex_digits):
        snapped_counts.append(len(p))
        return object(),_empty_snaps_for(p)
    monkeypatch.setattr("net_dbscan.pipeline._snap_points_to_network",fake_snap)
    monkeypatch.setattr("net_dbscan.pipeline._analysis_metadata", lambda crs, graph: {})

    seen=[]
    def fake_run(subset,snaps,graph,config,point_id_col):
        seen.append((subset["kind"].tolist(),config.eps))
        out=subset.copy(); out["cluster_id"]=None; out["is_noise"]=True; out["is_core"]=False
        out["snap_distance"]=0.0; out["snapped_x"]=out.geometry.x; out["snapped_y"]=out.geometry.y
        return ClusterOutputs(out,pd.DataFrame(columns=["cluster_id","n_points","n_positions","n_core","n_border"]),
                              {"n_points":len(out),"eps":config.eps,"min_samples":config.min_samples,"seconds":0.0})
    monkeypatch.setattr("net_dbscan.pipeline._run_unit",fake_run)
    monkeypatch.setattr("net_dbscan.pipeline.write_geoparquet",lambda *a,**k: Path(a[0]))
    monkeypatch.setattr("net_dbscan.pipeline.write_parquet",lambda *a,**k: Path(a[0]))
    monkeypatch.setattr("net_dbscan.pipeline.write_csv",lambda *a,**k: Path(a[0]))
    monkeypatch.setattr("net_dbscan.pipeline.write_json",lambda *a,**k: Path(a[0]))

    summary=cluster_files_by_column(
        points_path="p.parquet",network_path="n.parquet",
        output_dir=tmp_path,group_col="kind",config=DBSCANConfig(eps=10),
        group_params={"A":{"eps":20}},group_universe=["A","B"],missing_group_policy="exclude",
    )
    assert summary["group"].tolist() == ["A","B"]
    assert seen == [(["A","A"],20.0),([],10.0)]
    assert summary.attrs["n_missing_group_input"] == 2
    assert snapped_counts == [2]  # excluded blank/null rows are never snapped


def test_missing_group_error(monkeypatch,tmp_path):
    points,network=_frames()
    def fake_read(path,*,name,layer=None):
        return {"points":points,"network":network}[name].copy()
    monkeypatch.setattr("net_dbscan.pipeline._require_pyarrow", lambda: None)
    monkeypatch.setattr("net_dbscan.pipeline.read_vector",fake_read)
    with pytest.raises(ValueError,match="missing/blank"):
        cluster_files_by_column(
            points_path="p.parquet",network_path="n.parquet",
            output_dir=tmp_path,group_col="kind",config=DBSCANConfig(eps=10),missing_group_policy="error",
        )
