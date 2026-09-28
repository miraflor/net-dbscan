"""Small synthetic data for the 0.3.0 tests."""

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Polygon

CRS = "EPSG:32651"


def grid_roads(n: int = 13, step: float = 100.0, jitter: float = 0.0, seed: int = 0) -> gpd.GeoDataFrame:
    """A noded grid; with ``jitter`` > 0 the segment lengths are not round numbers."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
    vertices = np.column_stack([gx.ravel(), gy.ravel()]) + rng.normal(0, jitter, size=(n * n, 2)) * (jitter > 0)
    index = np.arange(n * n).reshape(n, n)
    edges = [(index[r, c], index[r, c + 1]) for r in range(n) for c in range(n - 1)]
    edges += [(index[r, c], index[r + 1, c]) for r in range(n - 1) for c in range(n)]
    first, second = np.array(edges).T
    return gpd.GeoDataFrame(geometry=shapely.linestrings(np.stack([vertices[first], vertices[second]], axis=1)), crs=CRS)


def touching_blobs(seed: int = 3) -> gpd.GeoDataFrame:
    """Two dense groups whose edges are close, so some border points are contested."""
    rng = np.random.default_rng(seed)
    xy = np.vstack([rng.normal([420, 600], 70, size=(160, 2)), rng.normal([780, 600], 70, size=(160, 2)), rng.uniform(0, 1200, size=(30, 2))])
    xy = np.clip(xy, 1, 1199)
    return gpd.GeoDataFrame({"point_id": [f"q{i:04d}" for i in range(len(xy))]}, geometry=shapely.points(xy), crs=CRS)


def square_boundary(size: float = 1200.0) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[Polygon([(-20, -20), (size + 20, -20), (size + 20, size + 20), (-20, size + 20)])], crs=CRS)


def contested_motifs() -> gpd.GeoDataFrame:
    """Points on the streets of ``grid_roads(jitter=0)`` with known contested border points.

    On each of several streets: four points of cluster A, four of cluster B
    further along, and one point between them that is within 40 m of a core
    point of each cluster but is not a core point itself (eps=40, min_samples=4).
    """
    rows = []
    for k, (x0, y) in enumerate([(105, 200), (312, 400), (518, 600), (725, 800), (140, 1000)]):
        for dx in [0, 10, 20, 30, 105, 115, 125, 135, 69]:
            rows.append((x0 + dx, y))
    xy = np.array(rows, dtype=float)
    return gpd.GeoDataFrame({"point_id": [f"m{i:03d}" for i in range(len(xy))]}, geometry=shapely.points(xy), crs=CRS)
