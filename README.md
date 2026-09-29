# net-dbscan

`net-dbscan` clusters geospatial point observations with **DBSCAN using shortest-path distance on a supplied spatial network** rather than Euclidean distance.

For a visual, step-by-step derivation of the algorithm—including network distance, ε-neighbourhoods, core connectivity, contested border points and the three border policies—see the **[net-dbscan documentation site](https://miraflor.github.io/net-dbscan/)**.

```text
points + projected spatial network
  ↓
Shapely/SciPy graph construction + deterministic snapping
  ↓
sparse shortest-path neighbourhoods within eps
  ↓
scikit-learn DBSCAN
  ↓
clustered points + cluster table + diagnostics
```

The package is standalone. It does not depend on another clustering package or on PySAL `spaghetti` at runtime.

## Installation

From the repository:

```powershell
python -m pip install -e ".[dev]"
```

The distribution name is `net-dbscan`, the import package is `net_dbscan`, and the command is `net-dbscan`.

Core dependencies are GeoPandas, NumPy, pandas, Pyogrio, SciPy, scikit-learn, and Shapely. Optional extras are:

- `files`: PyArrow for GeoParquet/Parquet I/O;
- `cli`: PyArrow + Typer for the command-line interface;
- `dev`: PyArrow + Typer + pytest;
- `reference`: optional `spaghetti` 1.x only for historical/reference parity work. It is not imported by the runtime.

## CLI

Single run:

```powershell
net-dbscan cluster `
  --points "data\points.parquet" `
  --network "data\roads.gpkg" `
  --network-layer roads `
  --point-id-col canonical_id `
  --eps 1000 `
  --min-samples 5 `
  --output-dir "output\dbscan"
```

Grouped run:

```powershell
net-dbscan cluster `
  --points "data\points.parquet" `
  --network "data\roads.gpkg" `
  --network-layer roads `
  --point-id-col canonical_id `
  --group-col io80_map_code `
  --group-universe "data\io80_universe.csv" `
  --missing-group-policy exclude `
  --eps 500 `
  --min-samples 5 `
  --output-dir "output\dbscan_io80"
```

`--vertex-digits` controls the significant-digit rounding used to identify shared network vertices. The default is `11`; this makes graph construction deterministic while tolerating tiny coordinate noise. Set `vertex_digits=None` only through the Python API when exact coordinate identity is required.

## DBSCAN parameters

`eps` is the DBSCAN neighbourhood radius in the linear units of the projected network CRS. It is also the exact maximum network distance searched by the sparse neighbour engine. Changing `eps` changes the DBSCAN model; it is not an adaptive computational horizon.

`min_samples` counts observations, including multiplicity when several observations snap to the same network position.

`border_policy` controls only non-core observations that lie within `eps` of a core cluster:

- `expansion` — standard scikit-learn DBSCAN expansion order;
- `nearest_core` — assign each border observation to the cluster of its nearest core position by network distance;
- `core_only` — leave every border observation unassigned/noise (DBSCAN*).

Core status and the connected components of core positions are identical under all three policies.

## Network semantics

Each LineString is split into straight arcs between consecutive vertices. MultiLineStrings are exploded into independent parts before graph construction.

Two network features connect **only when they share a vertex after `vertex_digits` rounding**. A geometric crossing without a shared vertex remains disconnected, which is appropriate for cases such as a bridge crossing another road.

Every observation is snapped to the nearest point on the nearest arc. Ties between equally near arcs are broken by canonical arc order, so snapping is deterministic under network-row reordering.

For observations `i` and `j`, with snapped positions `s_i` and `s_j`, the clustering metric is

```text
d(i, j) = shortest network distance from s_i to s_j.
```

Point-to-network snap distance is not added to the clustering metric. It is retained as QA metadata. Observations on disconnected network components are never neighbours.

## Sparse neighbour search

The package does not build an all-pairs distance matrix. It searches only network paths that can produce position pairs within `eps`, using spatially local SciPy shortest-path batches and hard-bounded candidate chunks.

`max_neighbor_pairs` is a safety limit on stored unordered pairs. The default is 10,000,000. If a run exceeds it, the package stops with a contextual error instead of allowing an uncontrolled allocation.

Repeated observations at exactly the same snapped network position are compressed into one position with integer multiplicity and passed to scikit-learn as `sample_weight`. This preserves ordinary DBSCAN `min_samples` semantics without materialising duplicate zero-distance pairs.

## Grouped runs

With `--group-col`, groups are clustered independently while the network is prepared once and the retained points are snapped once.

Null and blank groups are excluded by default. Use `--missing-group-policy include` to treat them as explicit groups or `error` to stop the run.

`--group-universe` accepts a one-column CSV of expected non-missing categories. Declared-but-absent groups still receive zero-row outputs. Observed groups outside the universe are processed and flagged.

`--group-params` accepts `eps`, `min_samples`, `noise_policy`, and `border_policy` overrides. Example:

```text
io80_map_code,eps,min_samples,border_policy
31,5000,5,expansion
55,1500,5,nearest_core
63,1500,5,
```

The reserved spellings `__null__` and `__blank__` target actual missing groups in parameter files. To target a literal string with one of those names, use `literal:__null__` or `literal:__blank__`.

## Outputs

A single run writes:

```text
clustered_points.parquet
cluster_table.parquet
summary.csv
manifest.json
```

A grouped run writes:

```text
points/group_<value>.parquet
clusters/group_<value>.parquet
summary.csv
manifest.json
```

Point output preserves the input fields and adds:

| field | meaning |
|---|---|
| `cluster_id` | stable public ID (`C000001`, ...); null for noise unless singleton IDs are requested |
| `is_noise` | whether the final selected border policy leaves the observation unassigned |
| `is_core` | whether the snapped position satisfies the DBSCAN core criterion |
| `is_border` | intrinsic DBSCAN border status: non-core but within `eps` of at least one core cluster |
| `n_candidate_clusters` | number of core clusters within `eps`; values `>=2` identify contested border observations |
| `snap_distance` | straight-line distance from the original point to its snapped network position |
| `snapped_x`, `snapped_y` | snapped coordinates in the network CRS |

`is_border` is independent of assignment policy. Under `core_only`, for example, a point can have `is_border=True` and `is_noise=True`.

`cluster_table.parquet` contains one row per assigned DBSCAN cluster with point, position, core, and assigned-border counts. Noise singleton IDs created by `noise_policy="singleton"` are downstream labels, not DBSCAN clusters, and are not included in this table.

`summary.csv` records point/position counts, neighbour-pair counts, network-component and snapping diagnostics, cluster/noise/core/intrinsic-border shares, `contested_share`, parameters, group metadata, and runtime.

`manifest.json` records input files and selected layers, package version, parameters, `vertex_digits`, CRS and distance unit, graph size, and the Shapely/SciPy/scikit-learn backend versions.

## Python API

```python
import geopandas as gpd
from net_dbscan import DBSCANConfig, cluster_geodataframes

out = cluster_geodataframes(
    points=gpd.read_parquet("points.parquet"),
    network=gpd.read_file("roads.gpkg"),
    config=DBSCANConfig(eps=500, min_samples=5),
    point_id_col="canonical_id",
)

out.points
out.clusters
out.summary
out.analysis
```

Lower-level public functions include `build_network_graph`, `snap_points`, `distinct_positions`, `neighbor_graph`, and `dbscan`. `RoadGraph`, `build_road_graph`, and `NetDBSCANConfig` remain compatibility aliases for earlier releases; new code should use the generic names.

## Validation and determinism

The lower-level `dbscan()` API validates sparse distance graphs before clustering. It rejects malformed shape, negative/non-finite values, distances beyond `eps`, duplicate stored entries, nonzero stored self-distances, and asymmetric relations unless symmetry checking is explicitly skipped for a graph known to be symmetric by construction.

Network vertices and arcs are canonicalized by geometry, nearest-arc ties are deterministic, and distinct snapped positions are numbered from network location rather than input row order or point IDs. Public cluster IDs are then numbered from canonical point-ID order. Renaming IDs can therefore change which component is called `C000001`, but not the underlying partition.

Grouped output filenames are preflighted case-insensitively so values such as `A` and `a` cannot overwrite one another on Windows.

## Scope

`net-dbscan` clusters every supplied point. Study-area filtering or other eligibility filtering belongs upstream of the package.

`net-dbscan` does not download networks, repair uncertain topology, infer intersections that are absent from the source geometry, choose the scientifically appropriate `eps`, or construct service/Voronoi territories.

## License

MIT.
