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

## Network semantics

Each LineString is split into straight arcs between consecutive vertices. MultiLineStrings are exploded into independent parts before graph construction.

Two network features connect **only when they share a vertex after `vertex_digits` rounding**. A geometric crossing without a shared vertex remains disconnected, which is appropriate for cases such as a bridge crossing another road.

Every observation is snapped to the nearest point on the nearest arc. Ties between equally near arcs are broken by canonical arc order, so snapping is deterministic under network-row reordering.

For observations $`i`$ and $`j`$, with snapped positions $`s_i`$ and $`s_j`$, the clustering metric is

```math
d(i, j) = d_N(s_i, s_j), \qquad s_i = \arg\min_{z \in N} \lVert x_i - z \rVert,
```

where $`d_N`$ is the shortest network distance, $`x_i`$ is the original point, and $`N`$ is the set of all locations on the network. The snap distance of observation $`i`$ is $`\lVert x_i - s_i \rVert`$, and $`d(i, j) = \infty`$ when $`s_i`$ and $`s_j`$ lie on different network components.

Point-to-network snap distance is not added to the clustering metric. It is retained as QA metadata. Observations on disconnected network components are never neighbours.

## DBSCAN parameters

`eps` is the DBSCAN neighbourhood radius in the linear units of the projected network CRS. It is also the exact maximum network distance searched by the sparse neighbour engine. Changing `eps` changes the DBSCAN model; it is not an adaptive computational horizon.

`min_samples` counts observations, including multiplicity when several observations snap to the same network position.

### The model in symbols

Observations at one snapped network position form one position $`p`$ with integer weight $`w_p`$, their number. The ε-neighbourhood of $`p`$ and its weight are

```math
N_\varepsilon(p) = \{\, q : d(p, q) \le \varepsilon \,\}, \qquad W(p) = \sum_{q \in N_\varepsilon(p)} w_q,
```

where the sum includes $`p`$ itself, because $`d(p, p) = 0`$.

- A position $`p`$ is a **core** position when $`W(p) \ge`$ `min_samples`.
- The DBSCAN clusters are the connected components of the core positions, where two core positions are connected when $`d(p, q) \le \varepsilon`$.
- A non-core position $`p`$ is a **border** position when $`N_\varepsilon(p)`$ contains at least one core position. `n_candidate_clusters` counts the distinct clusters of those core positions; for a core position it is 1.

### Border policies

`border_policy` controls only non-core observations that lie within `eps` of a core cluster:

- `expansion` — standard scikit-learn DBSCAN expansion order;
- `nearest_core` — assign each border observation to the cluster of its nearest core position by network distance;
- `core_only` — leave every border observation unassigned/noise (DBSCAN*).

Core status and the connected components of core positions are identical under all three policies.

With `nearest_core`, a border position $`p`$ takes the cluster of the core position

```math
q^*(p) = \arg\min_{q \in N_\varepsilon(p),\; q \text{ core}} \bigl(d(p, q),\ \mathrm{cluster}(q)\bigr),
```

where pairs are compared in order: the smallest network distance first, and for an exact tie between clusters, the smaller cluster label.

## Sparse neighbour search

The package does not build an all-pairs distance matrix. It searches only network paths that can produce position pairs within `eps`, using spatially local SciPy shortest-path batches and hard-bounded candidate chunks.

In symbols, the stored neighbour graph over distinct positions is exactly

```math
E_\varepsilon = \{\, (p, q) : p \ne q,\ d(p, q) \le \varepsilon \,\}.
```

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

## Provenance

`net-dbscan` implements published methods. It does not propose a new clustering algorithm. The clustering model is DBSCAN (Ester et al., 1996). The distance is the shortest-path length between positions on a network, as in earlier network-constrained density clustering (Yiu and Mamoulis, 2004; Wang et al., 2019). The DBSCAN step itself is scikit-learn's `DBSCAN` (Pedregosa et al., 2011), called with `metric="precomputed"` on the sparse network-distance graph and with `sample_weight` for co-located observations.

DBSCAN does not require Euclidean distance. The generalization by the original authors allows any symmetric predicate to define a neighbourhood (Ester et al., 1997; Sander et al., 1998). Shortest-path distance on the undirected network used by this package is symmetric, so the network ε-neighbourhood is such a predicate.

### Source of each step

| Step in `net-dbscan` | Published source or earlier implementation |
|---|---|
| Observations lie on network edges, and the distance is the shortest-path length along the network | Yiu and Mamoulis (2004) cluster objects that lie on the edges of a weighted spatial network, using shortest-path distance. |
| Each observation is snapped to the nearest point of the nearest arc | NS-DBSCAN moves each point to its nearest road segment and splits the segment at that location (Wang et al., 2019). An earlier public Python example attaches each point to its nearest network node instead (Boeing, 2018). |
| ε-neighbourhoods come from bounded shortest-path searches, without an all-pairs matrix | NS-DBSCAN finds ε-neighbours by expanding shortest paths from each point until the radius is reached (Wang et al., 2019). Shortest paths: Dijkstra (1959), computed with SciPy (Virtanen et al., 2020). |
| Core positions, clusters as connected components of core positions, and noise | DBSCAN (Ester et al., 1996), computed by scikit-learn `DBSCAN` with `metric="precomputed"`. |
| Co-located observations are compressed into one position with an integer weight | scikit-learn `DBSCAN` accepts `sample_weight`; a sample whose weight reaches `min_samples` is a core sample by itself. Core status is therefore the same as when each observation is repeated. |
| `border_policy="expansion"` | The original DBSCAN algorithm gives a border point that several clusters can reach to the cluster that reaches it first (Ester et al., 1996). In this package, scikit-learn's expansion order decides which cluster is first. |
| `border_policy="core_only"` | DBSCAN*, which leaves border points as noise (Campello et al., 2013). |
| `border_policy="nearest_core"` | Assigning an ambiguous border point to the cluster of its closest core point is a described way to resolve the ambiguity (Kapp-Joswig and Keller, 2022). Tran et al. (2013) revised DBSCAN so that border assignment does not depend on processing order. The rule for exact ties (the smaller cluster label) is a convention of this package. |

### What is specific to this package

The package-specific parts are engineering and conventions: canonical vertex and arc identity, deterministic tie-breaking in snapping, batching and memory limits in the neighbour search, validation of sparse distance graphs, grouped execution, and the diagnostic fields `is_border`, `n_candidate_clusters` and `contested_share`. None of them defines a new clustering model. Under every border policy, core status and the connected components of core positions are those of DBSCAN.

A different method has a similar name. NET-DBSCAN (Stefanakis, 2007) clusters the nodes of a linear network whose edges may be temporarily inaccessible. `net-dbscan` clusters point observations at any location along the edges of a fixed network, and it is not an implementation of NET-DBSCAN. Both methods extend DBSCAN to networks.

## References

Boeing, G. (2018, April). *Network-based spatial clustering* [Blog post]. https://geoffboeing.com/2018/04/network-based-spatial-clustering/

Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2013)*, Lecture Notes in Computer Science 7819, 160–172. Springer. https://doi.org/10.1007/978-3-642-37456-2_14

Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269–271.

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)*, 226–231. AAAI Press.

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1997). Density-connected sets and their application for trend detection in spatial databases. In *Proceedings of the Third International Conference on Knowledge Discovery and Data Mining (KDD-97)*, 10–15. AAAI Press.

Kapp-Joswig, J.-O. F., & Keller, B. G. (2022). *Clustering — basic concepts and methods*. arXiv:2212.01248. https://doi.org/10.48550/arXiv.2212.01248

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (1998). Density-based clustering in spatial databases: The algorithm GDBSCAN and its applications. *Data Mining and Knowledge Discovery*, 2(2), 169–194. https://doi.org/10.1023/A:1009745219419

Stefanakis, E. (2007). NET-DBSCAN: Clustering the nodes of a dynamic linear network. *International Journal of Geographical Information Science*. https://doi.org/10.1080/13658810601034226

Tran, T. N., Drab, K., & Daszykowski, M. (2013). Revised DBSCAN algorithm to cluster data with dense adjacent clusters. *Chemometrics and Intelligent Laboratory Systems*. https://doi.org/10.1016/j.chemolab.2012.11.006

Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2

Wang, T., Ren, C., Luo, Y., & Tian, J. (2019). NS-DBSCAN: A density-based clustering algorithm in network space. *ISPRS International Journal of Geo-Information*, 8(5), 218. https://doi.org/10.3390/ijgi8050218

Yiu, M. L., & Mamoulis, N. (2004). Clustering objects on a spatial network. In *Proceedings of the 2004 ACM SIGMOD International Conference on Management of Data*, 443–454. https://doi.org/10.1145/1007568.1007619

## License

MIT.
