# net-dbscan design

## Purpose

`net-dbscan` implements ordinary DBSCAN on geospatial point observations when neighbourhood distance is shortest-path distance along a supplied spatial network.

The package is standalone. Its runtime stack is GeoPandas/Shapely for spatial data and geometry, SciPy for sparse graph operations and bounded shortest paths, and scikit-learn for DBSCAN.

## Pipeline

```text
input points + projected line network
  → validate/reproject
  → construct deterministic network graph
  → snap points to network arcs
  → collapse identical snapped positions with multiplicity
  → materialise only position pairs with network distance <= eps
  → scikit-learn DBSCAN with sample weights
  → expand labels/diagnostics back to observations
```

The clustering package does not define a study-area boundary. Callers that need geographic eligibility filtering should filter the input points before calling `net-dbscan`.

## Network topology

Every LineString is split at its existing vertices. MultiLineStrings are treated as separate parts. Vertices are canonicalized by coordinate after optional significant-digit rounding (`vertex_digits=11` by default).

Two arcs connect only through a shared canonical vertex. Geometric crossings without a shared vertex are not noded automatically.

Arcs are undirected and weighted by their geometric length in the projected network CRS. Directed routing, turn restrictions, and non-length impedance are outside the current scope.

## Snapping

Points are snapped to the nearest point of the nearest canonical arc using a Shapely STRtree. When several arcs are exactly equally near, the smallest canonical arc index is selected. This makes snapping invariant to network-row order.

The straight-line snap distance is QA metadata and is not included in DBSCAN distance.

## Position compression

Observations at exactly the same snapped network position are represented by one position with integer multiplicity. Vertex positions reached through different incident arcs are one position; coincident interior points on different arcs remain distinct because the arcs may cross without connecting.

Position numbering depends on canonical network location, not input row order or point IDs.

## Neighbour graph

DBSCAN needs only pairs with distance at most `eps`. The neighbour engine therefore searches bounded local subgraphs rather than constructing an all-pairs matrix.

For each snapped position, shortest paths can leave through either endpoint of its containing arc. Spatial batching restricts SciPy Dijkstra searches to network vertices that can possibly participate in paths of length `<= eps`. Direct same-arc distances are handled explicitly and merged with paths through vertices.

Candidate arrays and same-arc batches have hard internal bounds. `max_neighbor_pairs` provides a second, user-visible safety limit on the total stored unordered pairs.

The returned sparse radius graph is symmetric and omits the diagonal. The DBSCAN layer adds explicit zero self-distances before passing the graph to scikit-learn.

## DBSCAN semantics

`eps` is both the model neighbourhood radius and the shortest-path search radius. It is never adapted automatically.

Multiplicity at a compressed position is passed as scikit-learn `sample_weight`, preserving observation-level `min_samples` semantics.

The lower-level DBSCAN API validates externally supplied sparse radius graphs, including shape, finite/nonnegative distances, radius compliance, duplicate entries, zero diagonal, and symmetry.

## Core, border, and noise

Core status is intrinsic to the radius graph and multiplicities. A border position is non-core but lies within `eps` of at least one core cluster. This classification is stored separately from the final assignment policy.

`border_policy` supports:

- `expansion`: standard DBSCAN expansion;
- `nearest_core`: assign to the nearest reachable core cluster;
- `core_only`: leave all border positions as noise (DBSCAN*).

Thus under `core_only`, an observation may simultaneously have `is_border=True` and `is_noise=True`.

## Determinism

Network vertices and arcs are canonicalized by geometry. Nearest-arc ties use canonical arc order. Distinct positions are numbered by network location. These choices make the computational clustering invariant to input network-row order, point-row order, and point-ID renaming, apart from the human-readable public cluster IDs.

Public IDs (`C000001`, ...) are numbered after sorting observations by point ID text. Renaming IDs may therefore rename clusters without changing the partition.

## Grouped processing

Grouped runs build the graph and snap the retained points once, then cluster each group independently. Null and blank groups use internal sentinel values that cannot collide with literal strings such as `"__null__"` and `"__blank__"`.

Output filename collisions are checked with case-folded names before any group files are written, protecting case-insensitive filesystems such as typical Windows installations.

## File I/O

PyArrow is optional because the in-memory API does not require it. File-based clustering always writes Parquet, so those entry points check for PyArrow before reading inputs or doing expensive graph work.

The manifest records input paths and selected layers, package/configuration data, `vertex_digits`, CRS and linear unit, graph counts, and backend versions.
