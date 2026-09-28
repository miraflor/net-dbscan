# Review notes for net-dbscan 0.4.0

## Scope

This release hardens the existing DBSCAN implementation rather than changing the clustering model. Standard scikit-learn DBSCAN remains the default method, `eps` remains a fixed model parameter, and the existing `expansion`, `nearest_core`, and `core_only` border policies are retained.

## Main architectural change

The runtime no longer depends on PySAL `spaghetti`. Version 0.4.0 constructs the network directly from line vertices, canonicalizes vertices/arcs by geometry, snaps points with a Shapely STRtree, and uses SciPy sparse shortest paths. This removes the deprecated `libpysal.cg`/KDTree warning path and the previous `<2` compatibility ceiling.

`spaghetti` is deliberately not installed by the normal `dev` extra. An optional `reference` extra exists only for historical parity experiments.

## Correctness hardening

The lower-level sparse DBSCAN interface now rejects duplicate stored sparse coordinates before CSR conversion, because SciPy would otherwise add duplicate distance values. It also rejects nonzero self-distances and boolean/non-integer position weights.

Border status is now represented explicitly as `is_border`. It is intrinsic to the radius graph rather than inferred from whether a point received a cluster label. This fixes the earlier diagnostic inconsistency in `core_only`, where structural border points were assigned noise and therefore disappeared from `border_share`.

Missing-group internal keys cannot collide with ordinary strings named `__null__` or `__blank__`. Planned group output filenames are checked using case-folded names so Windows cannot silently overwrite `group_A.parquet` with `group_a.parquet`.

## Network semantics

The graph joins line features only at shared vertices after optional significant-digit rounding. It does not infer a junction merely because two lines cross geometrically. MultiLineString parts are treated independently. Equidistant snapping ties are resolved by canonical arc order, which is invariant to input row order.

The bounded neighbour engine includes explicit slow paths for a single source or same-arc run that exceeds the normal internal chunk target. Those paths preserve the chunk bound rather than allowing a one-item exception to allocate an arbitrarily large temporary array.

## Operational hardening

File entry points fail immediately if PyArrow is unavailable, before potentially expensive reads, graph construction, or clustering. Manifests now record selected layers, CRS/unit, graph size, `vertex_digits`, and backend versions.

## Verification in the build sandbox

The repository contains 59 collected tests. In the build sandbox, 57 pass and 2 file-I/O tests are skipped because PyArrow is unavailable. The skipped tests are expected to run in the normal `.[dev]` environment, where PyArrow is installed. The wheel builds and imports without `spaghetti` installed.

The final release candidate should still be run on the existing Quezon City grouped dataset to compare the 0.4.0 Shapely/SciPy backend with the saved 0.3.0 output before pushing the tag.
