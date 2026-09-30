# Changelog

## 0.5.0 — 2026-09-30

### Changed

- Breaking: remove study-area boundary filtering from the CLI and Python/file APIs. `net-dbscan` now clusters every supplied point; geographic eligibility filtering belongs upstream.
- Add a visual algorithm guide covering network distance, ε-neighbourhoods, intrinsic core/border/noise status, core connectivity, the three border policies, sparse shortest-path search, and weighted duplicate positions.
- Standardize the GitHub Pages site under `docs/`.
- Keep the DBSCAN model, grouped execution, deterministic Shapely/SciPy network engine, sparse-neighbour safeguards, border policies, diagnostics, and output structure otherwise unchanged.

## 0.4.0

### Changed

- Replaced the runtime PySAL `spaghetti` backend with a standalone Shapely/SciPy network engine. Network construction, canonical vertex numbering, nearest-arc snapping, and bounded shortest-path search are now owned by `net-dbscan`. `spaghetti` is available only through the optional `reference` extra for historical parity work.
- Added deterministic nearest-arc tie breaking by canonical arc index and explicit `vertex_digits` control (11 significant digits by default) for shared-vertex identification.
- Added hard-bounded handling for unusually dense candidate joins and same-arc pairs.
- Added intrinsic `is_border` output. `border_share` now measures structural DBSCAN border status independently of the selected assignment policy, so `core_only` can correctly report border points that are left as noise.
- Renamed the diagnostic `road_components_used` to `network_components_used`; user-facing errors now refer to the generic spatial network rather than roads.
- Manifests now record selected input layers, CRS, distance unit, graph vertex/arc/component counts, `vertex_digits`, and Shapely/SciPy/scikit-learn backend versions.
- File-based entry points check for PyArrow before reading inputs or starting network work.
- Group handling now uses collision-proof internal sentinels for null/blank groups; literal `__null__` and `__blank__` group strings remain ordinary values and can be targeted with the `literal:` escape in parameter files.
- Group output filenames are checked case-insensitively before writing, preventing collisions such as `A` versus `a` on Windows.
- Per-group parameter documentation now includes `border_policy` and `noise_policy`.

### Validation

- Sparse DBSCAN input validation now rejects duplicate stored entries before CSR conversion can sum them, rejects nonzero stored self-distances, and rejects boolean/non-integer multiplicity arrays.
- Added tests for topology semantics, vertex rounding, deterministic equidistant snapping, malformed sparse graphs, intrinsic border semantics, missing-group sentinel collisions, Windows-safe filenames, early PyArrow failure, and hard-bounded internal chunks.
- Existing road-row, point-row, and point-ID invariance tests continue to run against the standalone backend.

### Compatibility

- `DBSCANConfig`, `NetDBSCANConfig`, `NetworkGraph`, `RoadGraph`, `build_network_graph`, and `build_road_graph` remain available. The generic names are preferred for new code.
- Standard `border_policy="expansion"` remains the default, scikit-learn remains the DBSCAN engine, and there is still no adaptive `eps`.

## 0.3.0

### Fixed

- Results no longer depend on the row order of the network layer. spaghetti numbers vertices in the order in which it reads the lines, and 0.2.0 numbered snapped positions by those vertex IDs; the position order decides which cluster standard DBSCAN gives a contested border point (a non-core point within `eps` of core points of two or more clusters). Shuffling the rows of a road layer therefore moved contested border points between clusters. Vertices are now renumbered by coordinates (x, then y) right after spaghetti builds the network, and snapping results are translated to the new numbers. Core points, noise and the clusters of core points were never affected.
- Snapped positions no longer depend on the row order of the network layer either. spaghetti still chooses the arc, but it measures the position on the arc from whichever end its own numbering puts first, so the last bits of positions and offsets changed with the row order (15 of 4,850 points on a synthetic province, by at most 2.7e-12 m). The position is now recomputed from the renumbered arc ends; the run stops with an error if the recomputed offset differs from spaghetti's by more than a millionth of the arc length.

### Changed (not compatible with 0.2.0)

- The DBSCAN module is now `net_dbscan.sparse_dbscan` (it was `net_dbscan.dbscan`). In 0.2.0 the function `dbscan` replaced the module of the same name as an attribute of the package, so `import net_dbscan.dbscan as m` returned the function. `from net_dbscan import dbscan` is unchanged.
- The single-run cluster table is written as `cluster_table.parquet` (it was `clusters.parquet`), as in `net-hdbscan`. Grouped runs still write `clusters/group_<value>.parquet`.
- PyArrow and Typer are optional extras: `files` (PyArrow), `cli` (PyArrow and Typer), `dev` (both plus pytest). Functions that need a missing extra stop with a message naming it; the command is started through a small entry module so that a missing Typer gives the same kind of message.
- Contested border points may be assigned differently from 0.2.0 when the road layer was not already in coordinate order, because of the fix above.

### Added

- `border_policy` (`--border-policy`): `expansion` (default, standard DBSCAN), `nearest_core` (the cluster of the nearest core position by network distance), or `core_only` (DBSCAN*: border points are noise). Core points and their clusters are the same under all three. It can also be set per group in the group parameter file.
- Point column `n_candidate_clusters` and summary value `contested_share`, which show how many observations the border rule affects.
- The boundary is optional (`boundary=None`, `boundary_path=None`, or no `--boundary`).
- Neutral names `NetworkGraph` and `build_network_graph`; `RoadGraph` and `build_road_graph` remain as aliases.
- `python -m net_dbscan` runs the command.

### Faster

- The pipeline skips the symmetry check of the neighbour graph, which it builds symmetric by construction. The check sorts every stored entry and took 1.4 of the 3.4 s of the DBSCAN step on 6.8 million pairs. `dbscan()` still checks by default (`check_symmetry=True`) for graphs from other sources.
- Neighbourhood weights are computed with one sparse matrix product instead of a Python loop over positions.
- Candidate clusters are counted only for non-core positions (a core position always has one, its own cluster), so the new count adds little time. On 6.8 million pairs the DBSCAN step took 2.28 s instead of 3.64 s in 0.2.0, with identical labels.

### Tests

- 16 new tests (45 in total): the three border policies on a hand-made contested point (including how reversing the order changes only the standard rule); network-row-order invariance and point-ID renaming through the pipeline on data with known contested points (the row-order test fails on the 0.2.0 code); border policies through the pipeline; network distances against a brute-force shortest-path reference on a network with dead ends and detours; snapped positions after shuffling the rows of a road layer whose segment lengths are not round numbers; the module path; the name aliases; the optional boundary and the cluster table file; the messages for missing extras; `python -m net_dbscan`; and a command-line run. The version test now compares with `net_dbscan.__version__` instead of a fixed string.

## 0.2.0

Renamed and reorganized `netdbscan` as `net-dbscan` / `net_dbscan`.

Main changes:

- reorganized the source tree to parallel `net-hdbscan` (`config.py`, `network.py`, `dbscan.py`, `pipeline.py`, `io.py`, `cli.py`);
- retained PySAL `spaghetti` for network construction and snapping;
- retained standard scikit-learn DBSCAN rather than reimplementing the clustering algorithm;
- adopted deterministic network-location numbering of distinct snapped positions;
- shared network construction and snapping across grouped runs;
- excluded null/blank groups are filtered before snapping, avoiding unnecessary network work;
- added `--missing-group-policy` with default `exclude`;
- added `--group-universe` for explicit absent categories;
- added per-group `eps` / `min_samples` overrides;
- added `clusters.parquet`, `summary.csv`, and `manifest.json` outputs;
- added group-level failure context;
- added input and inside-boundary missing-group counts;
- changed singleton-noise IDs to one ID per snapped position;
- unified the CLI as `net-dbscan cluster` with optional `--group-col`;
- retained `NetDBSCANConfig` as an alias for migration, while making `DBSCANConfig` the preferred name.

DBSCAN semantics remain standard: `eps` is the actual clustering neighbourhood radius and is never adaptively enlarged by the package.
