# Algorithm comparison: DBSCAN and HDBSCAN*

This note is conceptual only. `net-dbscan` is an independent package and has no runtime dependency on another clustering repository.

Both density-based methods can operate on a shortest-path network metric, but their radius parameters mean different things.

| Concern | DBSCAN | HDBSCAN* |
|---|---|---|
| Primary density control | fixed `eps` + `min_samples` | hierarchy over density scales |
| Search radius in this package | exactly `eps` | implementation-dependent truncation horizon |
| Adaptive radius | would change the DBSCAN model | can be a computational strategy when the hierarchy is unchanged |
| Flat output | direct DBSCAN partition | selection from a hierarchy |
| Border assignment | standard expansion or explicit alternatives | hierarchy/membership semantics |

For DBSCAN, `eps` is part of the statistical definition of the clustering. `net-dbscan` therefore never increases it automatically merely to make a computation converge.

Version 0.4.0 uses a direct Shapely/SciPy spatial-network engine for deterministic topology, snapping, and bounded shortest-path search, with scikit-learn remaining the DBSCAN engine.
