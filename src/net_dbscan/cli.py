"""Command line interface for ``net-dbscan cluster ...``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from ._version import __version__
from .config import DBSCANConfig, read_group_params, read_group_universe
from .network import DEFAULT_MAX_NEIGHBOR_PAIRS, DEFAULT_VERTEX_DIGITS
from .pipeline import _summary_frame, cluster_files, cluster_files_by_column

app = typer.Typer(add_completion=False, no_args_is_help=True)

_SHOWN = [
    ("group", "group"),
    ("n_points", "points"),
    ("n_clusters", "clusters"),
    ("noise_share", "noise"),
    ("core_share", "core"),
    ("largest_cluster_share", "largest"),
    ("eps", "eps"),
    ("max_position_weight", "max stack"),
    ("seconds", "seconds"),
]


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show the version and exit."),
) -> None:
    """Cluster points with DBSCAN using shortest-path distance on a spatial network."""


def _print_summary(summary: pd.DataFrame) -> None:
    table = summary[[c for c, _ in _SHOWN]].copy()
    table.columns = [name for _, name in _SHOWN]
    for column in ["noise", "core", "largest"]:
        table[column] = table[column].map(lambda v: "" if pd.isna(v) else f"{100 * v:.1f}%")
    table["group"] = table["group"].map(lambda v: "(all)" if v is None or pd.isna(v) else str(v))
    typer.echo(table.to_string(index=False))


@app.command()
def cluster(
    points: Path = typer.Option(..., "--points", help="Point layer (GeoParquet, GeoPackage, ...)."),
    boundary: Path | None = typer.Option(None, "--boundary", help="Optional polygon layer; only covered points are used."),
    network: Path = typer.Option(..., "--network", help="Line network in a projected CRS."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Folder for clustered points and diagnostics."),
    eps: float = typer.Option(..., "--eps", help="DBSCAN neighbourhood radius in network-CRS units."),
    min_samples: int = typer.Option(5, "--min-samples", help="Minimum weighted observations in an eps-neighbourhood."),
    noise_policy: str = typer.Option("exclude", "--noise-policy", help="'exclude' or 'singleton' (one ID per noise position)."),
    border_policy: str = typer.Option(
        "expansion", "--border-policy", help="'expansion' (standard DBSCAN), 'nearest_core', or 'core_only' (DBSCAN*)."
    ),
    max_snap_distance: float | None = typer.Option(None, "--max-snap-distance", help="Fail if a point is farther from the network."),
    max_neighbor_pairs: int = typer.Option(DEFAULT_MAX_NEIGHBOR_PAIRS, "--max-neighbor-pairs", help="Safety limit on stored position pairs."),
    point_id_col: str = typer.Option("point_id", "--point-id-col"),
    group_col: str | None = typer.Option(None, "--group-col", help="Cluster each value of this column independently."),
    group_params: Path | None = typer.Option(None, "--group-params", help="CSV of per-group eps/min_samples/noise_policy/border_policy overrides."),
    group_universe: Path | None = typer.Option(None, "--group-universe", help="One-column CSV declaring expected group values."),
    missing_group_policy: str = typer.Option("exclude", "--missing-group-policy", help="'exclude', 'include', or 'error' for null/blank group values."),
    points_layer: str | None = typer.Option(None, "--points-layer"),
    boundary_layer: str | None = typer.Option(None, "--boundary-layer"),
    network_layer: str | None = typer.Option(None, "--network-layer"),
    vertex_digits: int = typer.Option(
        DEFAULT_VERTEX_DIGITS,
        "--vertex-digits",
        help="Significant digits used to identify shared network vertices.",
    ),
    force: bool = typer.Option(False, "--force", help="Replace existing outputs."),
) -> None:
    """Cluster points and write DBSCAN labels, cluster summaries and diagnostics."""
    config = DBSCANConfig(
        eps=eps,
        min_samples=min_samples,
        noise_policy=noise_policy,
        border_policy=border_policy,
        max_snap_distance=max_snap_distance,
        max_neighbor_pairs=max_neighbor_pairs,
    )
    common = dict(
        points_path=points,
        boundary_path=boundary,
        network_path=network,
        output_dir=output_dir,
        config=config,
        point_id_col=point_id_col,
        points_layer=points_layer,
        boundary_layer=boundary_layer,
        network_layer=network_layer,
        vertex_digits=vertex_digits,
        force=force,
    )
    if group_col is None:
        if group_params is not None:
            raise typer.BadParameter("--group-params requires --group-col")
        if group_universe is not None:
            raise typer.BadParameter("--group-universe requires --group-col")
        outputs = cluster_files(**common)
        summary = _summary_frame([outputs.summary])
    else:
        overrides = read_group_params(group_params, group_col) if group_params is not None else None
        universe = read_group_universe(group_universe, group_col) if group_universe is not None else None
        summary = cluster_files_by_column(
            group_col=group_col,
            group_params=overrides,
            group_universe=universe,
            missing_group_policy=missing_group_policy,
            **common,
        )

    _print_summary(summary)
    excluded = int(summary.attrs.get("n_missing_group_input", 0)) if hasattr(summary, "attrs") else 0
    inside_excluded = int(summary.attrs.get("n_missing_group_inside_boundary", 0)) if hasattr(summary, "attrs") else 0
    if excluded and summary.attrs.get("missing_group_policy") == "exclude":
        typer.echo(
            f"Excluded {excluded:,} input row(s) with null/blank group values "
            f"({inside_excluded:,} inside the boundary)."
        )
    outside = list(summary.attrs.get("groups_outside_universe") or [])
    if outside:
        shown = ", ".join(outside[:10]) + (" ..." if len(outside) > 10 else "")
        typer.echo(
            f"WARNING: {len(outside)} observed group(s) are not in --group-universe and were processed anyway "
            f"(see declared_in_universe in summary.csv): {shown}"
        )
    typer.echo(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    app()
