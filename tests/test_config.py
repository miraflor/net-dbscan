from pathlib import Path

import pytest

from net_dbscan import DBSCANConfig, NetDBSCANConfig
from net_dbscan.config import read_group_params, read_group_universe, resolve_group_config


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_eps_must_be_positive_finite(value):
    with pytest.raises(ValueError):
        DBSCANConfig(eps=value)


def test_old_config_name_is_alias():
    assert NetDBSCANConfig is DBSCANConfig


def test_group_params(tmp_path: Path):
    path = tmp_path / "params.csv"
    path.write_text("kind,eps,min_samples\nA,2500,7\nB,,\n", encoding="utf-8")
    got = read_group_params(path, "kind")
    cfg = resolve_group_config(DBSCANConfig(eps=1000, min_samples=5), got, "A")
    assert cfg.eps == 2500
    assert cfg.min_samples == 7
    assert resolve_group_config(DBSCANConfig(eps=1000), got, "B").eps == 1000


def test_group_universe_preserves_order(tmp_path: Path):
    path = tmp_path / "groups.csv"
    path.write_text("kind\nB\nA\n", encoding="utf-8")
    assert read_group_universe(path, "kind") == ["B", "A"]
