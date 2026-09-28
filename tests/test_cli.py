from typer.testing import CliRunner

from net_dbscan import __version__
from net_dbscan.cli import app


def test_version():
    r=CliRunner().invoke(app,["--version"])
    assert r.exit_code == 0
    assert r.stdout.strip() == __version__
