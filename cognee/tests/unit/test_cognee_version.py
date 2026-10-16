"""Unit tests for local-version detection from a nearby pyproject.toml.

Guards the bug these tests exist for: ``get_cognee_version()`` and
``_get_cognee_version()`` used to read whatever ``pyproject.toml`` sits at the
package root and take its ``version`` line at face value. The bundled
cognee-mcp setup (docker-compose.yml) mounts the cognee package at
``/app/cognee`` right next to the image's own ``/app/pyproject.toml`` — which
is cognee-mcp's, ``version = "0.5.7"`` — so those deployments reported
``0.5.7-local`` as the cognee version in telemetry, logs, the CLI and the UI,
hiding their real failure rate under a version label that never shipped
(issue #5807).

A nearby pyproject.toml counts as cognee's only when its project name is
exactly ``cognee``; anything else falls through to the installed-package
metadata.
"""

import importlib.metadata
import importlib.metadata as importlib_metadata
from pathlib import Path

import cognee.version
from cognee.version import get_cognee_version

FOREIGN_PYPROJECT = """\\
[project]
name = "cognee-mcp"
version = "0.5.7"
"""

OWN_PYPROJECT = """\\
[project]
name = "cognee"
version = "9.9.9"
"""


def _use_fake_package_root(monkeypatch, tmp_path: Path, pyproject_body: str) -> None:
    """Place pyproject_body where both version readers look for it.

    ``cognee/version.py`` reads ``Path(__file__).parent.parent`` and
    ``cognee/shared/logging_utils.py`` reads ``Path(__file__).parent.parent.parent``;
    repointing each module's ``__file__`` moves that root to ``tmp_path``.
    """
    (tmp_path / "pyproject.toml").write_text(pyproject_body, encoding="utf-8")
    from cognee.shared import logging_utils

    monkeypatch.setattr(cognee.version, "__file__", str(tmp_path / "cognee" / "version.py"))
    monkeypatch.setattr(
        logging_utils,
        "__file__",
        str(tmp_path / "cognee" / "shared" / "logging_utils.py"),
    )


def _installed_version(monkeypatch, version: str) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda dist: version)


def test_own_nearby_pyproject_version_is_used(monkeypatch, tmp_path):
    """cognee's own pyproject.toml still yields '<version>-local'."""
    _use_fake_package_root(monkeypatch, tmp_path, OWN_PYPROJECT)
    _installed_version(monkeypatch, "1.2.3")

    assert get_cognee_version() == "9.9.9-local"


def test_foreign_nearby_pyproject_is_ignored(monkeypatch, tmp_path):
    """A neighbour's pyproject.toml must not become the cognee version.

    This is the cognee-mcp shape from issue #5807: its pyproject.toml sits next
    to the mounted cognee package, and its 0.5.7 used to be reported as
    ``0.5.7-local`` instead of the installed cognee version.
    """
    _use_fake_package_root(monkeypatch, tmp_path, FOREIGN_PYPROJECT)
    _installed_version(monkeypatch, "1.2.3")

    assert get_cognee_version() == "1.2.3"
    assert get_cognee_version() != "0.5.7-local"


def test_foreign_nearby_pyproject_without_installed_cognee_reports_unknown(monkeypatch, tmp_path):
    """No local pyproject and no installed cognee -> honest 'unknown'."""

    def _raise(dist):
        raise importlib_metadata.PackageNotFoundError(dist)

    _use_fake_package_root(monkeypatch, tmp_path, FOREIGN_PYPROJECT)
    monkeypatch.setattr(importlib.metadata, "version", _raise)

    assert get_cognee_version() == "unknown"


def test_nearby_pyproject_without_version_line_falls_through(monkeypatch, tmp_path):
    """A nameless/versionless pyproject.toml is ignored, not a crash."""
    _use_fake_package_root(monkeypatch, tmp_path, '[project]\nname = "cognee"\n')
    _installed_version(monkeypatch, "1.2.3")

    assert get_cognee_version() == "1.2.3"


def test_logging_version_ignores_foreign_nearby_pyproject(monkeypatch, tmp_path):
    """The logging bootstrap carries the same rule as cognee.version."""
    from cognee.shared import logging_utils

    _use_fake_package_root(monkeypatch, tmp_path, FOREIGN_PYPROJECT)
    _installed_version(monkeypatch, "1.2.3")

    assert logging_utils._get_cognee_version() == "1.2.3"
    assert logging_utils._get_cognee_version() != "0.5.7-local"


def test_logging_version_uses_own_nearby_pyproject(monkeypatch, tmp_path):
    from cognee.shared import logging_utils

    _use_fake_package_root(monkeypatch, tmp_path, OWN_PYPROJECT)
    _installed_version(monkeypatch, "1.2.3")

    assert logging_utils._get_cognee_version() == "9.9.9-local"
