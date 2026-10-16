import importlib.metadata
from pathlib import Path


def _local_pyproject_version(pyproject_path: Path) -> str | None:
    """Return the version from a nearby pyproject.toml — but only cognee's own.

    Source checkouts can sit next to another project's pyproject.toml: the
    cognee-mcp image (docker-compose.yml) mounts the cognee package at
    /app/cognee right next to the image's own /app/pyproject.toml, and taking
    that file's version line at face value reported cognee-mcp's version as
    cognee's (issue #5807). A nearby pyproject.toml therefore only counts when
    its project name is exactly "cognee"; anything else falls through to the
    installed-package metadata.
    """
    try:
        pyproject = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return None

    name = None
    version = None
    for line in pyproject.splitlines():
        stripped = line.strip()
        if name is None and stripped.startswith("name") and "=" in stripped:
            name = stripped.split("=", 1)[1].strip("'\"\n ")
        elif version is None and stripped.startswith("version") and "=" in stripped:
            version = stripped.split("=", 1)[1].strip("'\"\n ")
        if name is not None and version is not None:
            break
    if name == "cognee" and version:
        return version
    return None


def get_cognee_version() -> str:
    """Returns either the version of installed cognee package or the one
    found in nearby pyproject.toml"""
    version = _local_pyproject_version(Path(__file__).parent.parent / "pyproject.toml")
    if version is not None:
        # Mark the version as a local Cognee library by appending “-dev”
        return f"{version}-local"
    try:
        return importlib.metadata.version("cognee")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
