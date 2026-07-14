"""Keep the documentation version synchronized with project metadata."""

from collections.abc import MutableMapping
from pathlib import Path

import tomllib

VERSION_PLACEHOLDER = "{{ trimwise_version }}"


def _project_version() -> str:
    """Read the single authoritative package version from ``pyproject.toml``.

    Returns:
        Static project version declared for the package build.
    """
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with project_file.open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


PROJECT_VERSION = _project_version()


def on_config(config: MutableMapping[str, object]) -> MutableMapping[str, object]:
    """Add the current package version to the site title.

    Args:
        config: Mutable MkDocs configuration.

    Returns:
        Configuration containing the versioned site title.
    """
    config["site_name"] = f"Trimwise {PROJECT_VERSION}"
    return config


def on_page_markdown(markdown: str, **_: object) -> str:
    """Replace documentation version placeholders before rendering.

    Args:
        markdown: Source Markdown for one documentation page.
        **_: Additional values supplied by the MkDocs hook contract.

    Returns:
        Markdown containing the current project version.
    """
    return markdown.replace(VERSION_PLACEHOLDER, PROJECT_VERSION)
