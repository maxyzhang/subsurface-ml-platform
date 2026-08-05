from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "default.yaml"
)


def load_yaml_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load a YAML configuration file."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration root must be a mapping."
        )

    return config


def get_config_value(
    config: dict[str, Any],
    *keys: str,
) -> Any:
    """Read a required nested configuration value."""

    current: Any = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            path = ".".join(keys)

            raise KeyError(
                f"Missing required configuration value: {path}"
            )

        current = current[key]

    return current