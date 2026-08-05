from pathlib import Path

import pytest

from subsurface_ml.settings import (
    get_config_value,
    load_yaml_config,
)


def test_load_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
project:
  random_state: 42

training:
  random_forest:
    n_estimators: 100
    max_depth: null
""".strip(),
        encoding="utf-8",
    )

    config = load_yaml_config(config_path)

    assert config["project"]["random_state"] == 42
    assert (
        config["training"]["random_forest"]["n_estimators"]
        == 100
    )
    assert (
        config["training"]["random_forest"]["max_depth"]
        is None
    )


def test_load_yaml_config_returns_empty_dict_for_empty_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_yaml_config(config_path) == {}


def test_load_yaml_config_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Configuration file",
    ):
        load_yaml_config(missing_path)


def test_load_yaml_config_rejects_non_mapping_root(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid.yaml"

    config_path.write_text(
        """
- first
- second
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="root must be a mapping",
    ):
        load_yaml_config(config_path)


def test_get_config_value() -> None:
    config = {
        "training": {
            "random_forest": {
                "n_estimators": 100,
            }
        }
    }

    result = get_config_value(
        config,
        "training",
        "random_forest",
        "n_estimators",
    )

    assert result == 100


def test_get_config_value_rejects_missing_key() -> None:
    config = {
        "training": {},
    }

    with pytest.raises(
        KeyError,
        match="training.random_forest.n_estimators",
    ):
        get_config_value(
            config,
            "training",
            "random_forest",
            "n_estimators",
        )