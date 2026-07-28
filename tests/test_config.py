from subsurface_ml.config import (
    FIGURE_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    ensure_project_directories,
)


def test_ensure_project_directories() -> None:
    ensure_project_directories()

    assert PROCESSED_DATA_DIR.exists()
    assert MODEL_DIR.exists()
    assert FIGURE_DIR.exists()