from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMMAND_TO_SCRIPT = {
    "prepare": PROJECT_ROOT / "scripts" / "prepare_data.py",
    "train": PROJECT_ROOT / "scripts" / "train_baselines.py",
    "tune": PROJECT_ROOT / "scripts" / "tune_random_forest.py",
    "select": PROJECT_ROOT / "scripts" / "select_best_model.py",
    "predict": PROJECT_ROOT / "scripts" / "predict.py",
    "importance": PROJECT_ROOT
    / "scripts"
    / "plot_feature_importance.py",
    "dashboard": PROJECT_ROOT / "scripts" / "build_dashboard.py", }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="subsurface-ml",
        description=(
            "Run subsurface machine-learning workflow commands."
        ),
    )

    parser.add_argument(
        "command",
        choices=sorted(COMMAND_TO_SCRIPT),
        help="Workflow command to execute.",
    )

    return parser


def resolve_script(command: str) -> Path:
    """Resolve a CLI command to its script path."""

    try:
        script_path = COMMAND_TO_SCRIPT[command]
    except KeyError as error:
        raise ValueError(
            f"Unsupported command: {command}"
        ) from error

    if not script_path.exists():
        raise FileNotFoundError(
            f"Command script was not found: {script_path}"
        )

    return script_path


def run_command(
    command: str,
) -> int:
    """Run the script associated with a CLI command."""

    script_path = resolve_script(command)

    completed_process = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )

    return completed_process.returncode


def main(
    argv: list[str] | None = None,
) -> int:
    """Parse CLI arguments and execute the selected command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    return run_command(
        arguments.command
    )


if __name__ == "__main__":
    raise SystemExit(main())