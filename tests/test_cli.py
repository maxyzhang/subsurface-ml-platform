from pathlib import Path
import subprocess

import pytest

import subsurface_ml.cli as cli


def test_resolve_script():
    path = cli.resolve_script("train")

    assert path.name == "train_baselines.py"


def test_invalid_command():
    with pytest.raises(ValueError):
        cli.resolve_script("abc")


def test_build_parser():
    parser = cli.build_parser()

    args = parser.parse_args(["train"])

    assert args.command == "train"


def test_run_command(monkeypatch):
    called = {}

    def fake_run(cmd, cwd, check):
        called["cmd"] = cmd
        called["cwd"] = cwd

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    rc = cli.run_command("train")

    assert rc == 0
    assert "train_baselines.py" in called["cmd"][1]


def test_main(monkeypatch):
    monkeypatch.setattr(
        cli,
        "run_command",
        lambda command: 0,
    )

    rc = cli.main(["prepare"])

    assert rc == 0