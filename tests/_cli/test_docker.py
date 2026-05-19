from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest

from inspect_ai._cli.docker import (
    DOCKER_REF_PREFIX,
    build_passthrough,
    dispatch_to_docker,
    env_passthrough_args,
    has_docker_ref,
    parse_cli_env,
    reject_docker_refs,
    run_container,
)


def _ctx(info_name: str = "eval") -> click.Context:
    ctx = MagicMock(spec=click.Context)
    ctx.info_name = info_name
    return ctx


def test_has_docker_ref_true_when_present() -> None:
    assert has_docker_ref(("docker://img", "task")) is True


def test_has_docker_ref_false_when_absent() -> None:
    assert has_docker_ref(("inspect_evals/mask",)) is False


def test_has_docker_ref_false_for_empty_tasks() -> None:
    assert has_docker_ref(()) is False


def test_has_docker_ref_false_for_none() -> None:
    assert has_docker_ref(None) is False


def test_dispatch_to_docker_rejects_multiple_docker_refs() -> None:
    with pytest.raises(click.UsageError, match="at most one"):
        dispatch_to_docker(_ctx(), ("docker://a", "docker://b"), "./logs", None)


def test_dispatch_to_docker_rejects_empty_image_ref() -> None:
    with pytest.raises(click.UsageError, match="Empty image"):
        dispatch_to_docker(_ctx(), (f"{DOCKER_REF_PREFIX}",), "./logs", None)


def test_dispatch_to_docker_runs_container_and_creates_log_dir(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    argv = ["inspect", "eval", "docker://img:1", "inspect_evals/mask"]
    with (
        patch.object(sys, "argv", argv),
        patch("inspect_ai._cli.docker.run_container", return_value=0) as run,
    ):
        rc = dispatch_to_docker(
            _ctx(),
            ("docker://img:1", "inspect_evals/mask"),
            str(log_dir),
            None,
        )
    assert rc == 0
    image, args, passed_log_dir, cli_env = run.call_args.args
    assert image == "img:1"
    assert args == ["inspect_evals/mask"]
    assert passed_log_dir == log_dir
    assert cli_env == {}
    assert log_dir.is_dir()


def test_dispatch_to_docker_forwards_env_pairs(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    argv = [
        "inspect",
        "eval",
        "docker://img:1",
        "task",
        "--env",
        "HUGGINGFACE_TOKEN=hf_abc",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch("inspect_ai._cli.docker.run_container", return_value=0) as run,
    ):
        dispatch_to_docker(
            _ctx(),
            ("docker://img:1", "task"),
            str(log_dir),
            ("HUGGINGFACE_TOKEN=hf_abc",),
        )
    _, _, _, cli_env = run.call_args.args
    assert cli_env == {"HUGGINGFACE_TOKEN": "hf_abc"}


def test_reject_docker_refs_raises_for_eval_set() -> None:
    with pytest.raises(click.UsageError, match="eval-set"):
        reject_docker_refs(("docker://a",), "eval-set")


def test_reject_docker_refs_raises_for_eval_retry() -> None:
    with pytest.raises(click.UsageError, match="eval-retry"):
        reject_docker_refs(("docker://a.eval",), "eval-retry")


def test_reject_docker_refs_noop_when_no_docker() -> None:
    reject_docker_refs(("inspect_evals/mask",), "eval-set")


def test_reject_docker_refs_noop_when_none() -> None:
    reject_docker_refs(None, "eval-set")


def test_build_passthrough_strips_docker_ref() -> None:
    with patch.object(sys, "argv", ["inspect", "eval", "docker://img", "t1", "t2"]):
        assert build_passthrough(_ctx(), "docker://img") == ["t1", "t2"]


def test_build_passthrough_strips_log_dir_space_form() -> None:
    argv = ["inspect", "eval", "--log-dir", "/x", "docker://img", "t"]
    with patch.object(sys, "argv", argv):
        assert build_passthrough(_ctx(), "docker://img") == ["t"]


def test_build_passthrough_strips_log_dir_equals_form() -> None:
    argv = ["inspect", "eval", "--log-dir=/x", "docker://img", "t"]
    with patch.object(sys, "argv", argv):
        assert build_passthrough(_ctx(), "docker://img") == ["t"]


def test_build_passthrough_preserves_other_flags() -> None:
    argv = [
        "inspect",
        "eval",
        "--model",
        "gpt-4o-mini",
        "docker://img",
        "t",
        "--limit",
        "1",
    ]
    with patch.object(sys, "argv", argv):
        assert build_passthrough(_ctx(), "docker://img") == [
            "--model",
            "gpt-4o-mini",
            "t",
            "--limit",
            "1",
        ]


def test_build_passthrough_handles_task_literally_named_eval() -> None:
    argv = ["inspect", "eval", "docker://img", "eval"]
    with patch.object(sys, "argv", argv):
        assert build_passthrough(_ctx(), "docker://img") == ["eval"]


def test_build_passthrough_asserts_on_wrong_subcommand() -> None:
    with pytest.raises(AssertionError):
        build_passthrough(_ctx("eval-set"), "docker://img")


def test_parse_cli_env_handles_value_with_equals() -> None:
    assert parse_cli_env(("FOO=bar=baz",)) == {"FOO": "bar=baz"}


def test_parse_cli_env_rejects_missing_value() -> None:
    with pytest.raises(click.UsageError, match="NAME=value"):
        parse_cli_env(("FOO",))


def test_parse_cli_env_empty_when_none() -> None:
    assert parse_cli_env(None) == {}


def test_env_passthrough_forwards_default_keys() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "k"}, clear=True):
        assert env_passthrough_args({}) == ["-e", "OPENAI_API_KEY"]


def test_env_passthrough_empty_when_no_keys_or_cli_env() -> None:
    with patch.dict("os.environ", {}, clear=True):
        assert env_passthrough_args({}) == []


def test_env_passthrough_cli_env_takes_priority_over_default() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "host"}, clear=True):
        out = env_passthrough_args({"OPENAI_API_KEY": "cli"})
    assert out == ["-e", "OPENAI_API_KEY=cli"]


def test_env_passthrough_includes_cli_env_explicit_values() -> None:
    with patch.dict("os.environ", {}, clear=True):
        out = env_passthrough_args({"HUGGINGFACE_TOKEN": "hf_abc"})
    assert out == ["-e", "HUGGINGFACE_TOKEN=hf_abc"]


def test_run_container_returns_subprocess_exit_code(tmp_path: Path) -> None:
    with patch("inspect_ai._cli.docker.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 42
        rc = run_container("img", ["task"], tmp_path, {})
    assert rc == 42
    cmd = mock_run.call_args.args[0]
    assert cmd[:4] == ["docker", "run", "--rm", "-i"]
    assert "img" in cmd
    assert cmd[-1] == "task"
    assert f"{tmp_path.resolve()}:/inspect-logs" in cmd
    assert "INSPECT_LOG_DIR=/inspect-logs" in cmd
    assert "/var/run/docker.sock:/var/run/docker.sock" in cmd
    assert "INSPECT_DOCKER_DISPATCH=1" in cmd


def test_run_container_raises_click_exception_when_docker_missing(
    tmp_path: Path,
) -> None:
    with patch("inspect_ai._cli.docker.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(click.ClickException, match="docker"):
            run_container("img", ["task"], tmp_path, {})


def test_run_container_returns_130_on_keyboard_interrupt(tmp_path: Path) -> None:
    with patch("inspect_ai._cli.docker.subprocess.run", side_effect=KeyboardInterrupt):
        assert run_container("img", ["task"], tmp_path, {}) == 130


def test_cli_eval_rejects_multiple_docker_refs() -> None:
    from click.testing import CliRunner

    from inspect_ai._cli.eval import eval_command

    result = CliRunner().invoke(eval_command, ["docker://a", "docker://b"])
    assert result.exit_code != 0
    assert "at most one" in result.output


def test_cli_eval_set_rejects_docker_ref() -> None:
    from click.testing import CliRunner

    from inspect_ai._cli.eval import eval_set_command

    result = CliRunner().invoke(eval_set_command, ["docker://img"])
    assert result.exit_code != 0
    assert "eval-set" in result.output


def test_cli_eval_retry_rejects_docker_ref() -> None:
    from click.testing import CliRunner

    from inspect_ai._cli.eval import eval_retry_command

    result = CliRunner().invoke(eval_retry_command, ["docker://img.eval"])
    assert result.exit_code != 0
    assert "eval-retry" in result.output
