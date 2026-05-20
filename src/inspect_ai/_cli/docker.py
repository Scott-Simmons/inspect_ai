"""Dispatch `inspect eval docker://<image>` to a container.

The host mounts the log dir and the docker socket (so sandbox-using evals
spawn sibling containers on the host daemon) and sets
`INSPECT_DOCKER_DISPATCH=1`, which arms `reject_bind_mounts_under_dispatch`
in the sandbox layer — bind mounts would resolve against the host fs, not
the eval container, and silently corrupt the sandbox.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import click

DOCKER_REF_PREFIX = "docker://"
CONTAINER_LOG_DIR = "/inspect-logs"
INSPECT_LOG_DIR_ENV = "INSPECT_LOG_DIR"
DOCKER_DISPATCH_ENV = "INSPECT_DOCKER_DISPATCH"
DOCKER_SOCKET = "/var/run/docker.sock"

# Options the host owns: the host computes the value and rewrites it for the
# container (so the container must not re-receive the original). Currently
# just --log-dir, which the host mounts at CONTAINER_LOG_DIR and re-exposes
# via INSPECT_LOG_DIR.
HOST_OWNED_OPTIONS: frozenset[str] = frozenset({"--log-dir"})

FORWARDED_ENV: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZUREAI_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_ANTHROPIC_API_KEY",
    "AZUREAI_ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "VERTEX_API_KEY",
    "TOGETHER_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "AZURE_MISTRAL_API_KEY",
    "AZUREAI_MISTRAL_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "AZURE_API_KEY",
    "AZUREAI_ENDPOINT_KEY",
    "HF_TOKEN",
)


def has_docker_ref(tasks: tuple[str, ...] | None) -> bool:
    return bool(tasks) and any(t.startswith(DOCKER_REF_PREFIX) for t in tasks)


def dispatch_to_docker(
    ctx: click.Context,
    tasks: tuple[str, ...],
    log_dir: str,
    env: tuple[str, ...] | None,
) -> int:
    refs = [t for t in tasks if t.startswith(DOCKER_REF_PREFIX)]
    if len(refs) > 1:
        raise click.UsageError(
            f"`inspect eval` accepts at most one {DOCKER_REF_PREFIX}<image> ref"
        )
    image_ref = refs[0]
    image = image_ref.removeprefix(DOCKER_REF_PREFIX)
    if not image:
        raise click.UsageError(f"Empty image reference: `{image_ref}`")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    return run_container(
        image,
        build_passthrough(ctx, image_ref),
        log_path,
        parse_cli_env(env),
    )


def reject_docker_refs(tasks: tuple[str, ...] | None, subcommand: str) -> None:
    if has_docker_ref(tasks):
        raise click.UsageError(
            f"{DOCKER_REF_PREFIX} task refs are not yet supported in "
            f"`inspect {subcommand}` (only `inspect eval`)"
        )


def build_passthrough(ctx: click.Context, image_ref: str) -> list[str]:
    assert ctx.info_name == "eval"
    try:
        eval_args = sys.argv[sys.argv.index("eval") + 1 :]
    except ValueError:
        return []
    return [a for a in drop_host_owned_options(eval_args) if a != image_ref]


def drop_host_owned_options(args: Iterable[str]) -> Iterator[str]:
    it = iter(args)
    for arg in it:
        if arg in HOST_OWNED_OPTIONS:
            next(it, None)
            continue
        if any(arg.startswith(f"{opt}=") for opt in HOST_OWNED_OPTIONS):
            continue
        yield arg


def parse_cli_env(env: tuple[str, ...] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in env or ():
        if "=" not in entry:
            raise click.UsageError(f"`--env {entry}` must be of the form NAME=value")
        name, value = entry.split("=", 1)
        out[name] = value
    return out


def run_container(
    image: str, args: list[str], log_dir: Path, cli_env: dict[str, str]
) -> int:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "-v",
        f"{log_dir.resolve()}:{CONTAINER_LOG_DIR}",
        "-v",
        f"{DOCKER_SOCKET}:{DOCKER_SOCKET}",
        "-e",
        f"{INSPECT_LOG_DIR_ENV}={CONTAINER_LOG_DIR}",
        "-e",
        f"{DOCKER_DISPATCH_ENV}=1",
        *env_passthrough_args(cli_env),
        image,
        *args,
    ]
    try:
        return subprocess.run(cmd, check=False).returncode
    except FileNotFoundError:
        raise click.ClickException(
            "`docker` was not found on PATH. `inspect eval docker://...` requires Docker."
        )
    except KeyboardInterrupt:
        return 130


def env_passthrough_args(cli_env: dict[str, str]) -> list[str]:
    out: list[str] = []
    for name in FORWARDED_ENV:
        if name in os.environ and name not in cli_env:
            out += ["-e", name]
    for name, value in cli_env.items():
        out += ["-e", f"{name}={value}"]
    return out
