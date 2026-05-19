"""Tests for `reject_bind_mounts_under_dispatch`.

Host-path bind-mount rejection when running under `inspect eval docker://`
dispatch.
"""

from __future__ import annotations

import pytest

from inspect_ai._util.error import PrerequisiteError
from inspect_ai.util._sandbox.docker.compose import reject_bind_mounts_under_dispatch


def _service(volumes: list) -> dict:
    return {"image": "img", "volumes": volumes}


def test_long_form_bind_is_rejected() -> None:
    services = {
        "default": _service(
            [{"type": "bind", "source": "/host/path", "target": "/in/container"}]
        )
    }
    with pytest.raises(PrerequisiteError, match=r"/host/path"):
        reject_bind_mounts_under_dispatch(services)


def test_long_form_volume_is_allowed() -> None:
    services = {
        "default": _service(
            [{"type": "volume", "source": "named_vol", "target": "/data"}]
        )
    }
    reject_bind_mounts_under_dispatch(services)


def test_short_form_absolute_path_is_rejected() -> None:
    services = {"default": _service(["/host/path:/in/container:ro"])}
    with pytest.raises(PrerequisiteError, match=r"/host/path"):
        reject_bind_mounts_under_dispatch(services)


def test_short_form_relative_path_is_rejected() -> None:
    services = {"default": _service(["./local-file:/in/container"])}
    with pytest.raises(PrerequisiteError, match=r"./local-file"):
        reject_bind_mounts_under_dispatch(services)


def test_short_form_named_volume_is_allowed() -> None:
    services = {"default": _service(["named_vol:/data"])}
    reject_bind_mounts_under_dispatch(services)


def test_no_volumes_field_is_allowed() -> None:
    reject_bind_mounts_under_dispatch({"default": {"image": "img"}})


def test_empty_volumes_list_is_allowed() -> None:
    reject_bind_mounts_under_dispatch({"default": _service([])})


def test_multiple_services_each_checked() -> None:
    services = {
        "default": _service([{"type": "volume", "source": "v", "target": "/d"}]),
        "sidecar": _service(["/host:/c"]),
    }
    with pytest.raises(PrerequisiteError, match=r"sidecar"):
        reject_bind_mounts_under_dispatch(services)
