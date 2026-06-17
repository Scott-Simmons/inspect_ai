"""Tests for git-based task version loading."""

from inspect_ai._eval.task.git_version import (
    extract_task_versions,
    load_module_from_source,
)
from inspect_ai._eval.task.util import split_spec


class TestSplitSpec:
    """Tests for the split_spec function with version support."""

    def test_path_only(self):
        path, name, version = split_spec("path/to/task")
        assert path == "path/to/task"
        assert name is None
        assert version is None

    def test_path_and_name(self):
        path, name, version = split_spec("path/to/task@task_name")
        assert path == "path/to/task"
        assert name == "task_name"
        assert version is None

    def test_path_and_version(self):
        path, name, version = split_spec("path/to/task:1.0.0")
        assert path == "path/to/task"
        assert name is None
        assert version == "1.0.0"

    def test_path_name_and_version(self):
        path, name, version = split_spec("path/to/task@task_name:1.0.0")
        assert path == "path/to/task"
        assert name == "task_name"
        assert version == "1.0.0"

    def test_simple_name_with_version(self):
        path, name, version = split_spec("my_task:2.0")
        assert path == "my_task"
        assert name is None
        assert version == "2.0"


class TestExtractTaskVersions:
    """Tests for AST-based task version extraction."""

    def test_simple_task(self):
        source = """
from inspect_ai import Task, task

@task
def my_task():
    return Task(dataset=[], version="1.0.0")
"""
        versions = extract_task_versions(source)
        assert versions == {"my_task": "1.0.0"}

    def test_task_with_named_decorator(self):
        source = """
from inspect_ai import Task, task

@task(name="custom_name")
def my_task():
    return Task(dataset=[], version="2.0")
"""
        versions = extract_task_versions(source)
        assert versions == {"custom_name": "2.0"}

    def test_multiple_tasks(self):
        source = """
from inspect_ai import Task, task

@task
def task_one():
    return Task(dataset=[], version="1.0")

@task
def task_two():
    return Task(dataset=[], version="2.0")
"""
        versions = extract_task_versions(source)
        assert versions == {"task_one": "1.0", "task_two": "2.0"}

    def test_integer_version(self):
        source = """
from inspect_ai import Task, task

@task
def my_task():
    return Task(dataset=[], version=1)
"""
        versions = extract_task_versions(source)
        assert versions == {"my_task": 1}

    def test_no_version(self):
        source = """
from inspect_ai import Task, task

@task
def my_task():
    return Task(dataset=[])
"""
        versions = extract_task_versions(source)
        assert versions == {}

    def test_syntax_error_returns_empty(self):
        source = "this is not valid python {{{{"
        versions = extract_task_versions(source)
        assert versions == {}

    def test_version_from_variable(self):
        source = """
from inspect_ai import Task, task

TASK_VERSION = "1.2.3"

@task
def my_task():
    return Task(dataset=[], version=TASK_VERSION)
"""
        versions = extract_task_versions(source)
        assert versions == {"my_task": "1.2.3"}

    def test_version_from_variable_with_prefix(self):
        source = """
from inspect_ai import Task, task

MY_TASK_VERSION = "2.0.0"

@task
def my_task():
    return Task(dataset=[], version=MY_TASK_VERSION)
"""
        versions = extract_task_versions(source)
        assert versions == {"my_task": "2.0.0"}


class TestLoadModuleFromSource:
    """Tests for loading modules from source strings."""

    def test_simple_module(self):
        source = """
x = 42
def foo():
    return "bar"
"""
        module = load_module_from_source(source, "test_module")
        assert module.x == 42
        assert module.foo() == "bar"

    def test_module_with_imports(self):
        source = """
import os
path_sep = os.sep
"""
        module = load_module_from_source(source, "test_imports")
        assert hasattr(module, "path_sep")
