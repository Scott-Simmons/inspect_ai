"""Git-based task version discovery."""

import ast
import subprocess
from pathlib import Path
from types import ModuleType

# This file is a smell, but all good for an experiment


def get_file_at_commit(
    repo_path: str | Path, commit: str, file_path: str
) -> str | None:
    """Get file contents at a specific commit."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{file_path}"],
        cwd=Path(repo_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def extract_task_versions(source: str) -> dict[str, int | str]:
    """Extract task names and versions

    Returns:
        Dict mapping task name to version.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    versions: dict[str, int | str] = {}
    current_task_name: str | None = None

    for node in ast.walk(tree):
        # Look for @task decorator to get task name
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "task":
                    current_task_name = node.name
                elif isinstance(decorator, ast.Call):
                    if (
                        isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "task"
                    ):
                        # Check for name= argument in @task(name="...")
                        for keyword in decorator.keywords:
                            if keyword.arg == "name" and isinstance(
                                keyword.value, ast.Constant
                            ):
                                name_val = keyword.value.value
                                if isinstance(name_val, str):
                                    current_task_name = name_val
                                break
                        else:
                            current_task_name = node.name

        # Look for Task() constructor calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "Task":
                for keyword in node.keywords:
                    if keyword.arg == "version" and isinstance(
                        keyword.value, ast.Constant
                    ):
                        version = keyword.value.value
                        if current_task_name and isinstance(version, (int, str)):
                            versions[current_task_name] = version

    return versions


def find_commit_for_version(
    repo_path: str | Path,
    task_file: str,
    task_name: str,
    version: int | str,
) -> str | None:
    """Find the commit where a task had a specific version.

    Args:
        repo_path: Path to the git repository.
        task_file: Relative path to the task file within the repo.
        task_name: Name of the task.
        version: The version to find.

    Returns:
        Commit SHA where this task had this version, or None if not found.
    """
    # Get all commits that touched this file
    result = subprocess.run(
        ["git", "log", "--format=%H", "--", task_file],
        cwd=Path(repo_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None

    commits = result.stdout.strip().split("\n")

    for commit in commits:
        source = get_file_at_commit(repo_path, commit, task_file)
        if source:
            versions = extract_task_versions(source)
            if versions.get(task_name) == version:
                return commit

    return None


def load_module_from_source(source: str, module_name: str) -> ModuleType:
    """Load a Python module from source code string."""
    module = ModuleType(module_name)
    exec(source, module.__dict__)
    return module


def load_task_at_version(
    repo_path: str | Path,
    task_file: str,
    task_name: str,
    version: int | str,
) -> ModuleType | None:
    """Load a task module at a specific version.

    Args:
        repo_path: Path to the git repository.
        task_file: Relative path to the task file within the repo.
        task_name: Name of the task.
        version: The version to load.

    Returns:
        The loaded module, or None if version not found.
    """
    commit = find_commit_for_version(repo_path, task_file, task_name, version)
    if not commit:
        return None

    source = get_file_at_commit(repo_path, commit, task_file)
    if not source:
        return None

    module_name = f"{task_name}_v{version}"
    return load_module_from_source(source, module_name)
