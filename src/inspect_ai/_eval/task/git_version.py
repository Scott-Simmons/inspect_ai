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

    # First pass: collect module-level constants that might be versions
    constants: dict[str, int | str] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(
                    node.value, ast.Constant
                ):
                    val = node.value.value
                    if isinstance(val, (int, str)):
                        constants[target.id] = val

    versions: dict[str, int | str] = {}

    # Find all functions with @task decorator
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        # Check if this function has a @task decorator
        task_name: str | None = None
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "task":
                task_name = node.name
                break
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == "task":
                    # Check for name= argument in @task(name="...")
                    for keyword in decorator.keywords:
                        if keyword.arg == "name" and isinstance(
                            keyword.value, ast.Constant
                        ):
                            name_val = keyword.value.value
                            if isinstance(name_val, str):
                                task_name = name_val
                            break
                    else:
                        task_name = node.name
                    break

        if task_name is None:
            continue

        # Look for Task() calls within this function
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == "Task":
                    for keyword in child.keywords:
                        if keyword.arg == "version":
                            # Handle literal value
                            if isinstance(keyword.value, ast.Constant):
                                version = keyword.value.value
                                if isinstance(version, (int, str)):
                                    versions[task_name] = version
                            # Handle variable reference
                            elif isinstance(keyword.value, ast.Name):
                                var_name = keyword.value.id
                                if var_name in constants:
                                    versions[task_name] = constants[var_name]
                            break

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
