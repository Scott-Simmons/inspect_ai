from inspect_ai import Task, eval, task
from inspect_ai._util.constants import PKG_NAME
from inspect_ai._util.registry import (
    LazyRegistryObject,
    RegistryInfo,
    is_lazy_registry_object,
    registry_add_lazy,
    registry_create_from_dict,
    registry_info,
    registry_lookup,
    registry_value,
)
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Metric, metric
from inspect_ai.scorer._metric import SampleScore
from inspect_ai.solver import Solver, solver, use_tools
from inspect_ai.tool import Tool, bash


def test_registry_namespaces() -> None:
    # define a local metric which we can lookup by simple name
    @metric(name="local_accuracy")
    def accuracy1(correct: str = "C") -> Metric:
        def metric(scores: list[SampleScore]) -> int | float:
            return 1

        return metric

    assert registry_lookup("metric", "local_accuracy")

    # confirm that inspect_ai builtins have their namespace auto-appended
    info = registry_info(registry_lookup("metric", f"{PKG_NAME}/accuracy"))
    assert info
    assert info.name == f"{PKG_NAME}/accuracy"


def test_registry_dict() -> None:
    @solver
    def create_solver(tool: Tool) -> Solver:
        return use_tools(tool)

    mysolver = create_solver(bash(timeout=10))
    solver_dict = registry_value(mysolver)
    assert solver_dict["type"] == "solver"
    assert solver_dict["params"]["tool"]["type"] == "tool"

    mysolver2 = registry_create_from_dict(solver_dict)
    assert isinstance(mysolver2, Solver)


@task
def task_with_default(variant: str = "default") -> Task:
    return Task(dataset=[Sample(input="")], plan=[])


def test_registry_tag_default_argument() -> None:
    task_instance = task_with_default()
    log = eval(task_instance)[0]
    assert log.eval.task_args == {"variant": "default"}


def test_registry_tag_overridden_default() -> None:
    task_instance = task_with_default(variant="override")
    log = eval(task_instance)[0]
    assert log.eval.task_args == {"variant": "override"}


@task
def task_with_default_and_required(required: str, variant: str = "default") -> Task:
    return Task(dataset=[Sample(input="")], plan=[])


def test_registry_tag_default_with_required() -> None:
    task_instance = task_with_default_and_required("required_value")
    log = eval(task_instance)[0]
    assert log.eval.task_args == {"required": "required_value", "variant": "default"}


def test_registry_tag_overridden_default_with_required() -> None:
    task_instance = task_with_default_and_required("required_value", variant="override")
    log = eval(task_instance)[0]
    assert log.eval.task_args == {"required": "required_value", "variant": "override"}


def test_lazy_registry_object() -> None:
    """Test that LazyRegistryObject loads on first access."""
    load_count = 0

    def loader() -> Metric:
        nonlocal load_count
        load_count += 1

        @metric(name="_lazy_test_metric_inner")
        def lazy_metric() -> Metric:
            def m(scores: list[SampleScore]) -> int | float:
                return 1

            return m

        return lazy_metric()

    info = RegistryInfo(type="metric", name="_lazy_test_metric", metadata={})
    lazy = LazyRegistryObject(info=info, loader=loader)

    # Not loaded yet
    assert not lazy.loaded
    assert load_count == 0

    # Load triggers the loader
    result = lazy.load()
    assert lazy.loaded
    assert load_count == 1
    assert result is not None

    # Second load returns cached value
    result2 = lazy.load()
    assert load_count == 1  # Still 1, not 2
    assert result2 is result


def test_registry_add_lazy() -> None:
    """Test that registry_add_lazy registers a lazy object that loads on lookup."""
    load_count = 0

    def loader() -> Metric:
        nonlocal load_count
        load_count += 1

        @metric(name="_lazy_registered_metric_inner")
        def lazy_metric() -> Metric:
            def m(scores: list[SampleScore]) -> int | float:
                return 42

            return m

        return lazy_metric()

    info = RegistryInfo(
        type="metric", name="_lazy_registered_metric", metadata={"version": 1}
    )
    registry_add_lazy(info=info, loader=loader)

    # Not loaded yet
    assert load_count == 0

    # Lookup triggers load
    result = registry_lookup("metric", "_lazy_registered_metric")
    assert load_count == 1
    assert result is not None

    # Verify registry info was transferred
    result_info = registry_info(result)
    assert result_info.name == "_lazy_registered_metric"
    assert result_info.metadata.get("version") == 1


def test_is_lazy_registry_object() -> None:
    """Test the is_lazy_registry_object type guard."""
    info = RegistryInfo(type="metric", name="_test", metadata={})
    lazy = LazyRegistryObject(info=info, loader=lambda: None)

    assert is_lazy_registry_object(lazy)
    assert not is_lazy_registry_object("not lazy")
    assert not is_lazy_registry_object(None)
    assert not is_lazy_registry_object(42)
