"""Task type registry — maps task type strings to ApiClient methods."""

from __future__ import annotations

from typing import Any, Callable, Coroutine

# Type alias for a task handler
TaskHandler = Callable[..., Coroutine[Any, Any, Any]]

# Registry: task_type -> (client_method_name, required_param_names)
TASK_REGISTRY: dict[str, tuple[str, list[str]]] = {
    # Working example: ping requires no params
    "ping": ("ping", []),
    # TODO: Register your app's task types here as you add endpoint methods.
    #       Pattern:
    #         "get_resource":      ("get_resource",  ["resource_id"]),
    #         "list_resources":    ("list_resources", []),
    #         "list_resources_all": ("get_all_pages", []),
}


def get_handler_info(task_type: str) -> tuple[str, list[str]]:
    """Return (method_name, required_params) for the given task type."""
    if task_type not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task type '{task_type}'. Known types: {list(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task_type]
