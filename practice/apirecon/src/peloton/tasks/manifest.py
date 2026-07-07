"""File-based task manifest for agent coordination.

tasks.json lives at the project root and is the single source of truth for
task state. Multiple agent processes can claim tasks concurrently via
fcntl.flock file locking.
"""

from __future__ import annotations

import fcntl
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = Path("tasks.json")


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    type: str
    params: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    result_file: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "params": self.params,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "result_file": self.result_file,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            id=data["id"],
            type=data["type"],
            params=data.get("params", {}),
            status=TaskStatus(data.get("status", "pending")),
            assigned_to=data.get("assigned_to"),
            result_file=data.get("result_file"),
            created_at=data.get(
                "created_at", datetime.now(tz=timezone.utc).isoformat()
            ),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )


class TaskManifest:
    """Reader/writer for tasks.json with file-level locking."""

    def __init__(self, path: Path = DEFAULT_MANIFEST_PATH) -> None:
        self.path = Path(path)
        self.tasks: list[Task] = []
        self.version: str = "1.0"
        self.created_at: str = datetime.now(tz=timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Factory / loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path = DEFAULT_MANIFEST_PATH) -> "TaskManifest":
        """Read and parse tasks.json from disk."""
        manifest = cls(path)
        if not Path(path).exists():
            logger.warning("tasks.json not found at %s — starting empty", path)
            return manifest
        try:
            data = json.loads(Path(path).read_text())
            manifest.version = data.get("version", "1.0")
            manifest.created_at = data.get(
                "created_at", datetime.now(tz=timezone.utc).isoformat()
            )
            manifest.tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        except Exception as exc:
            logger.error("Failed to parse tasks.json: %s", exc)
        return manifest

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write the manifest to disk."""
        data = {
            "version": self.version,
            "created_at": self.created_at,
            "tasks": [t.to_dict() for t in self.tasks],
        }
        self.path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Task operations (all use file locking for concurrent safety)
    # ------------------------------------------------------------------

    def claim_next(self, agent_id: str) -> Optional[Task]:
        """
        Atomically claim the first pending task.

        Uses fcntl.flock to prevent race conditions between parallel agents.
        Returns the claimed Task, or None if no pending tasks remain.
        """
        with open(self.path, "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                raw = json.load(fh)
                tasks = raw.get("tasks", [])
                claimed = None
                for t in tasks:
                    if t["status"] == TaskStatus.PENDING.value:
                        t["status"] = TaskStatus.IN_PROGRESS.value
                        t["assigned_to"] = agent_id
                        t["started_at"] = datetime.now(tz=timezone.utc).isoformat()
                        claimed = Task.from_dict(t)
                        break
                if claimed is not None:
                    fh.seek(0)
                    json.dump(raw, fh, indent=2)
                    fh.truncate()
                    logger.info(
                        "Agent %s claimed task %s (%s)",
                        agent_id,
                        claimed.id,
                        claimed.type,
                    )
                return claimed
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def complete(self, task_id: str, result_file: str) -> None:
        """Mark a task as completed and record the result file path."""
        self._update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            result_file=result_file,
            completed_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    def fail(self, task_id: str, error: str) -> None:
        """Mark a task as failed with an error message."""
        self._update_task(
            task_id,
            status=TaskStatus.FAILED,
            error=error,
            completed_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    def add_task(
        self,
        task_type: str,
        params: dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Task:
        """Append a new pending task to the manifest and persist."""
        task = Task(
            id=task_id or f"task-{uuid.uuid4().hex[:8]}",
            type=task_type,
            params=params,
        )
        self._append_task_locked(task)
        return task

    def get_status(self) -> dict[str, int]:
        """Return a summary of task counts grouped by status."""
        counts: dict[str, int] = {s.value: 0 for s in TaskStatus}
        manifest = TaskManifest.load(self.path)
        for t in manifest.tasks:
            counts[t.status.value] = counts.get(t.status.value, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # High-level task generation
    # ------------------------------------------------------------------

    @classmethod
    def generate_initial_tasks(
        cls,
        path: Path = DEFAULT_MANIFEST_PATH,
    ) -> "TaskManifest":
        """
        Create a fresh manifest with an initial set of tasks.

        TODO: Replace the ping task below with tasks meaningful for your app.
              Common starting points:
                - A task that fetches the user's resource list
                - A task that fetches the user's profile/account info
              Once those complete and IDs are known, fan-out tasks can be
              appended via add_task().

        Example:
            manifest.add_task("get_resource_list", {"user_id": user_id},
                              task_id="task-get-resources")
        """
        manifest = cls(path)
        manifest.created_at = datetime.now(tz=timezone.utc).isoformat()
        manifest.tasks = []
        manifest.add_task("ping", {}, task_id="task-ping")
        logger.info("Generated initial manifest at %s", path)
        return manifest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_task(self, task_id: str, **updates: Any) -> None:
        """Atomically update fields on a task, keyed by task_id."""
        with open(self.path, "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                raw = json.load(fh)
                for t in raw.get("tasks", []):
                    if t["id"] == task_id:
                        for k, v in updates.items():
                            if isinstance(v, TaskStatus):
                                t[k] = v.value
                            else:
                                t[k] = v
                        break
                fh.seek(0)
                json.dump(raw, fh, indent=2)
                fh.truncate()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _append_task_locked(self, task: Task) -> None:
        """Append a task to tasks.json with file locking. Creates file if absent."""
        if not self.path.exists():
            self.tasks.append(task)
            self.save()
            return
        with open(self.path, "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                raw = json.load(fh)
                raw.setdefault("tasks", []).append(task.to_dict())
                fh.seek(0)
                json.dump(raw, fh, indent=2)
                fh.truncate()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        self.tasks.append(task)
