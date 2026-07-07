"""Task executor — picks up, runs, and completes tasks from tasks.json."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from ..auth import AuthError, Authenticator, TokenCache
from ..client import ApiClient, AuthExpiredError
from ..config import Settings, settings
from .definitions import get_handler_info
from .manifest import DEFAULT_MANIFEST_PATH, Task, TaskManifest, TaskStatus

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


class TaskRunner:
    """Executes tasks from the manifest on behalf of a named agent."""

    def __init__(
        self,
        agent_id: str,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        cfg: Optional[Settings] = None,
    ) -> None:
        self.agent_id = agent_id
        self.manifest_path = Path(manifest_path)
        self._cfg = cfg or settings
        self._auth = Authenticator(cfg)
        self._token: Optional[TokenCache] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_one(self) -> bool:
        """
        Claim and execute a single pending task.

        Returns True if a task was executed, False if no pending tasks remain.
        """
        return asyncio.run(self._run_one_async())

    def run_all(self) -> int:
        """
        Loop run_one() until no pending tasks remain.

        Returns the total number of tasks executed.
        """
        executed = 0
        while self.run_one():
            executed += 1
        logger.info("Agent %s finished — %d tasks executed", self.agent_id, executed)
        return executed

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _run_one_async(self) -> bool:
        manifest = TaskManifest.load(self.manifest_path)
        task = manifest.claim_next(self.agent_id)
        if task is None:
            logger.info("Agent %s: no pending tasks", self.agent_id)
            return False

        token = self._get_token()
        async with ApiClient(token, self._cfg) as client:
            try:
                result = await self._execute(client, task)
                result_file = self._write_result(task.id, result)
                manifest.complete(task.id, str(result_file))
                logger.info("Task %s completed → %s", task.id, result_file)

            except AuthExpiredError:
                logger.warning("Auth expired — invalidating and retrying task %s", task.id)
                self._auth.invalidate()
                self._token = None
                token = self._get_token()
                async with ApiClient(token, self._cfg) as retry_client:
                    try:
                        result = await self._execute(retry_client, task)
                        result_file = self._write_result(task.id, result)
                        manifest.complete(task.id, str(result_file))
                    except Exception as exc:
                        manifest.fail(task.id, str(exc))
                        logger.error("Task %s failed after retry: %s", task.id, exc)

            except Exception as exc:
                manifest.fail(task.id, str(exc))
                logger.error("Task %s failed: %s", task.id, exc)

        return True

    async def _execute(self, client: ApiClient, task: Task) -> Any:
        """Dispatch a task to the appropriate client method and return the result."""
        method_name, required_params = get_handler_info(task.type)

        for param in required_params:
            if param not in task.params:
                raise ValueError(
                    f"Task {task.id} ({task.type}) missing required param '{param}'"
                )

        if task.type.endswith("_all"):
            base_type = task.type[: -len("_all")]
            base_method = getattr(client, base_type, None)
            if base_method is None:
                raise ValueError(f"No client method for base type '{base_type}'")
            result = await client.get_all_pages(base_method, **task.params)
        else:
            method = getattr(client, method_name)
            result = await method(**task.params)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_token(self) -> TokenCache:
        if self._token is None or not self._token.is_valid():
            self._token = self._auth.get_token()
        return self._token

    @staticmethod
    def _write_result(task_id: str, result: Any) -> Path:
        RESULTS_DIR.mkdir(exist_ok=True)
        result_file = RESULTS_DIR / f"{task_id}.json"
        if hasattr(result, "model_dump"):
            payload = result.model_dump()
        elif isinstance(result, list):
            payload = [
                r.model_dump() if hasattr(r, "model_dump") else r for r in result
            ]
        else:
            payload = result
        result_file.write_text(json.dumps(payload, indent=2, default=str))
        return result_file
