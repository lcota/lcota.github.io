"""Async HTTP API client for Peloton.

Completely decoupled from auth concerns — receives a TokenCache
and issues typed API requests using httpx.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

import httpx

from .auth import TokenCache
from .config import Settings, settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# TODO: Replace with the User-Agent string seen in captured traffic headers.
USER_AGENT = "Peloton/3.0 CFNetwork/1"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """Generic API error with status code and response body."""

    def __init__(self, status_code: int, body: Any, message: str = "") -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(message or f"API error {status_code}: {body}")


class AuthExpiredError(ApiError):
    """Raised when the server returns 401, signalling token expiry."""


class RateLimitError(ApiError):
    """Raised when the server returns 429."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ApiClient:
    """Async HTTP client for Peloton API endpoints."""

    def __init__(
        self,
        token: TokenCache,
        cfg: Optional[Settings] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._token = token
        self._cfg = cfg or settings
        self._http = http_client or httpx.AsyncClient(
            base_url=self._cfg.base_url,
            headers={
                # TODO: Inspect captured traffic to find the exact Authorization
                #       header format. Common patterns:
                #         "Bearer {token}"
                #         "{user_id}:{token}"
                #         Just the token string
                "Authorization": f"{token.user_id}:{token.token}",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            timeout=httpx.Timeout(self._cfg.request_timeout),
        )

    async def __aenter__(self) -> "ApiClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internal request helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, **params: Any) -> Any:
        resp = await self._http.get(
            path, params={k: v for k, v in params.items() if v is not None}
        )
        self._raise_for_status(resp)
        return resp.json()

    async def _post(self, path: str, body: dict | None = None, **params: Any) -> Any:
        resp = await self._http.post(
            path,
            json=body,
            params={k: v for k, v in params.items() if v is not None},
        )
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code == 401:
            raise AuthExpiredError(resp.status_code, resp.text)
        if resp.status_code == 429:
            raise RateLimitError(resp.status_code, resp.text)
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, resp.text)

    # ------------------------------------------------------------------
    # Endpoint methods
    # ------------------------------------------------------------------

    async def ping(self) -> dict:
        """POST /ping — used to validate token and refresh expiry.

        TODO: Update the path and body to match the app's actual keepalive
              endpoint (visible in captured traffic).
        """
        resp = await self._http.post(
            "/ping",
            json={},  # TODO: add required body fields observed in traffic
        )
        self._raise_for_status(resp)
        return resp.json()

    # TODO: Add typed endpoint methods as you discover them from captured traffic.
    #       Pattern to follow:
    #
    #   async def get_resource(self, resource_id: int) -> MyModel:
    #       """GET /resources/{resource_id}"""
    #       raw = await self._get(f"/resources/{resource_id}")
    #       return MyModel.model_validate(raw.get("data", raw))
    #
    #   async def list_resources(self, page: int = 0) -> PaginatedResponse:
    #       """GET /resources?page={page}"""
    #       from .models import PaginatedResponse
    #       raw = await self._get("/resources", page=page)
    #       return self._parse_paginated(raw)

    # ------------------------------------------------------------------
    # Pagination helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_paginated(raw: dict) -> "PaginatedResponse":
        from .models import PaginatedMeta, PaginatedResponse
        meta_raw = raw.get("meta", {}) or {}
        meta = PaginatedMeta(
            current_page=meta_raw.get("current_page", 0),
            total_pages=meta_raw.get("total_pages", 1),
            total_items=meta_raw.get("total_items"),
        )
        data = raw.get("data", [])
        if not isinstance(data, list):
            data = [data]
        return PaginatedResponse(data=data, meta=meta)

    async def get_all_pages(
        self,
        method: Callable[..., Any],
        **kwargs: Any,
    ) -> list[Any]:
        """
        Call a paginated method repeatedly until all pages are fetched.

        Example:
            all_items = await client.get_all_pages(client.list_resources)
        """
        all_items: list[Any] = []
        page = 0
        while True:
            result = await method(page=page, **kwargs)
            all_items.extend(result.data)
            meta = result.meta
            if meta is None or page >= (meta.total_pages - 1):
                break
            page += 1
        return all_items
