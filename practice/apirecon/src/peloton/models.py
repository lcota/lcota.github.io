"""Pydantic response models for the Peloton API."""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class _BaseModel(BaseModel):
    """Base model that tolerates unknown fields."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class PaginatedMeta(_BaseModel):
    current_page: int = 0
    total_pages: int = 1
    total_items: Optional[int] = None


class PaginatedResponse(_BaseModel, Generic[T]):
    data: list[Any] = []
    meta: Optional[PaginatedMeta] = None


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------


class ApiEnvelope(_BaseModel):
    """Generic API response envelope."""

    data: Any = None
    meta: Optional[Any] = None


# ---------------------------------------------------------------------------
# App-specific models
# ---------------------------------------------------------------------------

# TODO: Define models here as you discover response shapes from captured traffic.
#       Pattern to follow:
#
#   class MyResource(_BaseModel):
#       """Returned by GET /resources/{id}"""
#       id: Optional[int] = None
#       name: Optional[str] = None
#       created_at: Optional[str] = None
#       # Add fields as observed — _BaseModel tolerates unknown fields,
#       # so you don't need to be exhaustive up front.
