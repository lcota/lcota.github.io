"""Authentication and token management for Peloton."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from .config import Settings, settings

logger = logging.getLogger(__name__)

# TODO: Set these after inspecting captured traffic.
#       LOGIN_URL is the endpoint the app POSTs credentials to.
#       API_HOST is the host you'll filter on in run_capture.sh.
#       USER_AGENT should match the value seen in intercepted request headers.
LOGIN_URL = "https://api.onepeloton.com/login"
API_HOST = "api.onepeloton.com"
USER_AGENT = "Peloton/3.0 CFNetwork/1"

TOKEN_EXPIRY_DAYS = 56
TOKEN_SAFETY_MARGIN_HOURS = 24


@dataclass
class TokenCache:
    """Cached authentication credentials."""

    user_id: int
    token: str
    expires_at: datetime
    obtained_at: datetime

    def is_valid(self) -> bool:
        """Return True if the token is not yet expired (with safety margin)."""
        cutoff = datetime.now(tz=timezone.utc) + timedelta(
            hours=TOKEN_SAFETY_MARGIN_HOURS
        )
        return self.expires_at > cutoff

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "token": self.token,
            "expires_at": self.expires_at.isoformat(),
            "obtained_at": self.obtained_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenCache":
        return cls(
            user_id=data["user_id"],
            token=data["token"],
            expires_at=datetime.fromisoformat(data["expires_at"]),
            obtained_at=datetime.fromisoformat(data["obtained_at"]),
        )


class AuthError(Exception):
    """Raised when authentication fails."""


class Authenticator:
    """Handles Peloton login and token caching."""

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self._cfg = cfg or settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_token(self) -> TokenCache:
        """
        Return a valid TokenCache.

        Checks the local cache first; if missing or expired, performs a
        fresh login to obtain a new token.
        """
        cached = self._load_cache()
        if cached is not None and cached.is_valid():
            logger.debug("Using cached token for user %s", cached.user_id)
            return cached

        logger.info("Cached token missing or expired — performing fresh login")
        return self._login_and_capture()

    def invalidate(self) -> None:
        """Delete the cached token file to force re-login."""
        cache_path = Path(self._cfg.token_cache)
        if cache_path.exists():
            cache_path.unlink()
            logger.info("Token cache invalidated: %s", cache_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _login_and_capture(self) -> TokenCache:
        """
        Log in via HTTP API and capture credentials.

        TODO: Fill this in after inspecting captured login traffic.
              Steps to complete:
              1. Find the POST endpoint the app uses to log in (e.g. /v1/login,
                 /auth/token, /api/auth).
              2. Note the request body shape (username/password field names).
              3. Note the response shape — which field is the token?
                 Which field is the user identifier? Is there an expiry field?
              4. Replace the stub below with real field names.
        """
        if not self._cfg.email or not self._cfg.password:
            raise AuthError(
                "PELOTON_EMAIL and PELOTON_PASSWORD must be set in environment or .env"
            )

        logger.info("Logging in via direct API call")

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # TODO: Replace field names with what the app actually sends
        payload = {"username": self._cfg.email, "password": self._cfg.password}

        try:
            resp = httpx.post(
                f"{self._cfg.base_url}/login",  # TODO: adjust path if needed
                json=payload,
                headers=headers,
                timeout=self._cfg.login_timeout,
            )
            resp.raise_for_status()
            data = resp.json().get("data", resp.json())  # TODO: unwrap envelope if needed
        except httpx.HTTPError as exc:
            logger.error("Login failed: %s", exc)
            if hasattr(exc, "response") and exc.response is not None:
                logger.error("Response body: %s", exc.response.text)
            raise AuthError("Failed to authenticate via API.") from exc

        # TODO: Replace these field names with the real ones from the response
        user_id = data.get("user_id")          # e.g. "id", "userId", "sub"
        token_str = data.get("access_token")   # e.g. "token", "access_token_secret"
        expires_raw = data.get("expires_at")   # e.g. "token_expires_at", "exp"

        if not user_id or not token_str:
            raise AuthError("Login response missing user_id or token")

        logger.debug("Successfully acquired credentials for user %s", user_id)

        if expires_raw:
            expires_at = datetime.fromisoformat(expires_raw)
        else:
            expires_at = datetime.now(tz=timezone.utc) + timedelta(
                days=TOKEN_EXPIRY_DAYS
            )

        token = TokenCache(
            user_id=int(user_id),
            token=token_str,
            expires_at=expires_at,
            obtained_at=datetime.now(tz=timezone.utc),
        )
        self._save_cache(token)
        return token

    def save_manual_token(self, auth_header: str) -> TokenCache:
        """Manually ingest a {user_id}:{token} string and save it to cache."""
        try:
            user_id_str, token_str = auth_header.split(":", 1)
        except ValueError:
            raise AuthError("Token must be in the format 'user_id:token_secret'")

        token = TokenCache(
            user_id=int(user_id_str),
            token=token_str,
            expires_at=datetime.now(tz=timezone.utc) + timedelta(days=TOKEN_EXPIRY_DAYS),
            obtained_at=datetime.now(tz=timezone.utc),
        )

        if self._validate_token(token):
            self._save_cache(token)
            return token
        else:
            raise AuthError("Provided token is invalid or expired.")

    def _fetch_expiry(self, user_id: int, token: str) -> datetime:
        """
        Call the ping/keepalive endpoint to determine the token's expiry date.

        TODO: Update the URL, headers, and response field names to match the
              app's actual keepalive endpoint (observed in captured traffic).
        """
        headers = {
            "Authorization": f"{user_id}:{token}",  # TODO: check auth header format
            "Accept": "application/json",
        }
        try:
            resp = httpx.post(
                f"{self._cfg.base_url}/ping",  # TODO: adjust if app uses a different path
                json={},  # TODO: add any required body fields
                headers=headers,
                timeout=self._cfg.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            expires_raw = data.get("token_expires_at") or data.get("expires_at")
            if expires_raw:
                return datetime.fromisoformat(expires_raw)
        except Exception as exc:
            logger.warning("Could not fetch expiry from ping endpoint: %s", exc)

        return datetime.now(tz=timezone.utc) + timedelta(days=TOKEN_EXPIRY_DAYS)

    def _validate_token(self, token: TokenCache) -> bool:
        """
        Call the ping endpoint to verify the cached token is still accepted.

        TODO: Same as _fetch_expiry — update URL, headers, and body to match
              the app's actual ping/keepalive endpoint.
        """
        headers = {
            "Authorization": f"{token.user_id}:{token.token}",  # TODO: check format
            "Accept": "application/json",
        }
        try:
            resp = httpx.post(
                f"{self._cfg.base_url}/ping",  # TODO: adjust path
                json={},  # TODO: add required body fields
                headers=headers,
                timeout=self._cfg.request_timeout,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                new_expiry = data.get("token_expires_at") or data.get("expires_at")
                if new_expiry:
                    token.expires_at = datetime.fromisoformat(new_expiry)
                self._save_cache(token)
                return True
        except Exception as exc:
            logger.warning("Token validation failed: %s", exc)
        return False

    def _load_cache(self) -> Optional[TokenCache]:
        cache_path = Path(self._cfg.token_cache)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text())
            return TokenCache.from_dict(data)
        except Exception as exc:
            logger.warning("Failed to load token cache: %s", exc)
            return None

    def _save_cache(self, token: TokenCache) -> None:
        cache_path = Path(self._cfg.token_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(token.to_dict(), indent=2))
        logger.debug("Token cached at %s", cache_path)
