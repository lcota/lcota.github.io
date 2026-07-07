# Customization Guide

These are the five steps that `init.sh` cannot automate — they require you to inspect captured traffic and make judgement calls about how the target app works.

---

## Step 1 — Capture login traffic and find the auth header format

Run a capture session and log in to the app:

```bash
bash run_capture.sh
# Log in to the app, then Ctrl-C
python3 analyze_log.py --method POST --search login
```

Look for:
- Which endpoint does the app POST to when logging in?
- What is the request body shape? (`username`/`password`, `email`/`password`, etc.)
- What does the response look like? Which field is the token? Which is the user identifier? Is there an expiry field?
- What is the `Authorization` header format on subsequent authenticated requests? (`Bearer {token}`, `{user_id}:{token}`, just the token, etc.)

---

## Step 2 — Fill in `auth.py`

Open `src/{your-slug}/auth.py` and fill in `_login_and_capture()`:

1. Set `LOGIN_URL` to the real login endpoint
2. Update the `payload` dict to match the app's field names
3. Update field extraction: `user_id = data.get("...")`, `token_str = data.get("...")`
4. Update `USER_AGENT` at the top of the file to the value seen in captured headers

Also update `_validate_token()` and `_fetch_expiry()` with the real ping endpoint path and any required body fields.

---

## Step 3 — Define models in `models.py`

For each API resource you want to work with, add a model class:

```python
class MyResource(_BaseModel):
    """Returned by GET /resources/{id}"""
    id: Optional[int] = None
    name: Optional[str] = None
    created_at: Optional[str] = None
    # _BaseModel tolerates unknown fields — add what you need
```

Use `python3 analyze_log.py --dump --search "resource_name"` to inspect exact response shapes.

---

## Step 4 — Add typed endpoint methods to `client.py`

For each endpoint discovered, add a method following the pattern in the TODO comment:

```python
async def get_resource(self, resource_id: int) -> MyResource:
    """GET /resources/{resource_id}"""
    raw = await self._get(f"/resources/{resource_id}")
    return MyResource.model_validate(raw.get("data", raw))

async def list_resources(self, page: int = 0) -> PaginatedResponse:
    """GET /resources?page={page}"""
    raw = await self._get("/resources", page=page)
    return self._parse_paginated(raw)
```

Also update the `Authorization` header format in `__init__` to match what you found in Step 1.

---

## Step 5 — Register task types in `tasks/definitions.py`

For each client method you want the task runner to be able to call, add an entry:

```python
TASK_REGISTRY: dict[str, tuple[str, list[str]]] = {
    "ping":              ("ping",          []),
    "get_resource":      ("get_resource",  ["resource_id"]),
    "list_resources":    ("list_resources", []),
    "list_resources_all": ("get_all_pages", []),
}
```

The tuple is `(client_method_name, required_param_names)`. Task types ending in `_all` automatically use `client.get_all_pages()` with the base method.

Then update `tasks/manifest.py`'s `generate_initial_tasks()` to seed the manifest with your starting tasks.
