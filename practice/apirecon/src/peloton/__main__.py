"""CLI entry point for the Peloton API client.

Usage:
    peloton auth login
    peloton auth status
    peloton auth invalidate
    peloton auth set <token>
    peloton tasks generate
    peloton tasks status
    peloton tasks run --agent-id agent-1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def cmd_auth_login(args: argparse.Namespace) -> None:
    from .auth import Authenticator
    from .config import settings

    auth = Authenticator()
    print("Logging in via API…")
    token = auth.get_token()
    print(f"Token cached for user {token.user_id}")
    print(f"  Expires at : {token.expires_at.isoformat()}")
    print(f"  Cache file : {settings.token_cache}")


def cmd_auth_status(args: argparse.Namespace) -> None:
    from .auth import Authenticator
    from .config import settings

    auth = Authenticator()
    cached = auth._load_cache()
    if cached is None:
        print("No cached token found.")
        sys.exit(1)
    print(f"User ID    : {cached.user_id}")
    print(f"Obtained at: {cached.obtained_at.isoformat()}")
    print(f"Expires at : {cached.expires_at.isoformat()}")
    print(f"Valid      : {cached.is_valid()}")
    print(f"Cache file : {settings.token_cache}")


def cmd_auth_invalidate(args: argparse.Namespace) -> None:
    from .auth import Authenticator

    auth = Authenticator()
    auth.invalidate()
    print("Token cache cleared.")


def cmd_auth_set(args: argparse.Namespace) -> None:
    from .auth import Authenticator, AuthError

    auth = Authenticator()
    try:
        token = auth.save_manual_token(args.token)
        print(f"Token cached for user {token.user_id}")
        print(f"  Expires at : {token.expires_at.isoformat()}")
    except AuthError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_tasks_generate(args: argparse.Namespace) -> None:
    from .tasks.manifest import TaskManifest

    path = Path(args.output)
    manifest = TaskManifest.generate_initial_tasks(path=path)
    manifest.save()
    print(f"Task manifest generated: {path}")
    status = manifest.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")


def cmd_tasks_status(args: argparse.Namespace) -> None:
    from .tasks.manifest import TaskManifest

    manifest = TaskManifest.load(Path(args.manifest))
    status = manifest.get_status()
    total = sum(status.values())
    print(f"Task manifest: {args.manifest}  (total: {total})")
    for k, v in status.items():
        bar = "#" * v
        print(f"  {k:12s}: {v:4d}  {bar}")


def cmd_tasks_run(args: argparse.Namespace) -> None:
    from .tasks.runner import TaskRunner

    runner = TaskRunner(
        agent_id=args.agent_id,
        manifest_path=Path(args.manifest),
    )
    if args.one:
        ran = runner.run_one()
        if not ran:
            print("No pending tasks.")
    else:
        n = runner.run_all()
        print(f"Completed {n} task(s).")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peloton",
        description="Peloton API client CLI",
    )
    parser.add_argument(
        "--manifest",
        default="tasks.json",
        help="Path to tasks.json (default: tasks.json)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- auth ---
    auth_p = sub.add_parser("auth", help="Authentication commands")
    auth_sub = auth_p.add_subparsers(dest="auth_command", required=True)

    auth_sub.add_parser("login", help="Log in via API and cache token")
    auth_sub.add_parser("status", help="Show cached token status")
    auth_sub.add_parser("invalidate", help="Clear cached token")

    set_p = auth_sub.add_parser(
        "set", help="Set token manually from intercepted auth header"
    )
    set_p.add_argument("token", help="Token in 'user_id:token_secret' format")

    # --- tasks ---
    tasks_p = sub.add_parser("tasks", help="Task manifest commands")
    tasks_sub = tasks_p.add_subparsers(dest="tasks_command", required=True)

    gen_p = tasks_sub.add_parser("generate", help="Generate initial task manifest")
    gen_p.add_argument(
        "--output", default="tasks.json", help="Output path for tasks.json"
    )

    tasks_sub.add_parser("status", help="Show task manifest status")

    run_p = tasks_sub.add_parser("run", help="Run tasks as an agent")
    run_p.add_argument("--agent-id", required=True, help="Unique agent identifier")
    run_p.add_argument(
        "--one", action="store_true", help="Execute only one task, then exit"
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "auth":
        dispatch = {
            "login": cmd_auth_login,
            "status": cmd_auth_status,
            "invalidate": cmd_auth_invalidate,
            "set": cmd_auth_set,
        }
        dispatch[args.auth_command](args)

    elif args.command == "tasks":
        dispatch = {
            "generate": cmd_tasks_generate,
            "status": cmd_tasks_status,
            "run": cmd_tasks_run,
        }
        dispatch[args.tasks_command](args)


if __name__ == "__main__":
    main()
