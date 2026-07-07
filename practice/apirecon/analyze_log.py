#!/usr/bin/env python3
"""
analyze_log.py — Inspect and filter the captured API log (api_log.jsonl).

Usage:
    python3 analyze_log.py                     # summary of all requests
    python3 analyze_log.py --host api.example  # filter by host substring
    python3 analyze_log.py --method POST       # filter by HTTP method
    python3 analyze_log.py --status 200        # filter by status code
    python3 analyze_log.py --search "video"    # search URLs and bodies
    python3 analyze_log.py --dump              # print full JSON for each entry
    python3 analyze_log.py --endpoints         # deduplicated endpoint list
"""

import argparse
import json
import os
import sys
from collections import Counter


DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "api_log.jsonl")


def load_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"Log file not found: {path}\nRun run_capture.sh first.")
        sys.exit(1)
    entries = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed line {i}: {e}", file=sys.stderr)
    return entries


def filter_entries(entries, args) -> list[dict]:
    out = []
    for e in entries:
        req = e.get("request", {})
        resp = e.get("response", {})
        url = req.get("url", "")
        host = req.get("host", "")
        method = req.get("method", "")
        status = resp.get("status_code")

        if args.host and args.host.lower() not in host.lower():
            continue
        if args.method and args.method.upper() != method.upper():
            continue
        if args.status and str(status) != str(args.status):
            continue
        if args.search:
            needle = args.search.lower()
            body_str = json.dumps(req.get("body", "")) + json.dumps(resp.get("body", ""))
            if needle not in url.lower() and needle not in body_str.lower():
                continue
        out.append(e)
    return out


def print_summary(entries):
    host_counts: Counter = Counter()
    method_counts: Counter = Counter()
    status_counts: Counter = Counter()
    slowest = []

    for e in entries:
        req = e.get("request", {})
        resp = e.get("response", {})
        host_counts[req.get("host", "?")] += 1
        method_counts[req.get("method", "?")] += 1
        status_counts[str(resp.get("status_code", "?"))] += 1
        ms = e.get("elapsed_ms")
        if ms is not None:
            slowest.append((ms, req.get("url", "")))

    print(f"\n{'━'*64}")
    print(f"  Total requests captured: {len(entries)}")
    print(f"{'━'*64}")

    print("\n  By host:")
    for host, count in host_counts.most_common(20):
        print(f"    {count:5d}  {host}")

    print("\n  By method:")
    for m, count in method_counts.most_common():
        print(f"    {count:5d}  {m}")

    print("\n  By status code:")
    for s, count in sorted(status_counts.items()):
        print(f"    {count:5d}  {s}")

    if slowest:
        print("\n  Slowest 5 requests:")
        for ms, url in sorted(slowest, reverse=True)[:5]:
            print(f"    {ms:>8.1f}ms  {url}")
    print()


def print_entries(entries, dump=False):
    for e in entries:
        req = e.get("request", {})
        resp = e.get("response", {})
        ms = e.get("elapsed_ms", "?")
        status = resp.get("status_code", "???")
        method = req.get("method", "?")
        url = req.get("url", "?")

        color = "\033[92m" if 200 <= (status or 0) < 300 else "\033[91m"
        reset = "\033[0m"

        print(f"{color}{method:6}{reset} {status}  {ms:>7}ms  {url}")
        if dump:
            print(json.dumps(e, indent=2, ensure_ascii=False))
            print()


def print_endpoints(entries):
    """Deduplicated list of (method, path-without-query) pairs per host."""
    seen = set()
    for e in entries:
        req = e.get("request", {})
        method = req.get("method", "?")
        host = req.get("host", "?")
        path = req.get("path", "?").split("?")[0]
        key = (method, host, path)
        if key not in seen:
            seen.add(key)
            print(f"  {method:6}  {host}{path}")
    print(f"\n  {len(seen)} unique endpoints.\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze captured API traffic log.")
    parser.add_argument("--log",      default=DEFAULT_LOG, help="Path to api_log.jsonl")
    parser.add_argument("--host",     default="",          help="Filter by host substring")
    parser.add_argument("--method",   default="",          help="Filter by HTTP method")
    parser.add_argument("--status",   default="",          help="Filter by status code")
    parser.add_argument("--search",   default="",          help="Search URLs and bodies")
    parser.add_argument("--dump",     action="store_true", help="Print full JSON for each entry")
    parser.add_argument("--endpoints",action="store_true", help="Deduplicated endpoint list")
    args = parser.parse_args()

    entries = load_log(args.log)
    filtered = filter_entries(entries, args)

    if args.endpoints:
        print(f"\n  Unique endpoints ({len(filtered)} matching requests):\n")
        print_endpoints(filtered)
    elif args.dump or any([args.host, args.method, args.status, args.search]):
        print_entries(filtered, dump=args.dump)
        print(f"\n  {len(filtered)} of {len(entries)} entries shown.\n")
    else:
        print_summary(filtered)


if __name__ == "__main__":
    main()
