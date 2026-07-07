"""
traffic_logger.py — mitmproxy addon for capturing Peloton API traffic.

Usage:
    mitmweb  -s traffic_logger.py            # browser UI + logging
    mitmdump -s traffic_logger.py            # terminal only
    mitmdump -s traffic_logger.py --set log_file=/path/to/output.jsonl

Options (set via --set key=value on the CLI):
    log_file     Path to JSONL output file (default: api_log.jsonl in same dir)
    filter_host  Only log requests whose host contains this string
                 (default: "" = log everything)
    log_bodies   Set to "false" to skip recording request/response bodies
                 (default: true)
    max_body_kb  Truncate bodies larger than this many KB (default: 64)
"""

import json
import os
import time
from datetime import datetime, timezone

from mitmproxy import ctx, http


class TrafficLogger:
    # ------------------------------------------------------------------ config
    def load(self, loader):
        loader.add_option(
            name="log_file",
            typespec=str,
            default=os.path.join(os.path.dirname(__file__), "api_log.jsonl"),
            help="Path to the JSONL output log file.",
        )
        loader.add_option(
            name="filter_host",
            typespec=str,
            default="",
            help="Only log requests whose host contains this string (empty = all).",
        )
        loader.add_option(
            name="log_bodies",
            typespec=bool,
            default=True,
            help="Whether to record request/response bodies.",
        )
        loader.add_option(
            name="max_body_kb",
            typespec=int,
            default=64,
            help="Truncate bodies larger than this many KiB.",
        )

    def configure(self, updates):
        if "log_file" in updates:
            log_path = ctx.options.log_file
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            ctx.log.info(f"[TrafficLogger] Logging to: {log_path}")

    # --------------------------------------------------------------- helpers
    def _should_log(self, flow: http.HTTPFlow) -> bool:
        host_filter = ctx.options.filter_host
        if not host_filter:
            return True
        return host_filter.lower() in flow.request.pretty_host.lower()

    def _safe_body(self, raw: bytes, content_type: str = "") -> str | None:
        """Return body as string, truncating and handling binary content."""
        if not ctx.options.log_bodies or raw is None:
            return None
        limit = ctx.options.max_body_kb * 1024
        truncated = len(raw) > limit
        data = raw[:limit]
        try:
            text = data.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            text = data.hex()
        if truncated:
            text += f"\n... [TRUNCATED — original size: {len(raw)} bytes]"
        return text

    def _headers_dict(self, headers) -> dict:
        return {k: v for k, v in headers.items()}

    # ------------------------------------------------------------- main hook
    def response(self, flow: http.HTTPFlow):
        if not self._should_log(flow):
            return

        req = flow.request
        resp = flow.response

        if flow.request.timestamp_start and flow.response.timestamp_end:
            elapsed_ms = round(
                (flow.response.timestamp_end - flow.request.timestamp_start) * 1000, 1
            )
        else:
            elapsed_ms = None

        def try_json(raw, ct):
            text = self._safe_body(raw, ct)
            if text and ("json" in ct or text.lstrip().startswith(("{", "["))):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
            return text

        req_ct = req.headers.get("content-type", "")
        resp_ct = resp.headers.get("content-type", "") if resp else ""

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": elapsed_ms,
            "request": {
                "method": req.method,
                "url": req.pretty_url,
                "host": req.pretty_host,
                "path": req.path,
                "http_version": req.http_version,
                "headers": self._headers_dict(req.headers),
                "query": dict(req.query),
                "body": try_json(req.content, req_ct) if req.content else None,
            },
            "response": {
                "status_code": resp.status_code if resp else None,
                "reason": resp.reason if resp else None,
                "headers": self._headers_dict(resp.headers) if resp else {},
                "body": try_json(resp.content, resp_ct) if (resp and resp.content) else None,
            },
        }

        log_path = ctx.options.log_file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        status = resp.status_code if resp else "???"
        color = "\033[92m" if 200 <= (status or 0) < 300 else "\033[91m"
        reset = "\033[0m"
        ctx.log.info(
            f"{color}{req.method:6}{reset} {status} {elapsed_ms:>6}ms  {req.pretty_url}"
        )


addons = [TrafficLogger()]
