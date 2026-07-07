# api-recon-template

A GitHub template for reverse-engineering macOS app APIs via HTTPS traffic interception. Captures traffic with mitmproxy, logs it to JSONL, and provides a scaffold for building a typed async API client.

**Reference implementation:** the SwingVision project this template was extracted from is a working example of everything described here.

---

## Using this template

### 1. Create your repo from this template

Click **"Use this template"** on GitHub (or clone directly).

### 2. Run init.sh

```bash
bash init.sh \
  --app-name  "Peloton" \
  --app-slug  "peloton" \
  --api-host  "api.onepeloton.com" \
  --user-agent "Peloton/3.x CFNetwork/..."
```

`init.sh` substitutes all placeholder sentinels, renames `src/peloton/` to `src/peloton/`, and prints a checklist of remaining manual steps.

### 3. Complete manual TODOs

See [CUSTOMIZATION.md](CUSTOMIZATION.md) for the five steps that `init.sh` cannot automate (they require inspecting captured traffic).

### 4. Capture traffic

```bash
bash run_capture.sh
```

Open the target app and exercise the features you want to reverse-engineer, then `Ctrl-C` to stop. Traffic is written to `api_log.jsonl`.

### 5. Analyze

```bash
python3 analyze_log.py                    # summary
python3 analyze_log.py --endpoints        # deduplicated endpoint list
python3 analyze_log.py --method POST      # filter by method
python3 analyze_log.py --search "login"   # search bodies
python3 analyze_log.py --dump             # full JSON per entry
```

---

## Template structure

```
├── init.sh                  Infrastructure — run once to customize
├── run_capture.sh           Infrastructure — start/stop traffic capture
├── traffic_logger.py        Infrastructure — mitmproxy addon (JSONL logger)
├── analyze_log.py           Infrastructure — CLI log analyzer
├── pyproject.toml           Infrastructure — package metadata
│
├── bruno_collection/        API explorer — import into Bruno app
│   ├── environments/        Per-target environment vars (baseUrl, authToken)
│   └── Session/ping.bru     Example request (update body/headers after init)
│
└── src/peloton/          Scaffold — fill in after capturing traffic
    ├── config.py            Pydantic Settings (env vars, base URL)
    ├── auth.py              Token cache + login flow (has TODO stubs)
    ├── client.py            Async httpx client (add endpoint methods)
    ├── models.py            Pydantic response models (add your types)
    └── tasks/
        ├── manifest.py      File-locked task queue (generic, keep as-is)
        ├── definitions.py   Task type registry (register your tasks here)
        └── runner.py        Task executor (generic, keep as-is)
```

### Infrastructure vs. Scaffold

| File | What to do |
|---|---|
| `run_capture.sh` | Leave as-is after init |
| `traffic_logger.py` | Leave as-is |
| `analyze_log.py` | Leave as-is |
| `tasks/manifest.py` | Leave as-is |
| `tasks/runner.py` | Leave as-is |
| `config.py` | Leave as-is after init |
| `auth.py` | Fill in `_login_and_capture()` |
| `client.py` | Add endpoint methods |
| `models.py` | Define response models |
| `tasks/definitions.py` | Register task types |

---

## Requirements

- macOS (proxy setup uses `networksetup` and `security`)
- [mitmproxy](https://mitmproxy.org/) (`brew install mitmproxy`)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

```bash
brew install mitmproxy
uv sync
```
