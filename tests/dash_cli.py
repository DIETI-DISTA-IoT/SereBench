#!/usr/bin/env python3
"""CLI client for SereBench dashboard endpoints.

This tool mirrors the HTTP requests sent by the dashboard web GUI,
while making configuration editable from the command line and persisted on disk.

The dashboard runs inside a Docker container (port 5000 exposed to the host).
All commands are issued from the host to http://localhost:5000 (configurable).

Typical workflow
----------------
# 1. Bring up the infrastructure (Kafka, Zookeeper, wandber, dashboard)
#    docker compose up -d

# 2. Initialise CLI state from Hydra config
#    python tests/dash_cli.py init-config

# 3. (optional) Inspect / tweak values
#    python tests/dash_cli.py show-config
#    python tests/dash_cli.py set federated_learning.aggregation_strategy fedyogi

# 4. Create vehicle containers, run the full experiment, then tear down
#    python tests/dash_cli.py create-vehicles
#    python tests/dash_cli.py start-experiment
#    python tests/dash_cli.py shutdown
#    python tests/dash_cli.py delete-vehicles
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml

try:
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf
    _HYDRA_AVAILABLE = True
except ImportError:
    _HYDRA_AVAILABLE = False

DEFAULT_STATE_PATH = Path.home() / ".serebench" / "dash_cli_state.yaml"
DEFAULT_CONFIG_DIR = (Path(__file__).resolve().parents[1] / "config").resolve()
DEFAULT_BASE_URL = "http://localhost:5000"


@dataclass
class HttpResult:
    method: str
    url: str
    status: int
    body: Any


class SereBenchCLI:
    def __init__(self, base_url: str, state_file: Path, config_dir: Path, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.state_file = state_file
        self.config_dir = config_dir
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def load_hydra_config(self, profile: str | None, hydra_overrides: list[str]) -> "DictConfig":
        if not _HYDRA_AVAILABLE:
            raise RuntimeError(
                "hydra-core and omegaconf are required for config loading.\n"
                "Install them: pip install hydra-core omegaconf"
            )
        overrides = list(hydra_overrides)
        if profile:
            overrides.append(f"override={profile}")

        with initialize_config_dir(config_dir=str(self.config_dir), version_base="1.2"):
            cfg = compose(config_name="default", overrides=overrides)

        override_name = str(cfg.get("override", "") or "").strip()
        if override_name:
            override_file = self.config_dir / "overrides" / f"{override_name}.yaml"
            if not override_file.exists():
                raise FileNotFoundError(
                    f"Configuration override not found: {override_file}. "
                    "Expected config/overrides/<profile>.yaml"
                )
            file_override = OmegaConf.load(str(override_file))
            cfg = OmegaConf.merge(cfg, file_override)

        return cfg

    def config_to_dict(self, cfg: "DictConfig") -> dict[str, Any]:
        if not _HYDRA_AVAILABLE:
            raise RuntimeError("omegaconf required")
        result = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(result, dict)
        return result

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def load_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {}
        return yaml.safe_load(self.state_file.read_text()) or {}

    def save_state(self, data: dict[str, Any]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(yaml.safe_dump(data, sort_keys=False))

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def http(self, method: str, path: str, *, json_payload: Any | None = None) -> HttpResult:
        url = f"{self.base_url}{path}"
        print(f"[waiting] {method.upper()} {url} ...", flush=True)
        start = time.monotonic()
        try:
            response = self.session.request(
                method=method, url=url, json=json_payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise
        elapsed = time.monotonic() - start
        print(f"[done] HTTP {response.status_code} in {elapsed:.2f}s", flush=True)

        try:
            body: Any = response.json()
        except Exception:
            body = response.text

        return HttpResult(method=method.upper(), url=url, status=response.status_code, body=body)


# ------------------------------------------------------------------
# Dot-path helpers (for set / unset / show-config)
# ------------------------------------------------------------------

def infer_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def set_by_dotpath(data: dict[str, Any], dotpath: str, value: Any) -> None:
    keys = dotpath.split(".")
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        node = cursor.get(key)
        if not isinstance(node, dict):
            node = {}
            cursor[key] = node
        cursor = node
    cursor[keys[-1]] = value


def has_dotpath(data: dict[str, Any], dotpath: str) -> bool:
    keys = dotpath.split(".")
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return False
        cursor = cursor[key]
    return True


def collect_dotpaths(data: dict[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in data.items():
        current = f"{prefix}.{key}" if prefix else key
        paths.add(current)
        if isinstance(value, dict):
            paths.update(collect_dotpaths(value, current))
    return paths


def delete_by_dotpath(data: dict[str, Any], dotpath: str) -> bool:
    keys = dotpath.split(".")
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        node = cursor.get(key)
        if not isinstance(node, dict):
            return False
        cursor = node
    return cursor.pop(keys[-1], None) is not None


# ------------------------------------------------------------------
# Output formatting
# ------------------------------------------------------------------

def pretty_print_result(result: HttpResult) -> None:
    print(f"\n=== {result.method} {result.url} ===")
    print(f"HTTP status: {result.status}")
    print("Response:")
    if isinstance(result.body, (dict, list)):
        print(json.dumps(result.body, indent=2, sort_keys=True))
    else:
        print(result.body)
    print("=" * 60)


# ------------------------------------------------------------------
# W&B helpers
# ------------------------------------------------------------------

def read_wandb_api_key(repo_root: Path) -> str | None:
    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        return env_key.strip()
    env_file = repo_root / ".env"
    if not env_file.exists():
        return None
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "WANDB_API_KEY":
            return value.strip().strip("'\"")
    return None


def query_wandb_run(
    entity: str,
    project: str,
    run_name: str,
    api_key: str,
    top_n_runs: int = 50,
    timeout_secs: int = 30,
    retries: int = 2,
    endpoint: str = "https://api.wandb.ai/graphql",
) -> dict[str, Any] | None:
    query = """
    query ProjectRuns($entity: String!, $project: String!, $first: Int!) {
      project(name: $project, entityName: $entity) {
        runs(first: $first, order: "-created_at") {
          edges {
            node {
              id
              name
              displayName
              state
              updatedAt
              summaryMetrics
            }
          }
        }
      }
    }
    """
    payload = {
        "query": query,
        "variables": {"entity": entity, "project": project, "first": top_n_runs},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    data: dict[str, Any] | None = None
    last_error: requests.RequestException | None = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        print(f"[waiting] W&B poll {attempt}/{attempts} ...", flush=True)
        start = time.monotonic()
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_secs)
            response.raise_for_status()
            data = response.json()
            elapsed = time.monotonic() - start
            print(f"[done] W&B poll succeeded in {elapsed:.2f}s", flush=True)
            break
        except requests.RequestException as exc:
            last_error = exc
            elapsed = time.monotonic() - start
            print(f"[warn] W&B poll failed in {elapsed:.2f}s: {exc}", flush=True)
            if attempt < attempts:
                backoff = min(10, 2 ** (attempt - 1))
                print(f"[waiting] retrying in {backoff}s ...", flush=True)
                time.sleep(backoff)

    if data is None:
        assert last_error is not None
        raise last_error

    edges = (
        (data.get("data", {}).get("project", {}) or {})
        .get("runs", {})
        .get("edges", [])
    )
    for edge in edges:
        node = edge.get("node", {})
        if node.get("displayName") == run_name or node.get("name") == run_name:
            return node
    return None


def print_wandb_summary(run: dict[str, Any], max_metrics: int = 25) -> None:
    try:
        summary_raw = run.get("summaryMetrics", {})
        if isinstance(summary_raw, str):
            summary_raw = json.loads(summary_raw)
        numeric = {
            k: float(v)
            for k, v in summary_raw.items()
            if not k.startswith("_") and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        summary = dict(sorted(numeric.items())[:max_metrics])
    except Exception:
        summary = {}

    print("\n=== Weights & Biases run monitor ===")
    print(f"Run id:          {run.get('id')}")
    print(f"Run name:        {run.get('name')}")
    print(f"Run displayName: {run.get('displayName')}")
    print(f"Run state:       {run.get('state')}")
    print(f"Updated at:      {run.get('updatedAt')}")
    if not summary:
        print("No numeric metrics found in W&B summaryMetrics yet.")
        return
    print("Metric summary (latest numeric values):")
    print(json.dumps(summary, indent=2, sort_keys=True))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SereBench dashboard CLI — control experiments without the web GUI."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Dashboard HTTP base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Persistent CLI config state file (YAML)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Hydra config directory (default: <repo_root>/config)",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="config/overrides/<profile>.yaml to merge (e.g. dev, fedavg)",
    )
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Hydra override (repeatable)",
    )
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- config management ---
    sub.add_parser("init-config", help="(Re)initialize CLI state from Hydra config")
    sub.add_parser("show-config", help="Print the current persisted config state")

    p_set = sub.add_parser("set", help="Set a config value by dot-path (e.g. federated_learning.aggregation_strategy fedyogi)")
    p_set.add_argument("key")
    p_set.add_argument("value", help="YAML/JSON scalar, list, or object")

    p_unset = sub.add_parser("unset", help="Remove a config key by dot-path")
    p_unset.add_argument("key")

    # --- vehicle lifecycle ---
    sub.add_parser("create-vehicles", help="Create all vehicle containers via the dashboard")
    sub.add_parser("delete-vehicles", help="Delete all vehicle containers")

    # --- producer / consumer control ---
    sub.add_parser("produce-all", help="Start all producers")
    sub.add_parser("stop-producing-all", help="Stop all producers")
    sub.add_parser("consume-all", help="Start all consumers")
    sub.add_parser("stop-consuming-all", help="Stop all consumers")

    # --- federated learning ---
    sub.add_parser("start-federated-learning", help="Start FL aggregation")
    sub.add_parser("stop-federated-learning", help="Stop FL aggregation")

    # --- W&B / wandber ---
    sub.add_parser("start-wandb", help="Start W&B logging (sends wandb config from state)")
    sub.add_parser("stop-wandb", help="Stop W&B logging")

    # --- security manager ---
    sub.add_parser("start-security-manager", help="Start the security manager (sends security_manager config from state)")
    sub.add_parser("stop-security-manager", help="Stop the security manager")

    # --- mitigation ---
    sub.add_parser("start-mitigation", help="Enable attack mitigation")
    sub.add_parser("stop-mitigation", help="Disable attack mitigation")

    # --- attacks ---
    p_attack_start = sub.add_parser("start-attack", help="Start attack from a specific vehicle")
    p_attack_start.add_argument("vehicle_name", help="Vehicle name (e.g. angela)")

    p_attack_stop = sub.add_parser("stop-attack", help="Stop attack from a specific vehicle")
    p_attack_stop.add_argument("vehicle_name", help="Vehicle name (e.g. angela)")
    p_attack_stop.add_argument("--origin", default="MANUAL", help="Stop origin label (default: MANUAL)")

    sub.add_parser("start-automatic-attacks", help="Start automatic (scheduled) attack injection")
    sub.add_parser("stop-automatic-attacks", help="Stop automatic attack injection")
    sub.add_parser("start-preconf-attack", help="Start pre-configured attack campaign")
    sub.add_parser("stop-preconf-attack", help="Stop pre-configured attack campaign")

    # --- full experiment lifecycle ---
    sub.add_parser(
        "start-experiment",
        help=(
            "Run the full experiment sequence: produce-all → consume-all → "
            "security-manager → FL → wandb → automatic-attacks"
        ),
    )
    sub.add_parser(
        "shutdown",
        help="Graceful shutdown: stop everything in reverse order",
    )

    # --- status / logs ---
    sub.add_parser("health", help="Health check the dashboard container")

    p_vstatus = sub.add_parser("vehicle-status", help="Get status of a specific vehicle")
    p_vstatus.add_argument("vehicle_name", help="Vehicle name (e.g. angela)")

    p_logs = sub.add_parser("logs", help="Tail logs from a container (via dashboard)")
    p_logs.add_argument("container_name", help="Docker container name")
    p_logs.add_argument(
        "--since",
        type=float,
        default=None,
        help="Unix timestamp (ms); only show logs after this time",
    )
    p_logs.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll and print new log lines every --interval-secs seconds",
    )
    p_logs.add_argument("--interval-secs", type=int, default=5, help="Polling interval when --watch is set")

    # --- W&B monitor ---
    p_wandb_mon = sub.add_parser("wandb-monitor", help="Poll W&B GraphQL API and print experiment metric summary")
    p_wandb_mon.add_argument("--entity", default="", help="W&B entity (org/user). Falls back to state wandb.entity")
    p_wandb_mon.add_argument("--project", default="", help="W&B project name. Falls back to state wandb.project_name")
    p_wandb_mon.add_argument("--run-name", default="", help="W&B run display name. Falls back to state wandb.run_name")
    p_wandb_mon.add_argument("--top-n-runs", type=int, default=50)
    p_wandb_mon.add_argument("--max-metrics", type=int, default=30)
    p_wandb_mon.add_argument("--watch", action="store_true", help="Continuously poll")
    p_wandb_mon.add_argument("--interval-secs", type=int, default=15)
    p_wandb_mon.add_argument("--wandb-timeout-secs", type=int, default=30)
    p_wandb_mon.add_argument("--wandb-retries", type=int, default=2)
    p_wandb_mon.add_argument("--wandb-endpoint", default="https://api.wandb.ai/graphql")

    args = parser.parse_args()

    client = SereBenchCLI(
        base_url=args.base_url,
        state_file=args.state_file,
        config_dir=args.config_dir,
        timeout=args.timeout,
    )

    # ------------------------------------------------------------------
    # init-config
    # ------------------------------------------------------------------
    if args.command == "init-config":
        cfg = client.load_hydra_config(args.profile or None, args.hydra_override)
        state = client.config_to_dict(cfg)
        client.save_state(state)
        print(f"Initialized config state at: {args.state_file}")
        print(f"Dashboard URL: {client.base_url}")
        print("Tip: use `show-config`, `set`, `unset` to tune before sending requests.")
        return 0

    # ------------------------------------------------------------------
    # show-config
    # ------------------------------------------------------------------
    if args.command == "show-config":
        state = client.load_state()
        if not state:
            print("No persisted state found. Run `init-config` first.")
            return 1
        print(yaml.safe_dump(state, sort_keys=False))
        return 0

    # Auto-init state if missing
    state = client.load_state()
    if not state and args.command not in {"health", "vehicle-status", "logs"}:
        if _HYDRA_AVAILABLE:
            cfg = client.load_hydra_config(args.profile or None, args.hydra_override)
            state = client.config_to_dict(cfg)
            client.save_state(state)
            print(f"No state file found — auto-initialized from Hydra at: {args.state_file}")
        else:
            print(
                "No state file found and hydra-core is not installed.\n"
                "Either run `init-config` manually or install hydra-core."
            )

    # ------------------------------------------------------------------
    # set / unset
    # ------------------------------------------------------------------
    if args.command == "set":
        if _HYDRA_AVAILABLE:
            cfg = client.load_hydra_config(args.profile or None, args.hydra_override)
            allowed_schema = client.config_to_dict(cfg)
            if not has_dotpath(allowed_schema, args.key):
                known = sorted(collect_dotpaths(allowed_schema))
                suggestions = difflib.get_close_matches(args.key, known, n=3, cutoff=0.55)
                print(f"Error: `{args.key}` does not exist in the composed config schema.")
                if suggestions:
                    print("Did you mean:")
                    for s in suggestions:
                        print(f"  - {s}")
                else:
                    print("Use `show-config` to inspect valid keys.")
                return 1
        set_by_dotpath(state, args.key, infer_value(args.value))
        client.save_state(state)
        print(f"Set `{args.key}` and saved {args.state_file}")
        return 0

    if args.command == "unset":
        removed = delete_by_dotpath(state, args.key)
        client.save_state(state)
        if removed:
            print(f"Unset `{args.key}` and saved {args.state_file}")
            return 0
        print(f"Key `{args.key}` was not present.")
        return 1

    # ------------------------------------------------------------------
    # wandb-monitor
    # ------------------------------------------------------------------
    if args.command == "wandb-monitor":
        api_key = read_wandb_api_key(Path(__file__).resolve().parents[1])
        if not api_key:
            print(
                "WANDB_API_KEY not found. Set it in the environment or in <repo>/.env "
                "as WANDB_API_KEY=<key>"
            )
            return 1

        wandb_cfg = state.get("wandb", {})
        entity = args.entity or wandb_cfg.get("entity", "")
        project = args.project or wandb_cfg.get("project_name", "")
        run_name = args.run_name or wandb_cfg.get("run_name", "")

        if not entity or not project or not run_name:
            print(
                "W&B entity / project_name / run_name are required.\n"
                "Set them via --entity / --project / --run-name flags, or in the config state "
                "(wandb.entity, wandb.project_name, wandb.run_name)."
            )
            return 1

        def do_poll() -> None:
            run = query_wandb_run(
                entity=entity,
                project=project,
                run_name=run_name,
                api_key=api_key,
                top_n_runs=args.top_n_runs,
                timeout_secs=args.wandb_timeout_secs,
                retries=args.wandb_retries,
                endpoint=args.wandb_endpoint,
            )
            if not run:
                print(f"No W&B run found for {entity}/{project}/{run_name}")
                return
            print_wandb_summary(run, max_metrics=args.max_metrics)

        if args.watch:
            try:
                while True:
                    try:
                        do_poll()
                    except requests.RequestException as exc:
                        print(f"[warn] W&B polling error: {exc}")
                    time.sleep(args.interval_secs)
            except KeyboardInterrupt:
                print("\nStopped W&B monitoring.")
            return 0

        try:
            do_poll()
        except requests.RequestException as exc:
            print(f"W&B polling failed: {exc}")
            return 2
        return 0

    # ------------------------------------------------------------------
    # logs (special: uses GET, streaming-friendly)
    # ------------------------------------------------------------------
    if args.command == "logs":
        path = f"/logs/{args.container_name}"

        def fetch_logs(since: float | None) -> str:
            params = {}
            if since is not None:
                params["since"] = since
            url = f"{client.base_url}{path}"
            resp = client.session.get(url, params=params, timeout=args.timeout)
            resp.raise_for_status()
            return resp.text

        if args.watch:
            last_ts: float | None = args.since
            try:
                while True:
                    try:
                        text = fetch_logs(last_ts)
                        if text.strip():
                            print(text, end="")
                        last_ts = time.time() * 1000  # millis for next call
                    except requests.RequestException as exc:
                        print(f"[warn] log fetch error: {exc}")
                    time.sleep(args.interval_secs)
            except KeyboardInterrupt:
                print("\nStopped log tailing.")
            return 0

        try:
            print(fetch_logs(args.since))
        except requests.RequestException as exc:
            print(f"Log fetch failed: {exc}")
            return 2
        return 0

    # ------------------------------------------------------------------
    # Simple endpoint dispatch table
    # ------------------------------------------------------------------
    # (method, path, payload_factory)
    # payload_factory is a callable(state) -> dict | None
    simple_routes: dict[str, tuple[str, str, Any]] = {
        # health
        "health": ("GET", "/health", None),
        # vehicle lifecycle
        "create-vehicles": ("POST", "/create-vehicles", None),
        "delete-vehicles": ("POST", "/delete-vehicles", None),
        # producers / consumers
        "produce-all": ("POST", "/produce-all", None),
        "stop-producing-all": ("POST", "/stop-producing-all", None),
        "consume-all": ("POST", "/consume-all", None),
        "stop-consuming-all": ("POST", "/stop-consuming-all", None),
        # FL
        "start-federated-learning": ("POST", "/start-federated-learning", None),
        "stop-federated-learning": ("POST", "/stop-federated-learning", None),
        # attacks
        "start-automatic-attacks": ("POST", "/start-automatic-attacks", None),
        "stop-automatic-attacks": ("POST", "/stop-automatic-attacks", None),
        "start-preconf-attack": ("POST", "/start-preconf-attack", None),
        "stop-preconf-attack": ("POST", "/stop-preconf-attack", None),
        # mitigation
        "start-mitigation": ("POST", "/start-mitigation", None),
        "stop-mitigation": ("POST", "/stop-mitigation", None),
        # wandb
        "stop-wandb": ("POST", "/stop-wandb", None),
        # security manager
        "stop-security-manager": ("POST", "/stop-security-manager", None),
        # experiment
        "start-experiment": ("POST", "/start-experiment", None),
        "shutdown": ("POST", "/shutdown", None),
    }

    if args.command in simple_routes:
        method, path, _ = simple_routes[args.command]
        try:
            result = client.http(method, path)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            return 2
        pretty_print_result(result)
        return 0 if 200 <= result.status < 300 else 2

    # ------------------------------------------------------------------
    # Commands that need a payload built from state
    # ------------------------------------------------------------------

    if args.command == "start-wandb":
        payload = {"wandb": state.get("wandb", {})}
        try:
            result = client.http("POST", "/start-wandb", json_payload=payload)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            return 2
        pretty_print_result(result)
        return 0 if 200 <= result.status < 300 else 2

    if args.command == "start-security-manager":
        payload = {"security_manager": state.get("security_manager", {})}
        try:
            result = client.http("POST", "/start-security-manager", json_payload=payload)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            return 2
        pretty_print_result(result)
        return 0 if 200 <= result.status < 300 else 2

    if args.command == "vehicle-status":
        payload = {"vehicle_name": args.vehicle_name}
        try:
            result = client.http("POST", "/vehicle-status", json_payload=payload)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            return 2
        pretty_print_result(result)
        return 0 if 200 <= result.status < 300 else 2

    if args.command == "start-attack":
        payload = {"vehicle_name": args.vehicle_name}
        try:
            result = client.http("POST", "/start-attack", json_payload=payload)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            return 2
        pretty_print_result(result)
        return 0 if 200 <= result.status < 300 else 2

    if args.command == "stop-attack":
        payload = {"vehicle_name": args.vehicle_name, "origin": args.origin}
        try:
            result = client.http("POST", "/stop-attack", json_payload=payload)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            return 2
        pretty_print_result(result)
        return 0 if 200 <= result.status < 300 else 2

    print(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
