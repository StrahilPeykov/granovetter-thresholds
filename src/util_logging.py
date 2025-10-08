import json
import os
import subprocess
from datetime import datetime, timezone


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_git_commit_short() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def utc_timestamp_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
