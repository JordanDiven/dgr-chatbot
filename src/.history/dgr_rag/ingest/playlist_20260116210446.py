from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

def dump_playlist_json(playlist_url: str, out_json: Path) -> Path:
    """
    Uses yt-dlp to dump a playlist's entries as a single JSON object.
    Requires yt-dlp installed (in requirements.txt).
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        playlist_url,
    ]

    # Capture stdout -> file
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (code {result.returncode}).\nSTDERR:\n{result.stderr}"
        )

    out_json.write_text(result.stdout, encoding="utf-8")
    return out_json

def load_playlist_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))