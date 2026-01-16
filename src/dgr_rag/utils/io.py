from __future__ import annotations

import re
from pathlib import Path

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def safe_filename(text: str, max_len: int = 120) -> str:
    text = text.strip()
    if not text:
        return "untitled"

    # Keep alphanum, spaces, hyphen, underscore
    text = re.sub(r"[^A-Za-z0-9 _-]+", "", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:max_len] if text else "untitled"