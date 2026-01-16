from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_dir: Path
    episode_index_csv: Path
    raw_playlist_dir: Path
    raw_transcripts_dir: Path
    logs_dir: Path

def get_paths() -> Paths:
    # Repo root = two levels up from this file: src/dgr_rag/config.py
    repo_root = Path(__file__).resolve().parents[2]

    data_dir_env = os.getenv("DATA_DIR", "").strip()
    data_dir = Path(data_dir_env) if data_dir_env else (repo_root / "data")

    episode_index_csv = data_dir / "episode_index.csv"
    raw_playlist_dir = data_dir / "raw" / "playlist"
    raw_transcripts_dir = data_dir / "raw" / "transcripts"
    logs_dir = data_dir / "logs"

    return Paths(
        repo_root=repo_root,
        data_dir=data_dir,
        episode_index_csv=episode_index_csv,
        raw_playlist_dir=raw_playlist_dir,
        raw_transcripts_dir=raw_transcripts_dir,
        logs_dir=logs_dir,
    )
