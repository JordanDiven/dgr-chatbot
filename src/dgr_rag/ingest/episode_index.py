from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

@dataclass
class EpisodeRow:
    episode_id: str
    title: str
    youtube_video_id: str
    url: str
    published_at: str = ""

EPISODE_FIELDS = ["episode_id", "title", "youtube_video_id", "url", "published_at"]

def write_episode_index(csv_path: Path, rows: Iterable[EpisodeRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "episode_id": r.episode_id,
                "title": r.title,
                "youtube_video_id": r.youtube_video_id,
                "url": r.url,
                "published_at": r.published_at,
            })

def read_episode_index(csv_path: Path) -> list[EpisodeRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[EpisodeRow] = []
        for row in reader:
            rows.append(EpisodeRow(
                episode_id=(row.get("episode_id") or "").strip(),
                title=(row.get("title") or "").strip(),
                youtube_video_id=(row.get("youtube_video_id") or "").strip(),
                url=(row.get("url") or "").strip(),
                published_at=(row.get("published_at") or "").strip(),
            ))
    return rows