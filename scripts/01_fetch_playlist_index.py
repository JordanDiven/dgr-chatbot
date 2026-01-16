from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dgr_rag.config import get_paths
from dgr_rag.ingest.episode_index import EpisodeRow, write_episode_index
from dgr_rag.ingest.playlist import dump_playlist_json, load_playlist_json

def main() -> None:
    playlist_url = os.getenv("PLAYLIST_URL", "").strip()
    if not playlist_url:
        raise RuntimeError("PLAYLIST_URL is not set. Add it to your .env file.")

    paths = get_paths()
    paths.raw_playlist_dir.mkdir(parents=True, exist_ok=True)

    out_json = paths.raw_playlist_dir / "playlist.json"
    dump_playlist_json(playlist_url, out_json)
    data = load_playlist_json(out_json)

    entries = data.get("entries", [])
    if not entries:
        raise RuntimeError("No entries found in playlist JSON. Check playlist URL.")

    rows: list[EpisodeRow] = []
    for i, e in enumerate(entries, start=1):
        video_id = (e.get("id") or "").strip()
        title = (e.get("title") or "").strip()
        url = (e.get("url") or "").strip()
        if not url and video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"

        upload_date = (e.get("upload_date") or "").strip()  # may be blank in flat mode
        published_at = ""
        if len(upload_date) == 8:
            published_at = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

        rows.append(EpisodeRow(
            episode_id=f"{i:03d}",
            title=title,
            youtube_video_id=video_id,
            url=url,
            published_at=published_at,
        ))

    write_episode_index(paths.episode_index_csv, rows)
    print(f"Wrote {len(rows)} rows -> {paths.episode_index_csv}")
    print(f"Playlist JSON saved -> {out_json}")

if __name__ == "__main__":
    main()