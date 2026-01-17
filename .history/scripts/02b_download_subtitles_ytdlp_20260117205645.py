from __future__ import annotations

import subprocess
from pathlib import Path
from tqdm import tqdm

from dgr_rag.config import get_paths
from dgr_rag.ingest.episode_index import read_episode_index

def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def main() -> None:
    paths = get_paths()
    episodes = read_episode_index(paths.episode_index_csv)

    out_dir = paths.data_dir / "raw" / "subtitles"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0

    for ep in tqdm(episodes, desc="Downloading subtitles (yt-dlp)"):
        url = ep.url or f"https://www.youtube.com/watch?v={ep.youtube_video_id}"

        # Output template: store as data/raw/subtitles/<video_id>.<ext> (yt-dlp appends lang automatically)
        out_tmpl = str(out_dir / "%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-sub",
            "--write-auto-sub",
            "--sub-lang", "en.*",
            "--sub-format", "vtt",
            "-o", out_tmpl,
            url,
        ]

        code, _, err = run_cmd(cmd)
        if code == 0:
            ok += 1
        else:
            fail += 1
            print(f"FAIL: {ep.youtube_video_id} ({ep.title})\n{err}\n")

    print(f"Done. OK={ok}, FAIL={fail}, TOTAL={len(episodes)}")

if __name__ == "__main__":
    main()
