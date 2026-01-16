from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    TooManyRequests,
)

from dgr_rag.utils.io import safe_filename

@dataclass
class TranscriptResult:
    ok: bool
    message: str
    out_path: Optional[Path] = None

def write_transcript_txt(out_path: Path, *, meta: dict, snippets) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        # Minimal metadata header
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("\n")

        for snip in snippets:
            start = float(snip.start)
            duration = float(snip.duration)
            text = (snip.text or "").replace("\n", " ").strip()
            if not text:
                continue
            f.write(f"[{start:.2f} --> {start + duration:.2f}] {text}\n")

def download_transcript(
    api: YouTubeTranscriptApi,
    *,
    episode_id: str,
    title: str,
    video_id: str,
    url: str,
    out_dir: Path,
    sleep_s: float = 0.6,
    overwrite: bool = False,
) -> TranscriptResult:
    slug = safe_filename(title)
    out_path = out_dir / f"{episode_id}_{video_id}_{slug}.txt"

    if out_path.exists() and not overwrite:
        return TranscriptResult(True, f"SKIP exists: {out_path.name}", out_path)

    try:
        fetched = api.fetch(video_id)
        meta = {
            "episode_id": episode_id,
            "title": title,
            "video_id": video_id,
            "url": url,
        }
        write_transcript_txt(out_path, meta=meta, snippets=fetched.snippets)
        time.sleep(sleep_s)
        return TranscriptResult(True, f"OK: {out_path.name}", out_path)

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        return TranscriptResult(False, f"NO TRANSCRIPT: {video_id} ({type(e).__name__})")
    except VideoUnavailable as e:
        return TranscriptResult(False, f"UNAVAILABLE: {video_id} ({type(e).__name__})")
    except TooManyRequests as e:
        time.sleep(10)
        return TranscriptResult(False, f"THROTTLED: {video_id} ({type(e).__name__})")
    except Exception as e:
        return TranscriptResult(False, f"ERROR: {video_id} ({type(e).__name__}: {e})")