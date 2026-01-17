from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Any
from types import SimpleNamespace
import random

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


def write_transcript_txt(out_path: Path, *, meta: dict, snippets: Iterable[Any]) -> None:
    """
    Writes transcript in a stable format:
      # metadata...
      [start --> end] text
    Supports snippet objects with .start/.duration/.text OR dicts with keys start/duration/text.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def get_start(s):
        return float(getattr(s, "start", s.get("start", 0.0)))

    def get_duration(s):
        return float(getattr(s, "duration", s.get("duration", 0.0)))

    def get_text(s):
        t = getattr(s, "text", s.get("text", ""))
        return (t or "").replace("\n", " ").strip()

    with out_path.open("w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("\n")

        for snip in snippets:
            text = get_text(snip)
            if not text:
                continue
            start = get_start(snip)
            duration = get_duration(snip)
            f.write(f"[{start:.2f} --> {start + duration:.2f}] {text}\n")


def _fetch_with_retries(api: YouTubeTranscriptApi, video_id: str, *, max_tries: int = 5):
    """
    Fetch transcript with basic backoff to handle intermittent failures / throttling.
    Uses new API (api.fetch) when available, otherwise falls back to get_transcript.
    """
    delay = 1.0
    last_exc: Exception | None = None

    for attempt in range(1, max_tries + 1):
        try:
            if hasattr(api, "fetch"):
                return api.fetch(video_id)  # NEW API: returns object with .snippets
            # Fallback: older API returns list[dict]
            return YouTubeTranscriptApi.get_transcript(video_id)

        except (TooManyRequests,) as e:
            last_exc = e
            if attempt == max_tries:
                raise
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= 2

        except Exception as e:
            # For transient parsing/network oddities, retry a few times.
            last_exc = e
            if attempt == max_tries:
                raise
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= 2

    raise last_exc if last_exc else RuntimeError("Unknown fetch failure")


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
        fetched = _fetch_with_retries(api, video_id)

        # NEW API shape: fetched.snippets
        if hasattr(fetched, "snippets"):
            snippets = fetched.snippets
        # Old API shape: list[dict]
        elif isinstance(fetched, list):
            snippets = fetched
        else:
            return TranscriptResult(False, f"ERROR: {video_id} (Unexpected fetch type: {type(fetched)})")

        meta = {
            "episode_id": episode_id,
            "title": title,
            "video_id": video_id,
            "url": url,
        }

        write_transcript_txt(out_path, meta=meta, snippets=snippets)
        time.sleep(sleep_s)
        return TranscriptResult(True, f"OK: {out_path.name}", out_path)

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        return TranscriptResult(False, f"NO TRANSCRIPT: {video_id} ({type(e).__name__})")
    except VideoUnavailable as e:
        return TranscriptResult(False, f"UNAVAILABLE: {video_id} ({type(e).__name__})")
    except TooManyRequests as e:
        return TranscriptResult(False, f"THROTTLED: {video_id} ({type(e).__name__})")
    except Exception as e:
        return TranscriptResult(False, f"ERROR: {video_id} ({type(e).__name__}: {e})")
