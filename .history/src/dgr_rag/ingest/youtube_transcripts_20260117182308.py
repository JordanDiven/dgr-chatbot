from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from xml.etree.ElementTree import ParseError

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# Some versions expose different throttling/blocking exception names. Make optional.
try:
    from youtube_transcript_api._errors import TooManyRequests  # type: ignore
except Exception:  # pragma: no cover
    TooManyRequests = None  # type: ignore

try:
    from youtube_transcript_api._errors import RequestBlocked  # type: ignore
except Exception:  # pragma: no cover
    RequestBlocked = None  # type: ignore

try:
    from youtube_transcript_api._errors import IpBlocked  # type: ignore
except Exception:  # pragma: no cover
    IpBlocked = None  # type: ignore

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

    Supports snippet objects with .start/.duration/.text (new API) and dicts
    with keys start/duration/text (fallback).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def get_start(s: Any) -> float:
        if hasattr(s, "start"):
            return float(s.start)
        return float(s.get("start", 0.0))

    def get_duration(s: Any) -> float:
        if hasattr(s, "duration"):
            return float(s.duration)
        return float(s.get("duration", 0.0))

    def get_text(s: Any) -> str:
        if hasattr(s, "text"):
            t = s.text
        else:
            t = s.get("text", "")
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


def _is_throttle_or_block_error(e: Exception) -> bool:
    if TooManyRequests is not None and isinstance(e, TooManyRequests):
        return True
    if RequestBlocked is not None and isinstance(e, RequestBlocked):
        return True
    if IpBlocked is not None and isinstance(e, IpBlocked):
        return True
    # Sometimes upstream responses cause XML parse errors; treat as transient.
    if isinstance(e, ParseError):
        return True
    # Fallback: some versions raise generic exceptions with these strings.
    msg = str(e).lower()
    if "too many requests" in msg or "429" in msg or "blocked" in msg:
        return True
    return False


def _fetch_with_retries(api: YouTubeTranscriptApi, video_id: str, *, max_tries: int = 6):
    """
    Fetches using the modern API if available:
      fetched = api.fetch(video_id)
      fetched.snippets -> iterable of snippet objects

    Retries with backoff on transient errors (throttling, blocks, parse errors).
    """
    delay = 1.0
    last_exc: Exception | None = None

    for attempt in range(1, max_tries + 1):
        try:
            if not hasattr(api, "fetch"):
                # If fetch is somehow missing, fall back to old API.
                return YouTubeTranscriptApi.get_transcript(video_id)

            return api.fetch(video_id)

        except Exception as e:
            last_exc = e
            if attempt == max_tries or not _is_throttle_or_block_error(e):
                raise
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= 2

    raise last_exc if last_exc else RuntimeError("Unknown transcript fetch failure")


def download_transcript(
    api: YouTubeTranscriptApi,
    *,
    episode_id: str,
    title: str,
    video_id: str,
    url: str,
    out_dir: Path,
    sleep_s: float = 0.8,
    overwrite: bool = False,
) -> TranscriptResult:
    """
    Downloads a transcript and writes:
      data/raw/transcripts/{episode_id}_{video_id}_{slug}.txt

    Returns a TranscriptResult with status and message.
    """
    slug = safe_filename(title)
    out_path = out_dir / f"{episode_id}_{video_id}_{slug}.txt"

    if out_path.exists() and not overwrite:
        return TranscriptResult(True, f"SKIP exists: {out_path.name}", out_path)

    meta = {
        "episode_id": episode_id,
        "title": title,
        "video_id": video_id,
        "url": url,
    }

    try:
        fetched = _fetch_with_retries(api, video_id)

        # New API shape
        if hasattr(fetched, "snippets"):
            snippets = fetched.snippets
        # Old API fallback shape (list[dict])
        elif isinstance(fetched, list):
            snippets = fetched
        else:
            return TranscriptResult(False, f"ERROR: {video_id} (Unexpected fetch type: {type(fetched)})")

        write_transcript_txt(out_path, meta=meta, snippets=snippets)

        # Small delay to reduce chance of throttling
        time.sleep(sleep_s)
        return TranscriptResult(True, f"OK: {out_path.name}", out_path)

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        return TranscriptResult(False, f"NO TRANSCRIPT: {video_id} ({type(e).__name__})")
    except VideoUnavailable as e:
        return TranscriptResult(False, f"UNAVAILABLE: {video_id} ({type(e).__name__})")
    except Exception as e:
        # If it's a throttle/block-like error and we still ended up here, label it clearly.
        if _is_throttle_or_block_error(e):
            return TranscriptResult(False, f"THROTTLED/BLOCKED: {video_id} ({type(e).__name__}: {e})")
        return TranscriptResult(False, f"ERROR: {video_id} ({type(e).__name__}: {e})")
