from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Matches: [12.34 --> 15.67] some text
TS_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)\s*-->\s*(?P<end>\d+(?:\.\d+)?)\]\s*(?P<text>.*)$"
)

# Chunking profiles. "guidance" favors fewer, cleaner blocks over quote-friendly overlap.
CHUNK_PROFILES = {
    "guidance": {
        "max_chunk_seconds": 420.0,  # ~7 minutes
        "max_chunk_chars": 1200,     # keep embeddings focused
        "overlap_seconds": 0.0,
        "min_chunk_seconds": 90.0,
        "min_chunk_chars": 400,
        "max_gap_seconds": 12.0,
    },
    "quotes": {
        "max_chunk_seconds": 240.0,  # ~4 minutes
        "max_chunk_chars": 3500,     # rough proxy for token count
        "overlap_seconds": 30.0,
        "min_chunk_seconds": 0.0,
        "min_chunk_chars": 0,
        "max_gap_seconds": 0.0,
    },
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


def parse_transcript_file(path: Path) -> Tuple[Dict[str, str], List[Segment]]:
    """
    Parses the saved transcript format:
      # key: value
      ...
      blank line
      [start --> end] text
    """
    meta: Dict[str, str] = {}
    segments: List[Segment] = []

    lines = path.read_text(encoding="utf-8").splitlines()

    i = 0
    # metadata header
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            break
        if line.startswith("#"):
            # "# key: value"
            content = line[1:].strip()
            if ":" in content:
                k, v = content.split(":", 1)
                meta[k.strip()] = v.strip()
        i += 1

    # segment lines
    for j in range(i, len(lines)):
        line = lines[j].strip()
        if not line:
            continue
        m = TS_LINE_RE.match(line)
        if not m:
            continue
        start = float(m.group("start"))
        end = float(m.group("end"))
        text = m.group("text").strip()
        if text:
            segments.append(Segment(start=start, end=end, text=text))

    return meta, segments


def build_chunks(
    segments: List[Segment],
    *,
    max_chunk_seconds: float = 240.0,     # ~4 minutes
    max_chunk_chars: int = 3500,          # rough proxy for token count
    overlap_seconds: float = 30.0,        # overlap window
    min_chunk_seconds: float = 0.0,       # avoid tiny chunks
    min_chunk_chars: int = 0,             # avoid tiny chunks
    max_gap_seconds: float = 0.0,         # split on long pauses
) -> List[Dict]:
    """
    Groups consecutive segments into time-based chunks with optional overlap,
    minimum size, and pause-aware boundaries.
    """
    if not segments:
        return []

    chunks: List[Dict] = []
    current: List[Segment] = []
    chunk_start = segments[0].start
    chunk_end = segments[0].end
    current_chars = 0

    def meets_min_size() -> bool:
        seconds_ok = min_chunk_seconds <= 0 or (chunk_end - chunk_start) >= min_chunk_seconds
        chars_ok = min_chunk_chars <= 0 or current_chars >= min_chunk_chars
        if min_chunk_seconds > 0 and min_chunk_chars > 0:
            return seconds_ok or chars_ok
        return seconds_ok and chars_ok

    def flush_chunk(*, allow_overlap: bool = True) -> None:
        nonlocal current, chunk_start, chunk_end, current_chars
        if not current:
            return

        text = " ".join(s.text for s in current).strip()
        chunks.append({
            "start_s": round(chunk_start, 2),
            "end_s": round(chunk_end, 2),
            "text": text,
        })

        # create overlap seed for next chunk
        if allow_overlap and overlap_seconds > 0:
            overlap_start = max(chunk_end - overlap_seconds, chunk_start)
            next_current = [s for s in current if s.end > overlap_start]
        else:
            next_current = []

        current = next_current
        if current:
            chunk_start = current[0].start
            chunk_end = current[-1].end
            current_chars = sum(len(s.text) + 1 for s in current)
        else:
            chunk_start = 0.0
            chunk_end = 0.0
            current_chars = 0

    for seg in segments:
        if not current:
            current = [seg]
            chunk_start = seg.start
            chunk_end = seg.end
            current_chars = len(seg.text) + 1
            continue

        gap_seconds = seg.start - chunk_end
        if max_gap_seconds > 0 and gap_seconds >= max_gap_seconds and meets_min_size():
            flush_chunk(allow_overlap=False)
            current = [seg]
            chunk_start = seg.start
            chunk_end = seg.end
            current_chars = len(seg.text) + 1
            continue

        proposed_end = seg.end
        proposed_seconds = proposed_end - chunk_start
        proposed_chars = current_chars + len(seg.text) + 1

        # if adding this segment would exceed thresholds, flush first
        if (proposed_seconds > max_chunk_seconds or proposed_chars > max_chunk_chars) and meets_min_size():
            flush_chunk()

            # start new chunk with this segment if overlap did not include it
            if not current:
                current = [seg]
                chunk_start = seg.start
                chunk_end = seg.end
                current_chars = len(seg.text) + 1
            else:
                # overlap already seeded; add seg
                current.append(seg)
                chunk_end = seg.end
                current_chars += len(seg.text) + 1
        else:
            current.append(seg)
            chunk_end = seg.end
            current_chars = proposed_chars

    flush_chunk()
    return chunks
