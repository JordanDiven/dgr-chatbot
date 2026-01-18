from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

# Matches: [12.34 --> 15.67] some text
TS_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)\s*-->\s*(?P<end>\d+(?:\.\d+)?)\]\s*(?P<text>.*)$"
)

@dataclass
class Segment:
    start: float
    end: float
    text: str

def parse_transcript_file(path: Path) -> Tuple[Dict[str, str], List[Segment]]:
    """
    Parses your saved transcript format:
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
) -> List[Dict]:
    """
    Groups consecutive segments into time-based chunks with overlap.
    """
    if not segments:
        return []

    chunks: List[Dict] = []
    current: List[Segment] = []
    chunk_start = segments[0].start
    chunk_end = segments[0].end
    current_chars = 0

    def flush_chunk():
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
        if overlap_seconds > 0:
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

        proposed_end = seg.end
        proposed_seconds = proposed_end - chunk_start
        proposed_chars = current_chars + len(seg.text) + 1

        # if adding this segment would exceed thresholds, flush first
        if proposed_seconds > max_chunk_seconds or proposed_chars > max_chunk_chars:
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

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    in_dir = repo_root / "data" / "raw" / "transcripts"
    out_path = repo_root / "data" / "processed" / "chunks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(f"No transcript .txt files found in: {in_dir}")

    total_chunks = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for fp in tqdm(files, desc="Chunking transcripts"):
            meta, segments = parse_transcript_file(fp)

            # sensible fallbacks if metadata is missing
            episode_id = meta.get("episode_id", fp.stem.split("_")[0])
            title = meta.get("title", fp.stem)
            video_id = meta.get("video_id", "")
            url = meta.get("url", "")

            chunks = build_chunks(
                segments,
                max_chunk_seconds=240.0,
                max_chunk_chars=3500,
                overlap_seconds=30.0,
            )

            for idx, ch in enumerate(chunks, start=1):
                chunk_obj = {
                    "chunk_id": f"{episode_id}_{video_id}_{idx:04d}".strip("_"),
                    "episode_id": episode_id,
                    "title": title,
                    "video_id": video_id,
                    "url": url,
                    "start_s": ch["start_s"],
                    "end_s": ch["end_s"],
                    "text": ch["text"],
                }
                f_out.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)

    print(f"Wrote {total_chunks} chunks -> {out_path}")

    # Print a few samples so you can sanity-check chunk boundaries
    print("\nSample chunks (first 3):")
    with out_path.open("r", encoding="utf-8") as f_in:
        for _ in range(3):
            line = f_in.readline().strip()
            if not line:
                break
            obj = json.loads(line)
            print(f"- {obj['chunk_id']} [{obj['start_s']}–{obj['end_s']}s] {obj['title']}")
            print(f"  {obj['text'][:200]}...\n")

if __name__ == "__main__":
    main()
