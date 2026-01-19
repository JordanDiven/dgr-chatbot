from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from dgr_rag.utils.transcripts import CHUNK_PROFILES, build_chunks, parse_transcript_file

CHUNK_PROFILE = "guidance"

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

            chunk_cfg = CHUNK_PROFILES[CHUNK_PROFILE]
            chunks = build_chunks(segments, **chunk_cfg)

            for idx, ch in enumerate(chunks, start=1):
                chunk_obj = {
                    "chunk_id": f"{episode_id}_{video_id}_{idx:04d}".strip("_"),
                    "chunk_profile": CHUNK_PROFILE,
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
            print(f"- {obj['chunk_id']} [{obj['start_s']} to {obj['end_s']}s] {obj['title']}")
            print(f"  {obj['text'][:200]}...\n")

if __name__ == "__main__":
    main()
