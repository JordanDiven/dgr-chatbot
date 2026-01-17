from __future__ import annotations

from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi

from dgr_rag.config import get_paths
from dgr_rag.ingest.episode_index import read_episode_index
from dgr_rag.ingest.youtube_transcripts import download_transcript

def main() -> None:
    paths = get_paths()
    episodes = read_episode_index(paths.episode_index_csv)

    api = YouTubeTranscriptApi()

    ok = 0
    fail = 0

for ep in tqdm(episodes, desc="Downloading transcripts"):
    result = download_transcript(
        api,
        episode_id=ep.episode_id,
        title=ep.title,
        video_id=ep.youtube_video_id,
        url=ep.url,
        out_dir=paths.raw_transcripts_dir,
        sleep_s=0.6,
        overwrite=False,
    )

    if not result.ok:
        print(result.message)   # <-- add this line

    ok += int(result.ok)
    fail += int(not result.ok)

    print(f"Done. OK={ok}, FAIL={fail}, TOTAL={len(episodes)}")

if __name__ == "__main__":
    main()