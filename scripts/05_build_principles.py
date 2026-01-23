from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from dgr_rag.config import get_paths
from dgr_rag.prompts import build_canon_prompt, build_principles_prompt
from dgr_rag.utils.transcripts import build_chunks, parse_transcript_file

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"

PRINCIPLES_CHUNK_PROFILE = {
    "max_chunk_seconds": 360.0,
    "max_chunk_chars": 1400,
    "overlap_seconds": 0.0,
    "min_chunk_seconds": 60.0,
    "min_chunk_chars": 350,
    "max_gap_seconds": 12.0,
}


def call_ollama(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.2,
    num_predict: int = 400,
    timeout_seconds: float = 120.0,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama request failed. Is the server running on localhost:11434?") from exc

    obj = json.loads(body)
    return obj.get("response", "").strip()


def extract_json_array(raw: str) -> List[Dict]:
    if not raw:
        return []

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start:end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start:end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                if isinstance(data.get("principles"), list):
                    return data["principles"]
                return [data]
        except json.JSONDecodeError:
            pass

    return []


def load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def normalize_key(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace()).strip()


def build_principles_for_episode(
    model: str,
    episode_meta: Dict[str, str],
    chunks: List[Dict],
    *,
    max_items_per_chunk: int,
    sleep_seconds: float,
    timeout_seconds: float,
    retries: int,
    retry_backoff: float,
    skip_on_error: bool,
) -> List[Dict]:
    seen = set()
    items: List[Dict] = []
    episode_id = episode_meta.get("episode_id", "")
    for ch in chunks:
        prompt = build_principles_prompt(ch["text"], max_items=max_items_per_chunk)
        raw = ""
        for attempt in range(retries + 1):
            try:
                raw = call_ollama(model, prompt, timeout_seconds=timeout_seconds)
                break
            except Exception as exc:
                if attempt >= retries:
                    msg = f"Ollama failed for episode {episode_id} chunk {ch['start_s']} to {ch['end_s']}s: {exc}"
                    if skip_on_error:
                        print(msg)
                        raw = ""
                        break
                    raise RuntimeError(msg) from exc
                delay = retry_backoff * (attempt + 1)
                print(
                    f"Ollama error for episode {episode_id} chunk {ch['start_s']} to {ch['end_s']}s; "
                    f"retrying in {delay:.1f}s ({attempt + 1}/{retries})"
                )
                time.sleep(delay)

        extracted = extract_json_array(raw)
        for obj in extracted:
            principle = str(obj.get("principle", "")).strip()
            if not principle:
                continue
            key = normalize_key(principle)
            if key in seen:
                continue
            seen.add(key)

            rationale = str(obj.get("rationale", "")).strip()
            tags = obj.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).strip().lower() for t in tags if str(t).strip()]

            items.append({
                "episode_id": episode_meta.get("episode_id", ""),
                "title": episode_meta.get("title", ""),
                "video_id": episode_meta.get("video_id", ""),
                "url": episode_meta.get("url", ""),
                "chunk_start_s": ch["start_s"],
                "chunk_end_s": ch["end_s"],
                "principle": principle,
                "rationale": rationale,
                "tags": tags,
            })

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return items


def build_principles(
    model: str,
    transcripts_dir: Path,
    out_path: Path,
    *,
    max_items_per_chunk: int,
    max_chunks: int | None,
    sleep_seconds: float,
    timeout_seconds: float,
    retries: int,
    retry_backoff: float,
    resume: bool,
    skip_on_error: bool,
    only_episode: str,
) -> List[Dict]:
    files = sorted(transcripts_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(f"No transcript .txt files found in: {transcripts_dir}")

    all_items: List[Dict] = []
    total_written = 0
    seen_episode_ids = set()
    write_mode = "w"
    if resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episode_id = str(obj.get("episode_id", "")).strip()
                if episode_id:
                    seen_episode_ids.add(episode_id)
        write_mode = "a"
        print(f"Resuming: skipping {len(seen_episode_ids)} episodes already in output.")

    with out_path.open(write_mode, encoding="utf-8") as f_out:
        for fp in tqdm(files, desc="Extracting principles"):
            meta, segments = parse_transcript_file(fp)

            episode_meta = {
                "episode_id": meta.get("episode_id", fp.stem.split("_")[0]),
                "title": meta.get("title", fp.stem),
                "video_id": meta.get("video_id", ""),
                "url": meta.get("url", ""),
            }
            episode_id = episode_meta["episode_id"]
            if only_episode and episode_id != only_episode:
                continue
            if resume and episode_id in seen_episode_ids and not only_episode:
                continue

            chunks = build_chunks(segments, **PRINCIPLES_CHUNK_PROFILE)
            if max_chunks is not None:
                chunks = chunks[:max_chunks]

            items = build_principles_for_episode(
                model,
                episode_meta,
                chunks,
                max_items_per_chunk=max_items_per_chunk,
                sleep_seconds=sleep_seconds,
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_backoff=retry_backoff,
                skip_on_error=skip_on_error,
            )

            episode_counter = 0
            for item in items:
                episode_counter += 1
                principle_id = f"{episode_meta['episode_id']}_{episode_meta['video_id']}_{episode_counter:04d}".strip("_")
                text = f"Principle: {item['principle']}"
                if item["rationale"]:
                    text += f" Rationale: {item['rationale']}"
                if item["tags"]:
                    text += f" Tags: {', '.join(item['tags'])}"

                record = {
                    "principle_id": principle_id,
                    "episode_id": item["episode_id"],
                    "title": item["title"],
                    "video_id": item["video_id"],
                    "url": item["url"],
                    "chunk_start_s": item["chunk_start_s"],
                    "chunk_end_s": item["chunk_end_s"],
                    "principle": item["principle"],
                    "rationale": item["rationale"],
                    "tags": item["tags"],
                    "text": text,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_items.append(record)
                total_written += 1

    print(f"Wrote {total_written} principles -> {out_path}")
    return all_items


def synthesize_canon(
    model: str,
    principles: List[str],
    *,
    target_count: int,
    timeout_seconds: float,
    retries: int,
    retry_backoff: float,
    skip_on_error: bool,
) -> List[Dict]:
    prompt = build_canon_prompt(principles, target_count=target_count)
    raw = ""
    for attempt in range(retries + 1):
        try:
            raw = call_ollama(model, prompt, num_predict=800, timeout_seconds=timeout_seconds)
            break
        except Exception as exc:
            if attempt >= retries:
                if skip_on_error:
                    print(f"Ollama failed during canon synthesis: {exc}")
                    raw = ""
                    break
                raise
            delay = retry_backoff * (attempt + 1)
            print(f"Ollama error during canon synthesis; retrying in {delay:.1f}s ({attempt + 1}/{retries})")
            time.sleep(delay)
    return extract_json_array(raw)


def build_canon(
    model: str,
    principles: List[Dict],
    canon_path: Path,
    *,
    target_count: int,
    batch_size: int,
    timeout_seconds: float,
    retries: int,
    retry_backoff: float,
    skip_on_error: bool,
) -> None:
    principle_texts = [p["principle"] for p in principles if p.get("principle")]
    if not principle_texts:
        print("No principles found; skipping canon build.")
        return

    candidates: List[Dict] = []
    for i in range(0, len(principle_texts), batch_size):
        batch = principle_texts[i:i + batch_size]
        extracted = synthesize_canon(
            model,
            batch,
            target_count=target_count,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_backoff=retry_backoff,
            skip_on_error=skip_on_error,
        )
        candidates.extend(extracted)

    if len(candidates) > target_count:
        candidate_texts = [c.get("principle", "") for c in candidates if c.get("principle")]
        candidates = synthesize_canon(
            model,
            candidate_texts,
            target_count=target_count,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_backoff=retry_backoff,
            skip_on_error=skip_on_error,
        )

    seen = set()
    canon_items: List[Dict] = []
    for obj in candidates:
        principle = str(obj.get("principle", "")).strip()
        if not principle:
            continue
        key = normalize_key(principle)
        if key in seen:
            continue
        seen.add(key)
        rationale = str(obj.get("rationale", "")).strip()
        tags = obj.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).strip().lower() for t in tags if str(t).strip()]
        canon_items.append({
            "principle": principle,
            "rationale": rationale,
            "tags": tags,
        })

    with canon_path.open("w", encoding="utf-8") as f_out:
        for idx, item in enumerate(canon_items, start=1):
            canon_id = f"canon_{idx:04d}"
            text = f"Principle: {item['principle']}"
            if item["rationale"]:
                text += f" Rationale: {item['rationale']}"
            if item["tags"]:
                text += f" Tags: {', '.join(item['tags'])}"
            record = {
                "canon_id": canon_id,
                "principle": item["principle"],
                "rationale": item["rationale"],
                "tags": item["tags"],
                "text": text,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote canon principles -> {canon_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-episode rehab principles with Ollama.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--max-items-per-chunk", type=int, default=4)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-on-error", action="store_true")
    parser.add_argument("--only-episode", default="")
    parser.add_argument("--canon-target", type=int, default=25)
    parser.add_argument("--canon-batch-size", type=int, default=60)
    parser.add_argument("--no-canon", action="store_true")
    parser.add_argument("--canon-only", action="store_true")
    args = parser.parse_args()

    paths = get_paths()
    transcripts_dir = paths.raw_transcripts_dir
    out_path = paths.data_dir / "processed" / "principles.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.canon_only and args.no_canon:
        raise RuntimeError("Cannot use --canon-only with --no-canon.")

    if args.canon_only:
        if not out_path.exists():
            raise RuntimeError(f"principles.jsonl not found at {out_path}")
        principles = load_jsonl(out_path)
    else:
        principles = build_principles(
            args.model,
            transcripts_dir,
            out_path,
            max_items_per_chunk=args.max_items_per_chunk,
            max_chunks=args.max_chunks,
            sleep_seconds=args.sleep_seconds,
            timeout_seconds=args.timeout,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            resume=args.resume,
            skip_on_error=args.skip_on_error,
            only_episode=args.only_episode.strip(),
        )

    if not args.no_canon:
        canon_path = paths.data_dir / "processed" / "canon.jsonl"
        build_canon(
            args.model,
            principles,
            canon_path,
            target_count=args.canon_target,
            batch_size=args.canon_batch_size,
            timeout_seconds=args.timeout,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            skip_on_error=args.skip_on_error,
        )


if __name__ == "__main__":
    main()
