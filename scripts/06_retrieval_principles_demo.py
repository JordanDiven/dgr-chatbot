from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from dgr_rag.config import get_paths

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = (
    "You are a rehab coaching assistant grounded in David Grey Rehab principles.\n"
    "Use only the provided principles as your knowledge source.\n"
    "Do not quote transcripts.\n"
    "If the question is not about physio rehab, redirect and ask the user to reframe.\n"
    "If episode-specific principles are weak or missing, say so and use the canon principles.\n"
    "Avoid diagnosis. Provide general rehab guidance and encourage professional evaluation.\n"
    "Format:\n"
    "1) Short answer\n"
    "2) Principles applied\n"
    "3) Plan\n"
    "4) Watch-outs\n"
)


def call_ollama_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    num_predict: int = 600,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_CHAT_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama request failed. Is the server running on localhost:11434?") from exc

    obj = json.loads(body)
    message = obj.get("message", {})
    return str(message.get("content", "")).strip()


def load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_doc(item: Dict) -> str:
    if item.get("text"):
        return item["text"]
    text = f"Principle: {item.get('principle', '')}"
    if item.get("rationale"):
        text += f" Rationale: {item['rationale']}"
    if item.get("tags"):
        text += f" Tags: {', '.join(item['tags'])}"
    return text.strip()


def prepare_collection(client: chromadb.PersistentClient, name: str) -> chromadb.Collection:
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def index_items(
    collection: chromadb.Collection,
    items: List[Dict],
    id_key: str,
    embedder: SentenceTransformer,
) -> None:
    ids = [str(item[id_key]) for item in items]
    docs = [build_doc(item) for item in items]
    metas = []
    for item in items:
        metas.append({
            "episode_id": item.get("episode_id", ""),
            "title": item.get("title", ""),
            "video_id": item.get("video_id", ""),
            "url": item.get("url", ""),
            "chunk_start_s": item.get("chunk_start_s", 0.0),
            "chunk_end_s": item.get("chunk_end_s", 0.0),
        })

    embeddings = embedder.encode(docs, show_progress_bar=True, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)


def format_context(items: List[Dict], label: str) -> str:
    if not items:
        return f"{label}:\n- (none)\n"
    lines = [f"{label}:"]
    for item in items:
        meta = item["meta"]
        title = meta.get("title", "").strip()
        episode_id = meta.get("episode_id", "").strip()
        prefix = f"(episode {episode_id}) " if episode_id else ""
        if title:
            prefix += f"{title}: "
        lines.append(f"- {prefix}{item['text']}")
    return "\n".join(lines) + "\n"


def to_similarity(distance: float) -> float:
    return 1.0 - float(distance)


def collect_results(res: Dict) -> List[Dict]:
    results: List[Dict] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for idx in range(len(ids)):
        dist = dists[idx] if idx < len(dists) else 1.0
        results.append({
            "id": ids[idx],
            "text": docs[idx],
            "meta": metas[idx],
            "distance": dist,
            "similarity": to_similarity(dist),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Principles-only RAG demo (with canon fallback).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--episode-k", type=int, default=6)
    parser.add_argument("--canon-k", type=int, default=4)
    parser.add_argument("--episode-min-sim", type=float, default=0.30)
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    paths = get_paths()
    principles_path = paths.data_dir / "processed" / "principles.jsonl"
    canon_path = paths.data_dir / "processed" / "canon.jsonl"
    if not principles_path.exists():
        raise RuntimeError("principles.jsonl not found. Run scripts/05_build_principles.py first.")

    principles = load_jsonl(principles_path)
    canon = load_jsonl(canon_path) if canon_path.exists() else []

    chroma_dir = paths.data_dir / "index" / "chroma_principles"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    episode_collection = prepare_collection(client, "dgr_principles")
    index_items(episode_collection, principles, "principle_id", embedder)

    canon_collection = prepare_collection(client, "dgr_canon")
    if canon:
        index_items(canon_collection, canon, "canon_id", embedder)

    while True:
        q = input("\nQuery (or blank to exit): ").strip()
        if not q:
            break

        q_emb = embedder.encode([q], normalize_embeddings=True).tolist()[0]
        ep_res = episode_collection.query(query_embeddings=[q_emb], n_results=args.episode_k)
        ep_items = collect_results(ep_res)

        canon_items = []
        if canon:
            canon_res = canon_collection.query(query_embeddings=[q_emb], n_results=args.canon_k)
            canon_items = collect_results(canon_res)

        episode_selected = [item for item in ep_items if item["similarity"] >= args.episode_min_sim]
        coverage_note = ""
        if not episode_selected:
            episode_selected = ep_items[:2] if ep_items else []
            coverage_note = (
                "No episode-specific principles matched with high confidence. "
                "Use canon principles for general guidance."
            )

        if args.show_context:
            print("\nRetrieved context:")
            print(format_context(episode_selected, "Episode principles"))
            print(format_context(canon_items, "Canon principles"))

        user_prompt = (
            f"User question: {q}\n\n"
            + format_context(episode_selected, "Episode principles")
            + format_context(canon_items, "Canon principles")
        )
        if coverage_note:
            user_prompt += f"\nNote: {coverage_note}\n"

        response = call_ollama_chat(args.model, SYSTEM_PROMPT, user_prompt)
        print("\nAnswer:\n")
        print(response)


if __name__ == "__main__":
    main()
