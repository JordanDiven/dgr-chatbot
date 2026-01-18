from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def load_chunks(jsonl_path: Path):
    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks

def main():
    repo_root = Path(__file__).resolve().parents[1]
    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    if not chunks_path.exists():
        raise RuntimeError("chunks.jsonl not found. Run scripts/03_make_chunks.py first.")

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks")

    # Persistent local DB under data/index/chroma
    chroma_dir = repo_root / "data" / "index" / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="dgr_chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Upsert in batches (idempotent: if IDs already exist, Chroma will error; simplest is to wipe and rebuild during dev)
    # For early dev: wipe and rebuild
    try:
        client.delete_collection("dgr_chunks")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="dgr_chunks")

    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{
        "episode_id": c.get("episode_id", ""),
        "title": c.get("title", ""),
        "video_id": c.get("video_id", ""),
        "url": c.get("url", ""),
        "start_s": c.get("start_s", 0.0),
        "end_s": c.get("end_s", 0.0),
    } for c in chunks]

    print("Embedding…")
    embeddings = model.encode(docs, show_progress_bar=True, normalize_embeddings=True).tolist()

    print("Indexing…")
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    while True:
        q = input("\nQuery (or blank to exit): ").strip()
        if not q:
            break

        q_emb = model.encode([q], normalize_embeddings=True).tolist()[0]
        res = collection.query(query_embeddings=[q_emb], n_results=5)

        print("\nTop results:")
        for i in range(len(res["ids"][0])):
            cid = res["ids"][0][i]
            meta = res["metadatas"][0][i]
            doc = res["documents"][0][i]
            print(f"\n{i+1}) {cid}")
            print(f"   {meta.get('title','')} | {meta.get('start_s')}–{meta.get('end_s')}s")
            print(f"   {doc[:350]}...")

if __name__ == "__main__":
    main()
