from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from dgr_rag.config import get_paths

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = (
    "You are a rehab coaching assistant grounded in David Grey Rehab principles.\n"
    "Use only the provided principles as your knowledge source.\n"
    "Do not diagnose. Do not quote transcripts.\n"
    "If the question is not about physio rehab, redirect and ask the user to reframe.\n"
    "If episode-specific principles are weak or missing, say so and use the canon principles.\n"
    "Avoid medical advice for serious conditions; include a brief safety note only when needed.\n"
    "Output format:\n"
    "1) Short answer\n"
    "2) 7-day plan (day-by-day)\n"
    "3) Progression rules\n"
    "4) Check-in questions\n"
    "5) Notes (include safety note only if needed)\n"
)

TRIAGE_PROMPT = (
    "You are doing non-diagnostic triage for a rehab coach.\n"
    "Return JSON only with keys:\n"
    "body_region, likely_structure, symptom, irritability, phase, needs_medical, red_flags, tags, query_expansion.\n"
    "Rules:\n"
    "- Do not diagnose.\n"
    "- likely_structure is a hypothesis only.\n"
    "- irritability: low, medium, high, or unknown.\n"
    "- phase: acute, subacute, chronic, or unknown.\n"
    "- needs_medical is true if red flags or serious medical issues are present.\n"
    "- tags should be 3-8 lower-case keywords.\n"
)

SUMMARY_PROMPT = (
    "Summarize the rehab state for continuity.\n"
    "Focus on symptoms, current plan, progress, constraints, and next steps.\n"
    "Return a short paragraph. Avoid diagnosis and quotes.\n"
)


@dataclass
class RetrievalItem:
    item_id: str
    text: str
    tags: List[str]
    similarity: float
    meta: Dict[str, str]


def call_ollama_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    num_predict: int = 700,
    timeout_seconds: float = 120.0,
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
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama request failed. Is the server running on localhost:11434?") from exc

    obj = json.loads(body)
    message = obj.get("message", {})
    return str(message.get("content", "")).strip()


def extract_json_object(raw: str) -> Dict:
    if not raw:
        return {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = raw[start:end + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def simple_tag_extract(text: str) -> List[str]:
    text = text.lower()
    keywords = [
        "knee", "patellar", "patellofemoral", "achilles", "calf", "heel", "ankle",
        "hip", "groin", "hamstring", "quad", "back", "neck", "shoulder", "elbow",
        "wrist", "foot", "plantar", "tendon", "tendinopathy", "strain", "sprain",
    ]
    tags = [kw for kw in keywords if kw in text]
    return sorted(set(tags))


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
) -> Dict[str, Dict]:
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

    item_map = {str(item[id_key]): item for item in items}
    return item_map


def collect_results(res: Dict, item_map: Dict[str, Dict]) -> List[RetrievalItem]:
    results: List[RetrievalItem] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for idx in range(len(ids)):
        dist = dists[idx] if idx < len(dists) else 1.0
        similarity = 1.0 - float(dist)
        item_id = str(ids[idx])
        base = item_map.get(item_id, {})
        tags = base.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        results.append(
            RetrievalItem(
                item_id=item_id,
                text=docs[idx],
                tags=tags,
                similarity=similarity,
                meta=metas[idx],
            )
        )
    return results


def filter_by_tags(items: List[RetrievalItem], desired_tags: List[str]) -> List[RetrievalItem]:
    if not desired_tags:
        return items
    tag_set = set(desired_tags)
    filtered = [item for item in items if set(item.tags) & tag_set]
    return filtered if filtered else items


def format_context(items: List[RetrievalItem], label: str, max_items: int) -> str:
    if not items:
        return f"{label}:\n- (none)\n"
    lines = [f"{label}:"]
    for item in items[:max_items]:
        meta = item.meta
        title = meta.get("title", "").strip()
        episode_id = meta.get("episode_id", "").strip()
        prefix = f"(episode {episode_id}) " if episode_id else ""
        if title:
            prefix += f"{title}: "
        lines.append(f"- {prefix}{item.text}")
    return "\n".join(lines) + "\n"


def run_intake() -> Dict:
    print("\nIntake questionnaire (short form):")
    age = input("Age: ").strip()
    activity = input("Activity level or sport: ").strip()
    issue = input("Main issue location (e.g., front of knee): ").strip()
    symptoms = input("Symptom description (e.g., soreness, sharp pain): ").strip()
    duration = input("How long has this been going on? ").strip()
    aggravators = input("What makes it worse? ").strip()
    eases = input("What makes it better? ").strip()
    pain = input("Pain scale 0-10 (typical day): ").strip()
    goals = input("Goals or timeline (if any): ").strip()
    constraints = input("Constraints (time, equipment, access): ").strip()

    return {
        "age": age,
        "activity": activity,
        "issue_location": issue,
        "symptoms": symptoms,
        "duration": duration,
        "aggravators": aggravators,
        "eases": eases,
        "pain_scale": pain,
        "goals": goals,
        "constraints": constraints,
        "updated_at": datetime.utcnow().isoformat(),
    }


def run_checkin() -> Dict:
    print("\nDaily check-in:")
    pain_today = input("Pain today 0-10: ").strip()
    stiffness = input("Stiffness today (none/mild/moderate/high): ").strip()
    function = input("Function today (worse/same/better): ").strip()
    aggravators = input("Any new aggravators? ").strip()
    notes = input("Anything else to note? ").strip()

    return {
        "pain_today": pain_today,
        "stiffness": stiffness,
        "function": function,
        "aggravators": aggravators,
        "notes": notes,
        "timestamp": datetime.utcnow().isoformat(),
    }


def progression_signal(checkins: List[Dict]) -> str:
    if not checkins:
        return "maintain"
    latest = checkins[-1]
    try:
        pain = float(latest.get("pain_today", ""))
    except ValueError:
        pain = None
    function = (latest.get("function", "") or "").lower()

    if pain is not None and pain >= 6:
        return "deload"
    if function == "better" and pain is not None and pain <= 3:
        return "progress"
    if function == "worse":
        return "deload"
    return "maintain"


def triage(
    model: str,
    user_text: str,
    intake: Dict,
    summary: str,
) -> Dict:
    prompt = (
        f"User text: {user_text}\n\n"
        f"Intake summary: {json.dumps(intake, ensure_ascii=True)}\n\n"
        f"Session summary: {summary}\n"
    )
    raw = call_ollama_chat(model, TRIAGE_PROMPT, prompt, num_predict=300)
    data = extract_json_object(raw)
    if not data:
        tags = simple_tag_extract(user_text)
        data = {
            "body_region": "",
            "likely_structure": "",
            "symptom": "",
            "irritability": "unknown",
            "phase": "unknown",
            "needs_medical": False,
            "red_flags": [],
            "tags": tags,
            "query_expansion": " ".join(tags),
        }
    if not isinstance(data.get("tags"), list):
        data["tags"] = simple_tag_extract(user_text)
    return data


def update_summary(model: str, summary: str, intake: Dict, history: List[Dict]) -> str:
    recent = history[-8:]
    prompt = (
        f"Previous summary: {summary}\n\n"
        f"Intake: {json.dumps(intake, ensure_ascii=True)}\n\n"
        f"Recent conversation:\n{json.dumps(recent, ensure_ascii=True)}\n"
    )
    return call_ollama_chat(model, SUMMARY_PROMPT, prompt, num_predict=220)


def load_session(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "user_id": path.stem,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "intake": {},
        "summary": "",
        "history": [],
        "checkins": [],
        "turn_count": 0,
    }


def save_session(path: Path, state: Dict) -> None:
    state["updated_at"] = datetime.utcnow().isoformat()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="DGR rehab coach CLI (principles-first).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--user-id", default="default")
    parser.add_argument("--episode-k", type=int, default=6)
    parser.add_argument("--canon-k", type=int, default=4)
    parser.add_argument("--episode-min-sim", type=float, default=0.35)
    parser.add_argument("--summary-every", type=int, default=3)
    args = parser.parse_args()

    paths = get_paths()
    principles_path = paths.data_dir / "processed" / "principles.jsonl"
    canon_path = paths.data_dir / "processed" / "canon.jsonl"
    if not principles_path.exists():
        raise RuntimeError("principles.jsonl not found. Run scripts/05_build_principles.py first.")

    principles = load_jsonl(principles_path)
    canon = load_jsonl(canon_path) if canon_path.exists() else []

    chroma_dir = paths.data_dir / "index" / "chroma_coach"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    episode_collection = prepare_collection(client, "dgr_principles_coach")
    episode_map = index_items(episode_collection, principles, "principle_id", embedder)

    canon_collection = prepare_collection(client, "dgr_canon_coach")
    canon_map = index_items(canon_collection, canon, "canon_id", embedder) if canon else {}

    sessions_dir = paths.data_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_path = sessions_dir / f"{args.user_id}.json"
    state = load_session(session_path)

    if not state.get("intake"):
        state["intake"] = run_intake()
        save_session(session_path, state)

    print("\nType ':checkin' for daily check-in, ':intake' to redo intake, ':summary' to view summary, ':quit' to exit.")

    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {":quit", ":exit"}:
            break
        if user_text.lower() == ":intake":
            state["intake"] = run_intake()
            save_session(session_path, state)
            continue
        if user_text.lower() == ":checkin":
            checkin = run_checkin()
            state.setdefault("checkins", []).append(checkin)
            save_session(session_path, state)
            continue
        if user_text.lower() == ":summary":
            print(f"\nSummary:\n{state.get('summary', '')}\n")
            continue

        triage_data = triage(args.model, user_text, state.get("intake", {}), state.get("summary", ""))
        tags = triage_data.get("tags", [])
        query_parts = [user_text, triage_data.get("query_expansion", "")]
        query_text = " ".join(part for part in query_parts if part).strip()
        if not query_text:
            query_text = user_text

        q_emb = embedder.encode([query_text], normalize_embeddings=True).tolist()[0]
        ep_res = episode_collection.query(query_embeddings=[q_emb], n_results=args.episode_k)
        ep_items = collect_results(ep_res, episode_map)
        ep_items = filter_by_tags(ep_items, tags)

        canon_items: List[RetrievalItem] = []
        if canon:
            canon_res = canon_collection.query(query_embeddings=[q_emb], n_results=args.canon_k)
            canon_items = collect_results(canon_res, canon_map)
            canon_items = filter_by_tags(canon_items, tags)

        episode_selected = [item for item in ep_items if item.similarity >= args.episode_min_sim]
        coverage_note = ""
        if not episode_selected:
            episode_selected = ep_items[:2] if ep_items else []
            coverage_note = (
                "Episode-specific principles were weak or missing for this issue. "
                "Using canon principles for general guidance."
            )

        prog_signal = progression_signal(state.get("checkins", []))
        prompt = (
            f"User question: {user_text}\n\n"
            f"Intake: {json.dumps(state.get('intake', {}), ensure_ascii=True)}\n\n"
            f"Latest check-in: {json.dumps(state.get('checkins', [])[-1:], ensure_ascii=True)}\n\n"
            f"Progression signal: {prog_signal}\n\n"
            f"Triage: {json.dumps(triage_data, ensure_ascii=True)}\n\n"
            + format_context(episode_selected, "Episode principles", max_items=args.episode_k)
            + format_context(canon_items, "Canon principles", max_items=args.canon_k)
        )
        if coverage_note:
            prompt += f"\nNote: {coverage_note}\n"

        response = call_ollama_chat(args.model, SYSTEM_PROMPT, prompt)
        print(f"\nCoach:\n{response}\n")

        state["history"].append({"role": "user", "content": user_text, "ts": time.time()})
        state["history"].append({"role": "assistant", "content": response, "ts": time.time()})
        state["turn_count"] = int(state.get("turn_count", 0)) + 1

        if state["turn_count"] % max(args.summary_every, 1) == 0:
            state["summary"] = update_summary(args.model, state.get("summary", ""), state.get("intake", {}), state["history"])

        save_session(session_path, state)


if __name__ == "__main__":
    main()
