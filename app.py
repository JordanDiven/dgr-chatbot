from __future__ import annotations

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
import streamlit as st
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


@st.cache_data(show_spinner=False)
def load_jsonl(path_str: str) -> List[Dict]:
    items: List[Dict] = []
    path = Path(path_str)
    if not path.exists():
        return items
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


def ensure_collection(
    client: chromadb.PersistentClient,
    name: str,
    items: List[Dict],
    id_key: str,
    embedder: SentenceTransformer,
    *,
    force_rebuild: bool,
) -> Tuple[chromadb.Collection, Dict[str, Dict]]:
    item_map = {str(item[id_key]): item for item in items}
    if not items:
        return client.get_or_create_collection(name=name), item_map

    collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    if force_rebuild or collection.count() != len(items):
        try:
            client.delete_collection(name)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
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
    return collection, item_map


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


def render_intake_form() -> Optional[Dict]:
    with st.form("intake_form"):
        st.subheader("Intake questionnaire")
        age = st.text_input("Age")
        activity = st.text_input("Activity level or sport")
        issue = st.text_input("Main issue location (e.g., front of knee)")
        symptoms = st.text_input("Symptom description (e.g., soreness, sharp pain)")
        duration = st.text_input("How long has this been going on?")
        aggravators = st.text_input("What makes it worse?")
        eases = st.text_input("What makes it better?")
        pain = st.text_input("Pain scale 0-10 (typical day)")
        goals = st.text_input("Goals or timeline (if any)")
        constraints = st.text_input("Constraints (time, equipment, access)")
        submitted = st.form_submit_button("Save intake")
    if not submitted:
        return None
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


def render_checkin_form() -> Optional[Dict]:
    with st.form("checkin_form"):
        st.subheader("Daily check-in")
        pain_today = st.text_input("Pain today 0-10")
        stiffness = st.selectbox("Stiffness today", ["none", "mild", "moderate", "high"], index=1)
        function = st.selectbox("Function today", ["worse", "same", "better"], index=1)
        aggravators = st.text_input("Any new aggravators?")
        notes = st.text_input("Anything else to note?")
        submitted = st.form_submit_button("Log check-in")
    if not submitted:
        return None
    return {
        "pain_today": pain_today,
        "stiffness": stiffness,
        "function": function,
        "aggravators": aggravators,
        "notes": notes,
        "timestamp": datetime.utcnow().isoformat(),
    }


@st.cache_resource(show_spinner=False)
def init_resources(principles_path: str, canon_path: str) -> Tuple[List[Dict], List[Dict], SentenceTransformer]:
    principles = load_jsonl(principles_path)
    canon = load_jsonl(canon_path)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return principles, canon, embedder


def ensure_state(session_path: Path) -> Dict:
    if "state" not in st.session_state:
        st.session_state["state"] = load_session(session_path)
    return st.session_state["state"]


def main() -> None:
    st.set_page_config(page_title="DGR Rehab Coach", page_icon="🦵", layout="wide")
    st.title("DGR Rehab Coach (Principles-First)")

    paths = get_paths()
    principles_path = paths.data_dir / "processed" / "principles.jsonl"
    canon_path = paths.data_dir / "processed" / "canon.jsonl"

    with st.sidebar:
        st.header("Settings")
        user_id = st.text_input("User ID", value=st.session_state.get("user_id", "default"))
        model = st.text_input("Ollama model", value=st.session_state.get("model", DEFAULT_MODEL))
        summary_every = st.number_input("Summarize every N turns", min_value=1, max_value=10, value=3, step=1)
        episode_k = st.number_input("Episode top-k", min_value=1, max_value=12, value=6, step=1)
        canon_k = st.number_input("Canon top-k", min_value=1, max_value=12, value=4, step=1)
        episode_min_sim = st.slider("Episode min similarity", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
        force_rebuild = st.checkbox("Rebuild index", value=False)

    st.session_state["user_id"] = user_id
    st.session_state["model"] = model

    if not principles_path.exists():
        st.error("principles.jsonl not found. Run scripts/05_build_principles.py first.")
        st.stop()

    principles, canon, embedder = init_resources(str(principles_path), str(canon_path))

    chroma_dir = paths.data_dir / "index" / "chroma_coach"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))

    episode_collection, episode_map = ensure_collection(
        client,
        "dgr_principles_coach",
        principles,
        "principle_id",
        embedder,
        force_rebuild=force_rebuild,
    )
    canon_collection, canon_map = ensure_collection(
        client,
        "dgr_canon_coach",
        canon,
        "canon_id",
        embedder,
        force_rebuild=force_rebuild,
    )

    sessions_dir = paths.data_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_path = sessions_dir / f"{user_id}.json"
    state = ensure_state(session_path)
    if state.get("user_id") != user_id:
        state = load_session(session_path)
        st.session_state["state"] = state

    st.subheader("Session")
    col_left, col_right = st.columns([2, 1])
    with col_right:
        with st.expander("Current summary", expanded=False):
            st.write(state.get("summary", "") or "No summary yet.")
        with st.expander("Triage snapshot", expanded=False):
            st.write(state.get("last_triage", {}))

    with col_left:
        if not state.get("intake"):
            intake = render_intake_form()
            if intake:
                state["intake"] = intake
                save_session(session_path, state)
                st.experimental_rerun()
        else:
            with st.expander("Intake", expanded=False):
                st.json(state.get("intake", {}))
            if st.button("Update intake"):
                state["intake"] = {}
                save_session(session_path, state)
                st.experimental_rerun()

        checkin = render_checkin_form()
        if checkin:
            state.setdefault("checkins", []).append(checkin)
            save_session(session_path, state)
            st.success("Check-in saved.")

    st.subheader("Coach Chat")
    for msg in state.get("history", []):
        if msg.get("role") not in {"user", "assistant"}:
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask a rehab question")
    if prompt:
        with st.spinner("Thinking..."):
            triage_data = triage(model, prompt, state.get("intake", {}), state.get("summary", ""))
            state["last_triage"] = triage_data
            tags = triage_data.get("tags", [])
            query_parts = [prompt, triage_data.get("query_expansion", "")]
            query_text = " ".join(part for part in query_parts if part).strip() or prompt

            q_emb = embedder.encode([query_text], normalize_embeddings=True).tolist()[0]
            ep_res = episode_collection.query(query_embeddings=[q_emb], n_results=int(episode_k))
            ep_items = filter_by_tags(collect_results(ep_res, episode_map), tags)

            canon_items: List[RetrievalItem] = []
            if canon:
                canon_res = canon_collection.query(query_embeddings=[q_emb], n_results=int(canon_k))
                canon_items = filter_by_tags(collect_results(canon_res, canon_map), tags)

            episode_selected = [item for item in ep_items if item.similarity >= float(episode_min_sim)]
            coverage_note = ""
            if not episode_selected:
                episode_selected = ep_items[:2] if ep_items else []
                coverage_note = (
                    "Episode-specific principles were weak or missing for this issue. "
                    "Using canon principles for general guidance."
                )

            prog_signal = progression_signal(state.get("checkins", []))
            user_prompt = (
                f"User question: {prompt}\n\n"
                f"Intake: {json.dumps(state.get('intake', {}), ensure_ascii=True)}\n\n"
                f"Latest check-in: {json.dumps(state.get('checkins', [])[-1:], ensure_ascii=True)}\n\n"
                f"Progression signal: {prog_signal}\n\n"
                f"Triage: {json.dumps(triage_data, ensure_ascii=True)}\n\n"
                + format_context(episode_selected, "Episode principles", max_items=int(episode_k))
                + format_context(canon_items, "Canon principles", max_items=int(canon_k))
            )
            if coverage_note:
                user_prompt += f"\nNote: {coverage_note}\n"

            response = call_ollama_chat(model, SYSTEM_PROMPT, user_prompt)

        state["history"].append({"role": "user", "content": prompt, "ts": time.time()})
        state["history"].append({"role": "assistant", "content": response, "ts": time.time()})
        state["turn_count"] = int(state.get("turn_count", 0)) + 1

        if state["turn_count"] % int(summary_every) == 0:
            state["summary"] = update_summary(model, state.get("summary", ""), state.get("intake", {}), state["history"])

        save_session(session_path, state)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
