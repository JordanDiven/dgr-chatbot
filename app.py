from __future__ import annotations

import json
import time
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
import streamlit as st
from sentence_transformers import SentenceTransformer

from dgr_rag.config import get_paths
from dgr_rag.llm import call_chat
from dgr_rag.prompts import (
    ADHOC_PROMPT,
    CHECKIN_PROMPT,
    FOLLOWUP_PROMPT,
    PRESCRIPTION_PROMPT,
    SUMMARY_PROMPT,
    TRIAGE_PROMPT,
)

DEFAULT_PRIMARY_PROVIDER = "openai"
DEFAULT_PRIMARY_MODEL = "gpt-4o-mini"
DEFAULT_DAILY_PROVIDER = "ollama"
DEFAULT_DAILY_MODEL = "phi3:mini"

 


@dataclass
class RetrievalItem:
    item_id: str
    text: str
    tags: List[str]
    similarity: float
    meta: Dict[str, str]


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


def normalize_query_part(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


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


def init_chroma_client(chroma_dir: Path, *, force_rebuild: bool) -> chromadb.PersistentClient:
    try:
        return chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
    except ValueError as exc:
        if "default_tenant" in str(exc):
            if force_rebuild:
                shutil.rmtree(chroma_dir, ignore_errors=True)
                chroma_dir.mkdir(parents=True, exist_ok=True)
                return chromadb.PersistentClient(
                    path=str(chroma_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
            raise RuntimeError(
                "Chroma index is incompatible. Enable 'Rebuild index' or delete data/index/chroma_coach."
            ) from exc
        raise


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


def latest_checkin_date(checkins: List[Dict]) -> Optional[datetime.date]:
    if not checkins:
        return None
    latest = checkins[-1]
    ts = latest.get("timestamp", "")
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts).date()
    except ValueError:
        return None


def checkin_done_today(checkins: List[Dict]) -> bool:
    last_date = latest_checkin_date(checkins)
    if not last_date:
        return False
    return last_date == datetime.now().date()


def triage(
    provider: str,
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
    raw = call_chat(provider, model, TRIAGE_PROMPT, prompt, num_predict=300)
    data = extract_json_object(raw)
    if not data:
        tags = simple_tag_extract(user_text)
        data = {
            "body_region": "",
            "likely_structure": "",
            "symptom": "",
            "irritability": "unknown",
            "phase": "unknown",
            "diagnosis": "",
            "diagnosis_confidence": "low",
            "rehabable": True,
            "needs_medical": False,
            "red_flags": [],
            "tags": tags,
            "query_expansion": " ".join(tags),
        }
    if not isinstance(data.get("tags"), list):
        data["tags"] = simple_tag_extract(user_text)
    if not isinstance(data.get("diagnosis_confidence"), str):
        data["diagnosis_confidence"] = "low"
    if not isinstance(data.get("rehabable"), bool):
        data["rehabable"] = True
    if not isinstance(data.get("needs_medical"), bool):
        data["needs_medical"] = False
    return data


def update_summary(provider: str, model: str, summary: str, intake: Dict, history: List[Dict]) -> str:
    recent = history[-8:]
    prompt = (
        f"Previous summary: {summary}\n\n"
        f"Intake: {json.dumps(intake, ensure_ascii=True)}\n\n"
        f"Recent conversation:\n{json.dumps(recent, ensure_ascii=True)}\n"
    )
    return call_chat(provider, model, SUMMARY_PROMPT, prompt, num_predict=220)


def generate_followup_questions(provider: str, model: str, intake: Dict) -> List[str]:
    prompt = f"Intake: {json.dumps(intake, ensure_ascii=True)}\n"
    try:
        raw = call_chat(provider, model, FOLLOWUP_PROMPT, prompt, num_predict=160)
    except Exception:
        raw = ""
    questions = [line.strip(" -\t") for line in raw.splitlines() if line.strip()]
    questions = [q for q in questions if q.endswith("?") or len(q.split()) >= 3]
    if questions:
        return questions[:5]
    return [
        "When does it hurt most (activity, range, or time of day)?",
        "What movements or loads make it worse right now?",
        "What makes it feel better or more tolerable?",
        "Do 10 single-leg calf raises on the sore side and rate pain 0-10.",
        "What activities do you need to return to in the next 4-8 weeks?",
    ]


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
        "followup_questions": [],
        "followup_answers": {},
        "followup_complete": False,
        "plan_generated": False,
        "phase": "intake",
        "pending_prompt": "",
        "pending_mode": "",
        "turn_count": 0,
    }


def save_session(path: Path, state: Dict) -> None:
    state["updated_at"] = datetime.utcnow().isoformat()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


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
    state = st.session_state["state"]
    if not state.get("phase"):
        if state.get("followup_complete") and state.get("plan_generated"):
            state["phase"] = "checkin"
        elif state.get("followup_complete"):
            state["phase"] = "prescription"
        else:
            state["phase"] = "intake"
    return state


def main() -> None:
    st.set_page_config(page_title="DGR Rehab Coach", page_icon="DGR", layout="wide")
    st.title("DGR Rehab Coach (Principles-First)")

    paths = get_paths()
    principles_path = paths.data_dir / "processed" / "principles.jsonl"
    canon_path = paths.data_dir / "processed" / "canon.jsonl"

    with st.sidebar:
        st.header("Settings")
        user_id = st.text_input("User ID", value=st.session_state.get("user_id", "default"))
        primary_provider = st.selectbox(
            "Primary provider",
            options=["openai", "ollama"],
            index=0 if st.session_state.get("primary_provider", DEFAULT_PRIMARY_PROVIDER) == "openai" else 1,
        )
        primary_model = st.text_input(
            "Primary model (diagnosis + plan)",
            value=st.session_state.get("primary_model", DEFAULT_PRIMARY_MODEL),
        )
        daily_provider = st.selectbox(
            "Daily provider",
            options=["ollama", "openai"],
            index=0 if st.session_state.get("daily_provider", DEFAULT_DAILY_PROVIDER) == "ollama" else 1,
        )
        daily_model = st.text_input(
            "Daily model (check-ins)",
            value=st.session_state.get("daily_model", DEFAULT_DAILY_MODEL),
        )
        summary_every = st.number_input("Summarize every N turns", min_value=1, max_value=10, value=3, step=1)
        episode_k = st.number_input("Episode top-k", min_value=1, max_value=12, value=6, step=1)
        canon_k = st.number_input("Canon top-k", min_value=1, max_value=12, value=4, step=1)
        episode_min_sim = st.slider("Episode min similarity", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
        force_rebuild = st.checkbox("Rebuild index", value=False)
        dev_mode = st.checkbox("Developer mode (show RAG + prompt)", value=False)

    st.session_state["user_id"] = user_id
    st.session_state["primary_model"] = primary_model
    st.session_state["daily_model"] = daily_model
    st.session_state["primary_provider"] = primary_provider
    st.session_state["daily_provider"] = daily_provider

    if not principles_path.exists():
        st.error("principles.jsonl not found. Run scripts/05_build_principles.py first.")
        st.stop()

    principles, canon, embedder = init_resources(str(principles_path), str(canon_path))

    chroma_dir = paths.data_dir / "index" / "chroma_coach"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = init_chroma_client(chroma_dir, force_rebuild=force_rebuild)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

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
                state["followup_questions"] = generate_followup_questions(primary_provider, primary_model, intake)
                state["followup_answers"] = {}
                state["followup_complete"] = False
                state["plan_generated"] = False
                state["phase"] = "intake"
                save_session(session_path, state)
                rerun_app()
        else:
            with st.expander("Intake", expanded=False):
                st.json(state.get("intake", {}))
            if st.button("Update intake"):
                state["intake"] = {}
                state["followup_questions"] = []
                state["followup_answers"] = {}
                state["followup_complete"] = False
                state["plan_generated"] = False
                state["phase"] = "intake"
                save_session(session_path, state)
                rerun_app()

        if state.get("followup_questions") and not state.get("followup_complete"):
            with st.form("followup_form"):
                st.subheader("Follow-up questions")
                answers = {}
                for idx, question in enumerate(state["followup_questions"], start=1):
                    key = f"followup_{idx}"
                    answers[question] = st.text_input(question, key=key)
                submitted = st.form_submit_button("Save follow-up answers")
            if submitted:
                state["followup_answers"] = answers
                state["followup_complete"] = True
                state["plan_generated"] = False
                state["phase"] = "prescription"
                save_session(session_path, state)
                rerun_app()
        else:
            checkin_required = state.get("phase") == "checkin" and not checkin_done_today(state.get("checkins", []))
            if checkin_required:
                st.info("Daily check-in required before continuing.")
                checkin = render_checkin_form()
                if checkin:
                    state.setdefault("checkins", []).append(checkin)
                    state["pending_prompt"] = (
                        "Daily check-in completed. Provide today's focus and instructions."
                    )
                    state["pending_mode"] = "checkin"
                    save_session(session_path, state)
                    rerun_app()
            else:
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

    prompt = None
    mode = "adhoc"
    if state.get("pending_prompt"):
        prompt = state.get("pending_prompt")
        mode = state.get("pending_mode") or "checkin"
    elif state.get("followup_questions") and not state.get("followup_complete"):
        st.info("Answer the follow-up questions to unlock the coach chat.")
    else:
        checkin_required = state.get("phase") == "checkin" and not checkin_done_today(state.get("checkins", []))
        if checkin_required:
            st.info("Daily check-in required before continuing.")
        else:
            prompt = st.chat_input("Ask a rehab question")

    auto_plan_ready = bool(state.get("followup_complete") and not state.get("plan_generated"))
    if auto_plan_ready:
        st.info("Generating your initial rehab plan from intake and follow-ups.")
        prompt = "Generate my initial rehab plan based on my intake and follow-up answers."
        mode = "prescription"

    if prompt:
        with st.spinner("Thinking..."):
            triage_intake = dict(state.get("intake", {}))
            if state.get("followup_answers"):
                triage_intake["followup_answers"] = state["followup_answers"]
            triage_data = triage(primary_provider, primary_model, prompt, triage_intake, state.get("summary", ""))
            state["last_triage"] = triage_data
            if triage_data.get("needs_medical"):
                response = (
                    "This may be a medical issue or a red-flag presentation. "
                    "I am not providing a rehab plan. Get assessed by a qualified clinician."
                )
            else:
                tags = triage_data.get("tags", [])
                query_parts = [prompt, triage_data.get("query_expansion", "")]
                query_text = " ".join(
                    normalize_query_part(part) for part in query_parts if part
                ).strip() or prompt

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
                    f"Intake: {json.dumps(triage_intake, ensure_ascii=True)}\n\n"
                    f"Latest check-in: {json.dumps(state.get('checkins', [])[-1:], ensure_ascii=True)}\n\n"
                    f"Progression signal: {prog_signal}\n\n"
                    f"Triage: {json.dumps(triage_data, ensure_ascii=True)}\n\n"
                    + format_context(episode_selected, "Episode principles", max_items=int(episode_k))
                    + format_context(canon_items, "Canon principles", max_items=int(canon_k))
                )
                if coverage_note:
                    user_prompt += f"\nNote: {coverage_note}\n"

                if mode == "prescription" or state.get("phase") == "prescription":
                    system_prompt = PRESCRIPTION_PROMPT
                    active_model = primary_model
                    active_provider = primary_provider
                elif mode == "checkin":
                    system_prompt = CHECKIN_PROMPT
                    active_model = daily_model
                    active_provider = daily_provider
                else:
                    system_prompt = ADHOC_PROMPT
                    active_model = daily_model
                    active_provider = daily_provider
                if dev_mode:
                    st.session_state["last_prompt"] = {
                        "system": system_prompt,
                        "user": user_prompt,
                        "phase": state.get("phase", ""),
                        "provider": active_provider,
                        "model": active_model,
                        "retrieved_episode": [item.text for item in episode_selected],
                        "retrieved_canon": [item.text for item in canon_items],
                    }
                response = call_chat(active_provider, active_model, system_prompt, user_prompt)

        state["history"].append({"role": "user", "content": prompt, "ts": time.time()})
        state["history"].append({"role": "assistant", "content": response, "ts": time.time()})
        state["turn_count"] = int(state.get("turn_count", 0)) + 1
        if auto_plan_ready:
            state["plan_generated"] = True
            state["phase"] = "checkin"
        if state.get("pending_prompt"):
            state["pending_prompt"] = ""
            state["pending_mode"] = ""

        if state["turn_count"] % int(summary_every) == 0:
            state["summary"] = update_summary(
                primary_provider,
                primary_model,
                state.get("summary", ""),
                state.get("intake", {}),
                state["history"],
            )

        save_session(session_path, state)
        rerun_app()

    if dev_mode and st.session_state.get("last_prompt"):
        with st.expander("Developer mode: prompt + retrieval", expanded=False):
            payload = st.session_state["last_prompt"]
            st.markdown("**Phase:** " + payload.get("phase", ""))
            st.markdown("**System prompt**")
            st.code(payload.get("system", ""), language="text")
            st.markdown("**User prompt**")
            st.code(payload.get("user", ""), language="text")
            st.markdown("**Retrieved episode principles**")
            st.write(payload.get("retrieved_episode", []))
            st.markdown("**Retrieved canon principles**")
            st.write(payload.get("retrieved_canon", []))


if __name__ == "__main__":
    main()
