# dgr-chatbot

Rehab coaching chatbot grounded in David Grey Rehab (DGR) podcast principles.
Uses a principles-first RAG pipeline and local LLMs (Ollama) for plan generation.

## Project overview

This project builds a rehab coaching assistant that:
- Extracts rehab principles from DGR podcast transcripts.
- Uses those principles to guide rehab plans (not verbatim quotes).
- Runs an intake and follow-up flow, then generates a plan.
- Switches to daily check-ins with a coach voice to drive adherence.

## Scope and boundaries

- Primary users: general population who cannot afford or will not access a physio.
- Scope: daily-life and sports-related injuries, not major trauma.
- No medical advice for serious conditions; redirect to professional care when needed.
- Plans are based on a working diagnosis hypothesis and adjusted by response over 10-14 days.

## Design decisions (summary)

- Principles-first RAG (episode principles + canon fallback).
- Phase-based flow: intake -> prescription -> daily check-ins.
- Stable early-phase plan with progressive overload from a safe entry point.
- Automatic plan generation after intake and follow-ups.
- Prompts are centralized in src/dgr_rag/prompts.py.
- Full details live in DESIGN_DECISIONS.txt.

## Pipeline

1) Fetch playlist metadata -> data/episode_index.csv
2) Download YouTube transcripts -> data/raw/transcripts/
3) Build principles + canon -> data/processed/principles.jsonl and data/processed/canon.jsonl
4) Run the Streamlit coach app (app.py)

## Setup

Create and activate a venv.

Windows PowerShell:
```
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:
```
python -m venv .venv
.\.venv\Scripts\activate.bat
```

Install deps:
```
pip install -r Requirements.txt
```

Copy .env.example to .env and set PLAYLIST_URL.

## Ollama

This project uses Ollama locally for LLM calls.
Make sure Ollama is running and the models are pulled.

Example:
```
ollama serve
ollama pull llama3.1:8b
ollama pull phi3:mini
```

## OpenAI (optional)

The prescription phase can use OpenAI GPT-4o. Set `OPENAI_API_KEY` in `.env`.
The Streamlit app lets you choose the provider and model for primary and daily phases.

## Build data

```
python scripts/01_fetch_playlist_index.py
python scripts/02_download_transcripts.py
python scripts/05_build_principles.py
```

If you only need to rebuild the canon from existing principles:
```
python scripts/05_build_principles.py --canon-only
```

## Run the Streamlit app

```
streamlit run app.py
```

The app uses a primary model for diagnosis and plan generation, and a daily model for check-ins.
You can change these in the Streamlit sidebar.

## Useful scripts

- scripts/03_make_chunks.py: chunk raw transcripts (quote-friendly or guidance mode)
- scripts/05_build_principles.py: extract per-episode principles and canon
- scripts/06_retrieval_principles_demo.py: principles-only retrieval demo
- scripts/07_rehab_coach_cli.py: CLI coach prototype

## Prompt configuration

Prompts are centralized in src/dgr_rag/prompts.py.
Design decisions live in DESIGN_DECISIONS.txt.

