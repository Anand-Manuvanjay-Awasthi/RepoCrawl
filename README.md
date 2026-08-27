# RepoLens

RepoLens lets you point at a public GitHub repository and ask natural-language
questions about the codebase. It answers using Retrieval-Augmented Generation
(RAG): relevant code is retrieved from the indexed repository first, and the
answer is generated only from that retrieved evidence — never invented.

> **Status:** under active phased implementation. This README will be
> completed with full setup, architecture, and API documentation once the
> project is finished (final phase). For now it documents what's runnable
> today.

## Tech Stack

**Backend:** Python 3.11+, FastAPI, LangChain/LangGraph, ChromaDB, Hugging
Face Inference API (Qwen2.5-7B-Instruct), GitPython.

**Frontend:** React 19, Vite, Tailwind CSS, Axios, react-markdown,
react-syntax-highlighter.

## Project Structure

```
RepoLens/
├── backend/     # FastAPI application (RAG pipeline, indexing, chat API)
└── frontend/    # React + Vite single-page app
```

## Current Progress

- [x] Phase 1 — Project foundation (backend skeleton, config, health check,
      frontend skeleton)
- [ ] Phase 2 — Repository ingestion (cloning, loading, filtering)
- [ ] Phase 3 — Indexing pipeline (chunking, embeddings, ChromaDB)
- [ ] Phase 4 — Repository intelligence (tech stack, repo map, suggestions)
- [ ] Phase 5 — Retrieval + intent classification
- [ ] Phase 6 — RAG generation (`/api/chat`)
- [ ] Phase 7 — Frontend experience (landing page, workspace, chat UI)
- [ ] Phase 8 — Integration, polish, final documentation

## Running the current skeleton

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env   # fill in HUGGINGFACEHUB_API_TOKEN when needed (Phase 6+)
uvicorn app.main:app --reload --port 8000
```

Verify it's up: `curl http://127.0.0.1:8000/api/health`

### Frontend

```bash
cd frontend
npm install
cp .env.example .env   # not required for local dev (Vite proxies /api)
npm run dev
```

Open the printed local URL (default `http://127.0.0.1:5173`). The page
shows a small "Backend: ok" indicator once it can reach the FastAPI health
endpoint — this confirms the two apps are wired together correctly.

## License

TBD.
