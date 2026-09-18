# Literature Review Agent — Full-Stack App

A human-in-the-loop literature review assistant:

```
[1. Query Expansion] -> [2. Paper Fetching] -> [3. Human Selection (PAUSED)] -> [4. Review Synthesis]
```

- **Backend**: FastAPI + LangGraph (`StateGraph`, `interrupt()`, `Command`, `MemorySaver`). Paper data is
  mocked (offline, no external API keys required) so the whole thing runs out of the box — swap
  `_mock_fetch_papers` in `backend/app/graph.py` for a real search API when you're ready.
- **Frontend**: Next.js 14 (App Router) + Tailwind CSS. Dark, academic-journal-inspired dashboard with a
  4-stage stepper, an interactive paper selection grid, and a rendered report view.

## Project layout

```
lit-review-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── schemas.py     # Pydantic request/response models
│   │   ├── graph.py       # LangGraph state machine (the 4 nodes + interrupt)
│   │   └── main.py        # FastAPI app: /api/start, /api/resume
│   └── requirements.txt
└── frontend/
    ├── src/app/
    │   ├── layout.tsx
    │   ├── globals.css
    │   └── page.tsx        # The dashboard
    ├── package.json
    ├── tailwind.config.ts
    ├── postcss.config.mjs
    ├── next.config.mjs
    ├── tsconfig.json
    └── .env.local.example
```

## 1. Run the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API is now live at `http://localhost:8000`. No `.env` file or API keys are needed — the paper
search is mocked. Check it with:

```bash
curl http://localhost:8000/api/health
```

### Endpoints

| Method | Path          | Body                                                | Behavior                                                                 |
|--------|---------------|------------------------------------------------------|---------------------------------------------------------------------------|
| POST   | `/api/start`  | `{ "topic": "..." }`                                 | Runs the graph to the human-selection interrupt, returns candidate papers |
| POST   | `/api/resume` | `{ "thread_id": "...", "approved_ids": [...] }`      | Resumes the paused graph, returns the finished report                     |
| GET    | `/api/health` | —                                                     | Liveness check                                                            |

## 2. Run the frontend

In a second terminal:

```bash
cd frontend
npm install
cp .env.local.example .env.local   # points the UI at http://localhost:8000
npm run dev
```

Open `http://localhost:3000`. Enter a topic, click **Start review**, select which candidate papers to
keep, click **Approve & Resume**, and the synthesized report renders below.

## How the human-in-the-loop pause works

1. `POST /api/start` calls `compiled_graph.invoke(...)`. The graph runs `expand_queries` →
   `fetch_papers` → `human_selection`.
2. Inside `human_selection_node`, `interrupt(payload)` suspends execution and the payload (the
   candidate list) is returned to FastAPI as `result["__interrupt__"]`. FastAPI relays it to the
   frontend as `StartResponse.candidates`, along with the `thread_id` (LangGraph's `MemorySaver`
   checkpoints the paused state under that thread).
3. The frontend renders the candidates as selectable cards; the header shows a glowing **Paused —
   awaiting you** badge on the "Human Selection" step.
4. When the user clicks **Approve & Resume**, the frontend calls `POST /api/resume` with the
   `thread_id` and the selected paper ids. FastAPI resumes the same thread with
   `compiled_graph.invoke(Command(resume=resume_value), config=config)`, which continues into
   `synthesize_review` and runs to completion.
5. FastAPI returns the finished `ReviewReport`; the frontend renders it.

## Notes

- The in-memory checkpointer (`MemorySaver`) means paused threads are lost if the backend process
  restarts — swap in a persistent checkpointer (e.g. Postgres/SQLite) for production use.
- CORS in `backend/app/main.py` is preconfigured for `http://localhost:3000`; update
  `allow_origins` if you deploy the frontend elsewhere.
- To wire in a real paper search (e.g. the `arxiv` package, as in the reference implementation this
  app is based on) or a real LLM for query expansion/synthesis, replace `_mock_fetch_papers`,
  `expand_queries_node`, and `synthesize_review_node` in `backend/app/graph.py` — the API and
  frontend don't need to change since they only depend on the `PaperDict` / `GapDict` shapes.
