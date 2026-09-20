# AI Literature Review Assistant

A multi-agent research assistant, built on **LangGraph**, that takes a research topic, searches arXiv, filters and ranks candidate papers, lets a human approve the final reading list, analyzes each paper in parallel, synthesizes cross-paper findings, surfaces research gaps with explicit epistemic labeling, validates its own citations, and answers follow-up questions about any completed round — all with persistent, resumable, multi-turn conversation state.

It's built to demonstrate (and actually use, not just simulate) the core patterns of a production agentic system: a stateful graph with conditional routing, parallel fan-out/fan-in, human-in-the-loop interrupts, structured-output validation, and graceful degradation when an LLM call fails.

---

## What it actually does

1. You give it a research topic.
2. It generates search queries, searches arXiv, deduplicates results, and has an LLM rank them for relevance — retrying with broadened queries if too few relevant papers turn up.
3. **It pauses and hands you the candidate list.** You approve all of them or pick a subset. Nothing gets analyzed without your sign-off.
4. Each approved paper is analyzed **in parallel** (problem, methodology, dataset, findings, limitations).
5. An LLM synthesizes the individual analyses into genuine cross-paper reasoning — not a concatenation of summaries: shared landscape, comparisons, conflicting findings, repeated limitations.
6. It identifies research gaps and drafts a potential research direction for each one, tagging every gap with an **epistemic label** (`directly_supported` / `reasonable_synthesis` / `hypothesis` / `speculative`) and stripping out unhedged certainty language ("definitely novel") if the model tries to use it.
7. It cross-checks its own gap claims against the papers actually cited — deterministically first (does the cited paper exist in the approved set?), then with a narrow LLM classification (does the paper's content actually support the claim?).
8. It renders the whole round as Markdown.
9. On your next message, an orchestrator classifies whether you're starting new research or asking a follow-up about a round you already ran. Follow-ups are answered **only from the data of that specific round** — no new search, no cross-round bleed. If it's not obvious which round you mean, it asks you.

Everything above is checkpointed, so a conversation can be closed and resumed, and a HITL interrupt can be answered any time later — including after a process restart (see [Persistence](#persistence)).

---

## Architecture

```
                              START
                                │
                          ┌─────▼─────┐
                          │orchestrator│  ← deterministic on round 1,
                          └─────┬─────┘     LLM-classified after that
                new_research    │    follow_up
              ┌─────────────────┴───────────────────┐
              ▼                                      ▼
    generate_search_queries                   resolve_round
              │                                      │
       academic_search (arXiv)              resolved?/ambiguous?
              │                                 │         │
          prefilter                        follow_up  clarify_round
              │                             _answer    (HITL interrupt)
      relevance_ranking (LLM)                   │         │
              │                                 └────┬────┘
     too few? ─┴─ enough?                             │
      (retry)      │                                  │
              paper_selection                          │
             (HITL interrupt)                          │
                    │                                  │
         ┌──────────┴──────────┐                       │
         ▼                     ▼                       │
  analyze_single_paper ×N   (nothing approved)          │
   (parallel fan-out)            │                      │
         └──────────┬────────────┘                      │
                     ▼                                  │
                 synthesis (LLM)                         │
                     │                                  │
             gaps_and_directions (LLM)                   │
                     │                                  │
              citation_check (LLM + deterministic)        │
                     │                                  │
              finalize_round                             │
                     │                                  │
                     └────────────────┬─────────────────┘
                                       ▼
                                    render
                                       │
                                      END
```

Every node above is a real, separately invoked LangGraph node — none of it is a single monolithic LLM call pretending to be a multi-agent system. The parallel analysis step uses LangGraph's `Send` API for genuine fan-out (one graph branch per approved paper, running concurrently), not a Python `for` loop calling the LLM in sequence.

---

## Tech stack (everything here is actually wired in, not aspirational)

| Layer | What's used | Where |
|---|---|---|
| Orchestration | **LangGraph** `StateGraph` — nodes, conditional edges, `Send` fan-out, `interrupt`/`Command` HITL, checkpointing | `graph.py`, every file in `nodes/` |
| LLM | **Groq** via `langchain-groq`'s `ChatGroq`, model configurable via env (`openai/gpt-oss-20b` by default) | `llm.py`, `config.py` |
| Structured output | Hand-rolled JSON-mode prompting + multi-stage repair (fence stripping, outermost-object extraction, backslash-escape repair) since the model isn't using a native structured-output API | `llm.py` |
| Search | **arXiv** via the official `arxiv` Python package (not raw HTTP — arXiv's API rejects bare requests without the right headers/rate-limiting the package handles) | `tools/arxiv_search.py` |
| State | Typed `GraphState` (`TypedDict`) with explicit reducers: `add_messages` for chat history, `operator.add` for accumulating research rounds and paper analyses, plain overwrite for single-writer-per-turn fields | `state.py` |
| Persistence | **SQLite**-backed LangGraph checkpointer (`langgraph-checkpoint-sqlite`) — conversations and pending HITL interrupts survive a process restart | `graph.py` |
| Backend API | **FastAPI** thin wrapper exposing the compiled graph over HTTP (`/chat`, `/resume`, `/health`) | `api.py` |
| Frontend | Single-file HTML/CSS/JS test console — renders assistant Markdown output (via `marked.js`), and turns HITL interrupts into interactive cards (checkboxes for paper selection, buttons for round clarification) instead of requiring hand-typed resume payloads | `frontend/index.html` |
| CLI | Plain `input()`/`print()` loop over the same compiled graph, for terminal-only use | `main.py` |
| Config | `python-dotenv` + environment variables, no hardcoded secrets | `config.py`, `.env` (not committed) |
| Logging | Standard library `logging`, single shared logger config | `logging_config.py` |

---

## Project structure

```
.
├── main.py                 # CLI entry point
├── api.py                  # FastAPI HTTP wrapper over the same compiled graph
├── graph.py                # Wires every node into the StateGraph + checkpointer
├── state.py                # GraphState schema, reducers, typed sub-shapes
├── llm.py                  # Shared ChatGroq instance, call_llm / call_llm_json helpers
├── config.py                # Env-driven configuration
├── logging_config.py       # Shared logging setup
├── requirements.txt
├── frontend/
│   └── index.html          # Local test console (talks to api.py)
├── nodes/
│   ├── orchestrator.py     # Classifies new_research vs follow_up
│   ├── search.py           # Query generation, arXiv search, prefilter, relevance ranking
│   ├── hitl_selection.py   # Paper-selection HITL interrupt
│   ├── analysis.py         # Parallel per-paper analysis (Send fan-out)
│   ├── synthesis.py        # Cross-paper synthesis
│   ├── gaps.py              # Gap detection + research directions + epistemic labeling
│   ├── citations.py        # Citation/claim validation
│   ├── finalize.py         # Folds the finished round into research_rounds
│   ├── follow_up.py        # Round resolution, clarification HITL, follow-up answering
│   └── render.py           # Pure-formatting Markdown renderer
└── tools/
    └── arxiv_search.py     # Deterministic arXiv search tool, no LLM involved
```

---

## Design decisions worth knowing about

**State is explicitly typed and reducer-aware.** `messages`, `research_rounds`, and `paper_analyses` all use accumulating reducers because multiple turns (and, for `paper_analyses`, multiple *parallel branches within a turn*) write to them. Every other field is a plain single-writer-per-turn value. Getting this distinction wrong is a common source of bugs in LangGraph apps that just reach for `operator.add` everywhere or nowhere — here it's chosen deliberately per field, with the reasoning documented in `state.py`.

**Human-in-the-loop is a first-class part of the graph, not a bolt-on.** Two points genuinely pause execution via LangGraph's `interrupt()`: approving/adjusting the paper shortlist, and disambiguating which research round a follow-up question refers to. Both require a checkpointer to actually be resumable across a real pause (not just a `Send` continuation), which is why persistence isn't optional here.

**Round-scoped analysis, not just paper-ID-scoped.** `paper_analyses` accumulates for the life of a conversation thread. If the same paper is approved again in a later round, it's tagged with `round_id` at analysis time so synthesis, citation-checking, and rendering for round 2 can't accidentally pick up round 1's analysis of that paper.

**Deterministic work stays deterministic.** Deduplication, existence checks for cited papers, and the arXiv search itself involve no LLM call — they're plain Python, which makes them fast, free, and impossible to hallucinate. The LLM is reserved for genuinely judgment-based steps: relevance ranking, synthesis, gap identification, claim classification.

**Failure is handled node-by-node, not with a single global try/except.** Every LLM call site distinguishes an API-level failure (`LLMCallError` — network, rate limit, timeout) from a parsing failure (the model responded, but not with valid JSON), and each node degrades in a way appropriate to what it does: paper analysis marks that one paper as failed without killing the parallel batch; citation checks fail closed to "unsupported" rather than silently passing; the orchestrator falls back to the cheaper failure path (`follow_up`'s clarification ask) rather than kicking off an expensive pipeline likely to hit the same outage.

**Persistence is real, not simulated.** Checkpoints — including a conversation paused mid-interrupt — are written to a SQLite file, so a restarted server doesn't lose an in-progress thread. (Single-file SQLite is a single-process story; a multi-instance deployment would swap in `langgraph-checkpoint-postgres` against a shared database, same interface.)

---

## Setup

### 1. Clone and create a virtual environment
```bash
git clone <your-repo-url>
cd ai-literature-review-assistant
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here

# Optional — sensible defaults are used if omitted
MODEL_NAME=openai/gpt-oss-20b
SEMANTIC_SCHOLAR_API_KEY=
MAX_SEARCH_RETRIES=2
MIN_RELEVANT_PAPERS=3
MAX_PAPERS_PER_ROUND=8
```

---

## Running it

**Option A — CLI**
```bash
python main.py
```
Type a research topic, respond to the paper-selection prompt when it pauses, keep chatting for follow-ups.

**Option B — API + browser test console**
```bash
uvicorn api:app --reload --port 8000
```
Then open `frontend/index.html` directly in a browser (no server needed for it — it's a static file that calls `http://localhost:8000`). Paper selection and round clarification render as interactive cards instead of raw JSON.

---
see demo :https://drive.google.com/file/d/1ODgy0YdkF5kUOSrvFJTZIrzTBPj2CS9E/view?usp=sharing

---
## License

Add your preferred license here (MIT is a common default for a project like this).
