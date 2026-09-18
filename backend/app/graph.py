"""
LangGraph state machine for the Literature Review Agent.

Flow:
    START -> expand_queries -> fetch_papers -> human_selection (interrupt)
          -> synthesize_review -> END

`human_selection` is where the graph pauses with `interrupt(...)`. The
caller (see app/main.py) resumes it with `Command(resume=...)` carrying the
user's approved paper ids.

Paper metadata here is mocked (no external API calls) so the whole graph
runs deterministically and fully offline — swap `_mock_fetch_papers` for a
real search (e.g. the arXiv API) when you're ready to go live, keeping the
same `PaperDict` return shape so nothing downstream needs to change.
"""
from __future__ import annotations

import random
import uuid
from typing import Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PaperDict(TypedDict):
    id: str
    title: str
    authors: list[str]
    year: Optional[int]
    abstract: str
    url: str
    relevance_score: float


class GapDict(TypedDict):
    id: str
    title: str
    description: str
    potential_direction: str


class GraphState(TypedDict, total=False):
    topic: str
    search_queries: list[str]
    candidates: list[PaperDict]
    approved_papers: list[PaperDict]
    synthesis: str
    comparison: str
    gaps: list[GapDict]
    step: Literal[
        "query_expansion",
        "paper_fetching",
        "human_selection",
        "review_synthesis",
        "done",
    ]


# ---------------------------------------------------------------------------
# Mock data helpers — deterministic-ish, offline, no external calls
# ---------------------------------------------------------------------------

_MOCK_VENUES = ["NeurIPS", "ICML", "ICLR", "AAAI", "ACL", "CVPR", "arXiv preprint"]

_MOCK_AUTHOR_POOL = [
    "A. Chen", "M. Rodriguez", "S. Patel", "J. Kim", "L. Novak",
    "R. Okafor", "T. Nakamura", "E. Fischer", "P. Andersson", "N. Osei",
]


def _mock_fetch_papers(query: str, n: int = 5) -> list[PaperDict]:
    """Generates plausible-looking candidate papers for a query.

    Stands in for a real academic search API (arXiv, Semantic Scholar,
    etc). Swap this function out for a real call and keep the return shape
    the same — nothing else in the graph needs to change.
    """
    rng = random.Random(hash(query) & 0xFFFFFFFF)
    papers: list[PaperDict] = []
    for i in range(n):
        year = rng.randint(2019, 2025)
        venue = rng.choice(_MOCK_VENUES)
        authors = rng.sample(_MOCK_AUTHOR_POOL, k=rng.randint(2, 4))
        title = f"{query.strip().title()}: A Study on Approach #{i + 1} ({venue} {year})"
        papers.append(
            PaperDict(
                id=str(uuid.uuid4()),
                title=title,
                authors=authors,
                year=year,
                abstract=(
                    f"We investigate {query.strip().lower()} and propose a method "
                    f"that improves performance over prior baselines. Our approach "
                    f"is evaluated on standard benchmarks and shows measurable gains, "
                    f"while also surfacing open challenges for future work."
                ),
                url=f"https://example.org/papers/{rng.randint(1000, 9999)}",
                relevance_score=round(rng.uniform(0.55, 0.98), 2),
            )
        )
    return papers


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def expand_queries_node(state: GraphState) -> GraphState:
    """Step 1 — Query Expansion.

    Expands the raw topic into a handful of related search queries. This
    is a rule-based expansion (no LLM call) to keep the demo self-contained;
    swap in an LLM call here for real semantic query expansion.
    """
    topic = state["topic"].strip()
    expansions = [
        topic,
        f"{topic} survey",
        f"{topic} benchmark evaluation",
        f"recent advances in {topic}",
        f"{topic} open problems",
    ]
    return {"search_queries": expansions, "step": "query_expansion"}


def fetch_papers_node(state: GraphState) -> GraphState:
    """Step 2 — Paper Fetching.

    Runs the (mock) search for each expanded query and merges + dedupes
    results by title, keeping the highest relevance score per title.
    """
    seen: dict[str, PaperDict] = {}
    for query in state["search_queries"]:
        for paper in _mock_fetch_papers(query, n=4):
            existing = seen.get(paper["title"])
            if existing is None or paper["relevance_score"] > existing["relevance_score"]:
                seen[paper["title"]] = paper

    candidates = sorted(seen.values(), key=lambda p: p["relevance_score"], reverse=True)[:12]
    return {"candidates": candidates, "step": "paper_fetching"}


def human_selection_node(state: GraphState) -> GraphState:
    """Step 3 — Human Selection (PAUSED).

    Pauses the graph with `interrupt(...)`, handing the candidate list to
    the caller. Execution resumes only when `Command(resume=...)` is sent
    back in with the user's chosen paper ids (or `approve_all`).
    """
    payload = {
        "type": "human_paper_selection",
        "message": "Select the papers to include in the review, then approve.",
        "candidates": state["candidates"],
    }
    decision = interrupt(payload)

    if decision.get("approve_all"):
        approved = state["candidates"]
    else:
        approved_ids = set(decision.get("approved_ids") or [])
        approved = [p for p in state["candidates"] if p["id"] in approved_ids]
        if not approved:
            # Nothing matched / empty selection — fall back to all
            # candidates rather than synthesizing a review from zero papers.
            approved = state["candidates"]

    return {"approved_papers": approved, "step": "human_selection"}


def synthesize_review_node(state: GraphState) -> GraphState:
    """Step 4 — Review Synthesis.

    Produces the final structured report: a synthesis paragraph, a
    cross-paper comparison, and a set of research gaps. This is a
    template-based synthesis (no LLM call) so the demo runs fully offline;
    swap in an LLM call here for real synthesis over `approved_papers`.
    """
    papers = state["approved_papers"]
    topic = state["topic"]

    years = sorted({p["year"] for p in papers if p.get("year")})
    year_range = f"{years[0]}\u2013{years[-1]}" if years else "recent years"

    synthesis = (
        f"Across the {len(papers)} selected papers on '{topic}', the "
        f"literature ({year_range}) converges on a small set of recurring "
        f"themes: benchmark-driven evaluation, incremental architectural "
        f"improvements over established baselines, and growing attention to "
        f"evaluation robustness. Most works report gains on standard "
        f"benchmarks but differ in how thoroughly they stress-test "
        f"generalization beyond the reported setting."
    )

    comparison_lines = []
    for p in papers:
        comparison_lines.append(
            f"- **{p['title']}** ({p.get('year', 'n.d.')}) \u2014 relevance "
            f"{p['relevance_score']:.2f}, authors: {', '.join(p['authors'])}."
        )
    comparison = "\n".join(comparison_lines) if comparison_lines else "No papers were selected."

    gap_templates = [
        (
            "Limited cross-domain evaluation",
            f"Most reviewed work on {topic} is evaluated within a single "
            f"benchmark family, leaving generalization across domains "
            f"under-explored.",
            "Design a cross-domain benchmark suite and re-evaluate leading "
            "methods on it to measure true transferability.",
        ),
        (
            "Sparse ablation reporting",
            "Several papers report headline results without systematic "
            "ablations, making it hard to attribute gains to specific "
            "design choices.",
            "Standardize an ablation protocol (component-by-component) for "
            "future comparative studies.",
        ),
        (
            "Reproducibility gaps",
            "Compute budgets, seeds, and hyperparameter search details are "
            "inconsistently reported across the reviewed papers.",
            "Adopt a shared reporting checklist (e.g. compute, seeds, search "
            "space) to make follow-up comparisons fair.",
        ),
    ]
    gaps: list[GapDict] = [
        GapDict(
            id=str(uuid.uuid4()),
            title=title,
            description=desc,
            potential_direction=direction,
        )
        for title, desc, direction in gap_templates
    ]

    return {
        "synthesis": synthesis,
        "comparison": comparison,
        "gaps": gaps,
        "step": "review_synthesis",
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("expand_queries", expand_queries_node)
    graph.add_node("fetch_papers", fetch_papers_node)
    graph.add_node("human_selection", human_selection_node)
    graph.add_node("synthesize_review", synthesize_review_node)

    graph.add_edge(START, "expand_queries")
    graph.add_edge("expand_queries", "fetch_papers")
    graph.add_edge("fetch_papers", "human_selection")
    graph.add_edge("human_selection", "synthesize_review")
    graph.add_edge("synthesize_review", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


compiled_graph = build_graph()
