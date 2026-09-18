"""
Phase 4 — Search / Retrieval (Steps 15-19).
"""
import uuid

from state import GraphState
from llm import call_llm_json
from tools.arxiv_search import search_arxiv
from config import MAX_SEARCH_RETRIES, MIN_RELEVANT_PAPERS

QUERY_GEN_SYSTEM = (
    "Given a research topic, produce 1-3 concrete search query strings "
    "suitable for a keyword search. Return PLAIN keyword phrases only — "
    "no field prefixes like 'ti:', 'cat:', or 'all:', no boolean operators "
    "like AND/OR, just the bare keywords/phrases themselves "
    "(e.g. \"random forest classification\", not \"ti:random forest\"). "
    "Return JSON: {\"queries\": [\"...\", ...]}"
)

RANKING_SYSTEM = (
    "You are ranking candidate papers for relevance to a research question. "
    "Given the research question and a list of candidate papers (id, title, "
    "abstract), return ONLY the papers that are genuinely relevant. "
    "Return JSON: {\"relevant\": [{\"id\": \"...\", \"justification\": \"...\"}]}"
)


def generate_search_queries_node(state: GraphState) -> dict:
    """Step 15. Only runs on the first attempt; retries append a broaden hint."""
    current = state.get("current_round", {})
    topic = current.get("topic") or state["messages"][-1].content
    round_id = current.get("round_id") or str(uuid.uuid4())[:8]

    retry_count = current.get("retry_count", 0)
    hint = ""
    if retry_count > 0:
        hint = (
            "\nPrevious queries returned too few relevant results. "
            "Broaden or rephrase the query(ies)."
        )
    result = call_llm_json(QUERY_GEN_SYSTEM, f"Research topic: {topic}{hint}")
    queries = result.get("queries", [topic]) if isinstance(result, dict) else [topic]

    return {
        "current_round": {
            **current,
            "round_id": round_id,
            "topic": topic,
            "search_queries": queries,
            "retry_count": retry_count,
        }
    }


def academic_search_node(state: GraphState) -> dict:
    """Step 16. Deterministic — calls the arXiv tool, no LLM."""
    current = state["current_round"]
    all_results: list[dict] = []
    for q in current["search_queries"]:
        all_results.extend(search_arxiv(q, max_results=10))
    return {"current_round": {**current, "raw_results": all_results}}


def prefilter_node(state: GraphState) -> dict:
    """Step 17. Deterministic dedup + missing-abstract filter."""
    current = state["current_round"]
    seen_ids = set()
    seen_titles = set()
    filtered = []
    for p in current["raw_results"]:
        if not p.get("abstract"):
            continue
        key_id = p.get("id")
        key_title = (p.get("title") or "").strip().lower()
        if key_id in seen_ids or key_title in seen_titles:
            continue
        seen_ids.add(key_id)
        seen_titles.add(key_title)
        filtered.append(p)
    return {"current_round": {**current, "filtered_results": filtered}}


def relevance_ranking_node(state: GraphState) -> dict:
    """Step 18. LLM selects/ranks relevant papers from the filtered candidates."""
    current = state["current_round"]
    candidates = current["filtered_results"]
    if not candidates:
        return {"current_round": {**current, "ranked_papers": []}}

    candidate_list = "\n".join(
        f"- id={p['id']} | title={p['title']} | abstract={p['abstract'][:400]}"
        for p in candidates
    )
    result = call_llm_json(
        RANKING_SYSTEM,
        f"Research topic: {current['topic']}\n\nCandidates:\n{candidate_list}",
    )
    relevant_ids = {
        r["id"] for r in result.get("relevant", [])
    } if isinstance(result, dict) else set()

    ranked = [p for p in candidates if p["id"] in relevant_ids]
    return {"current_round": {**current, "ranked_papers": ranked}}


def check_search_sufficiency(state: GraphState) -> str:
    """
    Step 19 conditional edge. Loops back to query generation (with retry_count
    bumped) if too few relevant papers and under the retry cap; otherwise
    proceeds to human-in-the-loop paper selection with whatever was found.
    """
    current = state["current_round"]
    ranked = current.get("ranked_papers", [])
    retry_count = current.get("retry_count", 0)

    if len(ranked) < MIN_RELEVANT_PAPERS and retry_count < MAX_SEARCH_RETRIES:
        return "retry"
    return "proceed"


def bump_retry_node(state: GraphState) -> dict:
    """Small helper node so the retry edge has somewhere to increment state."""
    current = state["current_round"]
    return {"current_round": {**current, "retry_count": current.get("retry_count", 0) + 1}}