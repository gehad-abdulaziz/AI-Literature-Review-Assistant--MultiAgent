"""
Phase 11 / Step 38 — fold the completed current_round into research_rounds.
"""
import datetime

from state import GraphState


def finalize_round_node(state: GraphState) -> dict:
    current = state["current_round"]
    approved_ids = {p["id"] for p in current.get("approved_papers", [])}
    round_entry = {
        "round_id": current["round_id"],
        "topic": current["topic"],
        "papers": current.get("approved_papers", []),
        "analyses": [
            a for a in state["paper_analyses"]
            if not a.get("failed") and a.get("paper_id") in approved_ids
        ],
        "synthesis": current.get("synthesis", ""),
        "gaps": current.get("gaps", []),
        "directions": current.get("directions", []),
        "citation_checks": current.get("citation_checks", []),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    return {
        "research_rounds": [round_entry],  # accumulated via reducer
        "current_round": {},               # reset scratch space (plain overwrite)
        # NOTE: paper_analyses uses operator.add, so it can't be reset here —
        # returning [] is a no-op under that reducer. It keeps growing across
        # rounds for the life of the thread; every consumer of it (synthesis,
        # citations, this node) filters by current-round paper ids instead of
        # relying on it being scoped. Fine for MVP; revisit if it becomes a
        # real memory/perf concern with many rounds.
    }
