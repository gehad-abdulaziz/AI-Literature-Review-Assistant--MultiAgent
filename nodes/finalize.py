"""
Phase 11 / Step 38 — fold the completed current_round into research_rounds.
"""
import datetime

from state import GraphState


def finalize_round_node(state: GraphState) -> dict:
    current = state["current_round"]
    round_id = current["round_id"]
    approved_ids = {p["id"] for p in current.get("approved_papers", [])}
    round_entry = {
        "round_id": round_id,
        "topic": current["topic"],
        "papers": current.get("approved_papers", []),
        # BUGFIX: scope by round_id (see analysis.py/synthesis.py/citations.py)
        # in addition to paper_id, since paper_analyses accumulates for the
        # whole thread and a paper can recur across rounds.
        "analyses": [
            a for a in state["paper_analyses"]
            if not a.get("failed")
            and a.get("round_id") == round_id
            and a.get("paper_id") in approved_ids
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
        # citations, this node) now filters by round_id + paper_id, so a
        # paper recurring in a later round no longer collides with its
        # earlier-round analysis. Still worth revisiting for memory/perf if
        # a thread runs very many rounds.
    }