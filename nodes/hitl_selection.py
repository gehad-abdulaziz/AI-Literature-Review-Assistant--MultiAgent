"""
Phase 5 — Human-in-the-Loop: Paper Selection (Steps 20-21).

Uses LangGraph's `interrupt` (requires a checkpointer, wired in graph.py).
The graph pauses here; whatever the caller resumes with via `Command(resume=...)`
becomes the return value of `interrupt(...)`.
"""
from langgraph.types import interrupt

from state import GraphState


def paper_selection_node(state: GraphState) -> dict:
    current = state["current_round"]
    ranked = current.get("ranked_papers", [])

    # Step 20: pause and present the ranked list for approval/adjustment.
    payload = {
        "type": "paper_selection",
        "message": "Review these candidate papers. Approve as-is or provide "
        "the list of paper ids to keep.",
        "candidates": [
            {"id": p["id"], "title": p["title"], "year": p.get("year")}
            for p in ranked
        ],
    }
    user_response = interrupt(payload)

    # Step 21: resume with the corrected list, not the original.
    # Expected user_response shape: {"approved_ids": [...]}  or {"approve_all": True}
    if isinstance(user_response, dict) and user_response.get("approve_all"):
        approved = ranked
    elif isinstance(user_response, dict) and "approved_ids" in user_response:
        keep_ids = set(user_response["approved_ids"])
        approved = [p for p in ranked if p["id"] in keep_ids]
    else:
        # Defensive default: if the resume payload is malformed, keep everything
        # rather than silently dropping the round's papers.
        approved = ranked

    return {"current_round": {**current, "approved_papers": approved}}
