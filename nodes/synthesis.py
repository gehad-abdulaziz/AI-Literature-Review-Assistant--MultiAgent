"""
Phase 7 — Cross-Paper Synthesis (Steps 27-28).
"""
from state import GraphState
from llm import call_llm

SYNTHESIS_SYSTEM = (
    "You synthesize findings across MULTIPLE research papers. Do not simply "
    "concatenate per-paper summaries — produce genuinely comparative, "
    "cross-paper reasoning: the shared research landscape, a structured "
    "comparison across papers, common assumptions, conflicting findings, "
    "repeated limitations, and underexplored combinations of approaches. "
    "Only some papers may be included below (failed analyses are excluded); "
    "treat the included set as the complete evidence base."
)


def synthesis_node(state: GraphState) -> dict:
    current = state["current_round"]
    # paper_analyses uses an accumulating (operator.add) reducer, so it holds
    # analyses from EVERY round in this session, not just the current one.
    # Scope down to this round's approved papers before using it.
    approved_ids = {p["id"] for p in current.get("approved_papers", [])}
    analyses = [
        a for a in state["paper_analyses"]
        if not a.get("failed") and a.get("paper_id") in approved_ids
    ]

    if not analyses:
        return {"current_round": {**current, "synthesis": "No successful paper analyses to synthesize."}}

    analyses_text = "\n\n".join(
        f"Paper [{a['paper_id']}] {a['title']}\n"
        f"Problem: {a.get('problem', '')}\n"
        f"Methodology: {a.get('methodology', '')}\n"
        f"Findings: {a.get('findings', '')}\n"
        f"Limitations: {a.get('limitations', '')}"
        for a in analyses
    )
    synthesis = call_llm(
        SYNTHESIS_SYSTEM,
        f"Research topic: {current['topic']}\n\nPaper analyses:\n{analyses_text}",
    )
    return {"current_round": {**current, "synthesis": synthesis}}
