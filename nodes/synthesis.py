"""
Phase 7 — Cross-Paper Synthesis (Steps 27-28).
"""
from state import GraphState
from llm import call_llm, LLMCallError

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
    round_id = current.get("round_id")
    # BUGFIX: paper_analyses accumulates across the whole thread (operator.add
    # reducer). If the same paper is approved again in a later round, filtering
    # by paper_id alone would pick up both rounds' entries for it. Scope by
    # round_id (added to PaperAnalysis in state.py) as well as paper_id.
    approved_ids = {p["id"] for p in current.get("approved_papers", [])}
    analyses = [
        a for a in state["paper_analyses"]
        if not a.get("failed")
        and a.get("round_id") == round_id
        and a.get("paper_id") in approved_ids
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

    try:
        synthesis = call_llm(
            SYNTHESIS_SYSTEM,
            f"Research topic: {current['topic']}\n\nPaper analyses:\n{analyses_text}",
        )
    except LLMCallError:
        synthesis = (
            "Synthesis could not be generated due to an LLM service error. "
            "Individual paper analyses are still available above."
        )

    return {"current_round": {**current, "synthesis": synthesis}}