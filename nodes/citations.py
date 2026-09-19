"""
Phase 9 — Evidence / Citation Validation (Steps 31-33).
"""
from state import GraphState
from llm import call_llm_json, LLMCallError

SUPPORT_SYSTEM = (
    "Given a claim and the analysis of the paper it cites, classify whether "
    "the paper's actual content supports the claim. Return JSON: "
    "{\"label\": \"supported\" | \"weakly_supported\" | \"unsupported\"}"
)


def citation_check_node(state: GraphState) -> dict:
    current = state["current_round"]
    round_id = current.get("round_id")
    papers_by_id = {p["id"]: p for p in current.get("approved_papers", [])}
    # BUGFIX: paper_analyses uses an accumulating (operator.add) reducer, so
    # it holds analyses from EVERY round in this thread, not just this one.
    # The same arXiv paper can be approved again in a later round, which
    # previously produced a second entry with the same paper_id — filtering
    # only by paper_id membership would then silently pick up whichever
    # round's analysis happened to match. Scoping by round_id makes this
    # round's lookup unambiguous.
    analyses_by_id = {
        a["paper_id"]: a
        for a in state["paper_analyses"]
        if not a.get("failed")
        and a.get("round_id") == round_id
        and a["paper_id"] in papers_by_id
    }
    gaps = current.get("gaps", [])

    checks = []
    for gap in gaps:
        claim = gap.get("potential_direction", "")
        for paper_id in gap.get("supporting_paper_ids", []):
            # Step 31: deterministic existence check, no LLM.
            exists = paper_id in papers_by_id
            if not exists:
                checks.append(
                    {"claim": claim, "paper_id": paper_id, "exists": False, "support_label": "unsupported"}
                )
                continue

            # Step 32: narrow LLM classification, only for claims that exist.
            analysis = analyses_by_id.get(paper_id)
            if analysis is None:
                checks.append(
                    {"claim": claim, "paper_id": paper_id, "exists": True, "support_label": "unsupported"}
                )
                continue

            try:
                result = call_llm_json(
                    SUPPORT_SYSTEM,
                    f"Claim: {claim}\n\nPaper analysis: {analysis}",
                )
                label = result.get("label", "unsupported") if isinstance(result, dict) else "unsupported"
            except (LLMCallError, ValueError):
                # Fail closed: if we can't verify the claim, don't mark it
                # "supported" by default — one failed classification
                # shouldn't be able to silently green-light a citation.
                label = "unsupported"

            checks.append({"claim": claim, "paper_id": paper_id, "exists": True, "support_label": label})

    # Step 33: simplest MVP option — flag unsupported claims inline rather
    # than dropping them or triggering re-synthesis.
    return {"current_round": {**current, "citation_checks": checks}}