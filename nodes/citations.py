"""
Phase 9 — Evidence / Citation Validation (Steps 31-33).
"""
from state import GraphState
from llm import call_llm_json

SUPPORT_SYSTEM = (
    "Given a claim and the analysis of the paper it cites, classify whether "
    "the paper's actual content supports the claim. Return JSON: "
    "{\"label\": \"supported\" | \"weakly_supported\" | \"unsupported\"}"
)


def citation_check_node(state: GraphState) -> dict:
    current = state["current_round"]
    papers_by_id = {p["id"]: p for p in current.get("approved_papers", [])}
    # Same reducer caveat as synthesis.py — scope to this round's papers.
    analyses_by_id = {
        a["paper_id"]: a
        for a in state["paper_analyses"]
        if not a.get("failed") and a["paper_id"] in papers_by_id
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

            result = call_llm_json(
                SUPPORT_SYSTEM,
                f"Claim: {claim}\n\nPaper analysis: {analysis}",
            )
            label = result.get("label", "unsupported") if isinstance(result, dict) else "unsupported"
            checks.append({"claim": claim, "paper_id": paper_id, "exists": True, "support_label": label})

    # Step 33: simplest MVP option — flag unsupported claims inline rather
    # than dropping them or triggering re-synthesis.
    return {"current_round": {**current, "citation_checks": checks}}
