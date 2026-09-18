"""
Phase 8 — Gap Detection & Research Direction Generation (Steps 29-30).
"""
from state import GraphState
from llm import call_llm_json

GAPS_SYSTEM = (
    "Given a cross-paper synthesis and a numbered list of the papers it "
    "draws on, identify research gaps using this exact evidence chain for "
    "each: observed_evidence -> existing_approaches -> common_limitation -> "
    "whats_missing -> potential_direction. "
    "Immediately attach a research direction to each gap with: "
    "why_it_matters, existing_related_work, what_would_be_different, "
    "possible_question, possible_methodology, possible_evaluation, risks.\n\n"
    "Every gap MUST be tagged with epistemic_label, one of exactly: "
    "'directly_supported', 'reasonable_synthesis', 'hypothesis', 'speculative'. "
    "NEVER claim something is 'definitely novel' or use similarly absolute "
    "certainty language — use hedged language appropriate to the label.\n\n"
    "For supporting_paper_ids, use ONLY the numbers from the numbered paper "
    "list given to you (as strings, e.g. \"1\", \"2\") — never invent an id, "
    "never use a paper's title or arXiv URL there, only the number.\n\n"
    "Return JSON: {\"gaps\": [{\"id\": \"g1\", \"observed_evidence\": \"...\", "
    "\"existing_approaches\": \"...\", \"common_limitation\": \"...\", "
    "\"whats_missing\": \"...\", \"potential_direction\": \"...\", "
    "\"epistemic_label\": \"...\", \"supporting_paper_ids\": [\"1\"], "
    "\"direction\": {\"why_it_matters\": \"...\", \"existing_related_work\": \"...\", "
    "\"what_would_be_different\": \"...\", \"possible_question\": \"...\", "
    "\"possible_methodology\": \"...\", \"possible_evaluation\": \"...\", "
    "\"risks\": \"...\"}}]}"
)

FORBIDDEN_PHRASES = ["definitely novel", "certainly novel", "guaranteed to be novel"]


def gaps_and_directions_node(state: GraphState) -> dict:
    current = state["current_round"]
    synthesis = current.get("synthesis", "")
    approved = current.get("approved_papers", [])

    # Build our OWN deterministic mapping from a simple number to the real
    # paper id (the arXiv URL). The model only ever sees and returns the
    # simple number — we do the translation back to the real id ourselves,
    # so citation checking downstream can never fail on a mismatched id
    # the model invented or mistyped.
    index_to_real_id = {str(i + 1): p["id"] for i, p in enumerate(approved)}
    paper_table = "\n".join(
        f"{i + 1}. {p.get('title', '')}" for i, p in enumerate(approved)
    )

    result = call_llm_json(
        GAPS_SYSTEM,
        f"Numbered papers:\n{paper_table}\n\nSynthesis:\n{synthesis}",
    )
    raw_gaps = result.get("gaps", []) if isinstance(result, dict) else []

    gaps, directions = [], []
    for g in raw_gaps:
        # Step 30: validate no forbidden certainty language leaked through.
        combined_text = " ".join(
            str(g.get(k, "")) for k in ("potential_direction", "observed_evidence")
        ).lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in combined_text:
                g["potential_direction"] += " [NOTE: certainty language flagged and should be hedged]"

        # Translate the model's simple numbers back to real paper ids.
        # Any number the model returns that isn't in our table is dropped
        # rather than passed through as a fake id.
        raw_ids = g.get("supporting_paper_ids", [])
        g["supporting_paper_ids"] = [
            index_to_real_id[str(rid)] for rid in raw_ids if str(rid) in index_to_real_id
        ]

        gaps.append({k: v for k, v in g.items() if k != "direction"})
        direction = g.get("direction", {})
        direction["gap_id"] = g.get("id")
        directions.append(direction)

    return {"current_round": {**current, "gaps": gaps, "directions": directions}}