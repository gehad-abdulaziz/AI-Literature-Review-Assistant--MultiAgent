"""
Phase 11 / Step 39 — pure formatting, no LLM call.
"""
from state import GraphState


def render_round_markdown(round_data: dict) -> str:
    papers = round_data.get("papers", [])
    analyses = round_data.get("analyses", [])
    gaps = round_data.get("gaps", [])
    directions = round_data.get("directions", [])
    checks = round_data.get("citation_checks", [])

    lines = [f"# Literature Review: {round_data.get('topic', '')}", ""]

    lines += ["## Research Scope", f"Topic: {round_data.get('topic', '')}", ""]

    lines += ["## Search Strategy", "Source: arXiv", ""]

    lines += ["## Selected Papers", "", "| Title | Authors | Year |", "|---|---|---|"]
    for p in papers:
        authors = ", ".join(p.get("authors", [])[:3])
        lines.append(f"| {p.get('title','')} | {authors} | {p.get('year','')} |")
    lines.append("")

    lines += ["## Paper-by-Paper Analysis", ""]
    for a in analyses:
        lines += [
            f"### {a.get('title','')}",
            f"- **Problem:** {a.get('problem','')}",
            f"- **Methodology:** {a.get('methodology','')}",
            f"- **Dataset:** {a.get('dataset','')}",
            f"- **Findings:** {a.get('findings','')}",
            f"- **Limitations:** {a.get('limitations','')}",
            "",
        ]

    lines += ["## Cross-Paper Comparison & Synthesis", "", round_data.get("synthesis", ""), ""]

    lines += ["## Research Gaps", ""]
    for g in gaps:
        lines += [
            f"### Gap {g.get('id','')} _(label: {g.get('epistemic_label','')})_",
            f"- **Observed evidence:** {g.get('observed_evidence','')}",
            f"- **Existing approaches:** {g.get('existing_approaches','')}",
            f"- **Common limitation:** {g.get('common_limitation','')}",
            f"- **What's missing:** {g.get('whats_missing','')}",
            "",
        ]

    lines += ["## Potential Directions", ""]
    for d in directions:
        lines += [
            f"### Direction for {d.get('gap_id','')}",
            f"- **Why it matters:** {d.get('why_it_matters','')}",
            f"- **Existing related work:** {d.get('existing_related_work','')}",
            f"- **What would be different:** {d.get('what_would_be_different','')}",
            f"- **Possible question:** {d.get('possible_question','')}",
            f"- **Possible methodology:** {d.get('possible_methodology','')}",
            f"- **Possible evaluation:** {d.get('possible_evaluation','')}",
            f"- **Risks:** {d.get('risks','')}",
            "",
        ]

    lines += ["## Evidence / Citations", ""]
    for c in checks:
        flag = "✅" if c.get("support_label") == "supported" else "⚠️"
        lines.append(f"- {flag} [{c.get('paper_id')}] {c.get('support_label')}: {c.get('claim')}")
    lines.append("")

    return "\n".join(lines)


def render_node(state: GraphState) -> dict:
    """
    Runs after either a freshly finalized round (new_research path) or a
    follow-up answer (follow_up path). Picks whichever is available.
    """
    if state.get("final_output"):
        # Follow-up path already produced its answer text.
        return {"final_output": state["final_output"]}

    # New-research path: render the round that was just finalized (the last
    # entry appended to research_rounds).
    latest_round = state["research_rounds"][-1]
    md = render_round_markdown(latest_round)
    return {"final_output": md}
