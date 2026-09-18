"""
Phase 2 — Orchestrator (Steps 9-11).

Step 9: deterministic short-circuit — no research_rounds yet -> always
new_research, no LLM call needed.
Step 10: LLM classification for the non-empty case, constrained to a single
word so downstream routing never has to parse free text.
"""
from state import GraphState
from llm import call_llm

CLASSIFY_SYSTEM = (
    "You classify a user's latest message in an ongoing literature-review "
    "assistant conversation. Reply with EXACTLY one word: "
    "'new_research' if the user wants a new topic researched, or "
    "'follow_up' if they're asking about research already done in this "
    "session (comparisons, summaries, clarifications, direction requests "
    "about existing rounds). No punctuation, no explanation."
)


def orchestrator_node(state: GraphState) -> dict:
    if not state.get("research_rounds"):
        # Step 9: deterministic short-circuit
        return {"intent": "new_research"}

    last_user_msg = state["messages"][-1].content
    round_topics = "\n".join(
        f"- {r['round_id']}: {r['topic']}" for r in state["research_rounds"]
    )
    user_prompt = (
        f"Existing research rounds so far:\n{round_topics}\n\n"
        f"User's latest message: {last_user_msg}"
    )
    raw = call_llm(CLASSIFY_SYSTEM, user_prompt).strip().lower()
    intent = "follow_up" if "follow" in raw else "new_research"
    return {"intent": intent}


def route_by_intent(state: GraphState) -> str:
    """Conditional edge target (Step 11)."""
    return "generate_search_queries" if state["intent"] == "new_research" else "resolve_round"
