"""
Phase 10 — Follow-up Path: Round Resolution + HITL (Steps 34-37).
"""
from langgraph.types import interrupt

from state import GraphState
from llm import call_llm, LLMCallError


def resolve_round_node(state: GraphState) -> dict:
    """Step 34: resolve immediately if trivial, else attempt keyword matching."""
    rounds = state["research_rounds"]
    last_msg = state["messages"][-1].content.lower()

    if len(rounds) == 1:
        return {"resolved_round_id": rounds[0]["round_id"]}

    # Simple keyword matching first (Step 34 says: start simple, only add
    # an LLM matcher if this proves insufficient in practice).
    matches = [
        r for r in rounds
        if any(word in last_msg for word in r["topic"].lower().split() if len(word) > 3)
    ]
    if len(matches) == 1:
        return {"resolved_round_id": matches[0]["round_id"]}

    # Ambiguous (0 or 2+ matches) — leave unresolved, conditional edge below
    # routes to the clarification interrupt.
    return {"resolved_round_id": None}


def check_round_resolved(state: GraphState) -> str:
    return "resolved" if state.get("resolved_round_id") else "ambiguous"


def clarify_round_node(state: GraphState) -> dict:
    """Step 35-36: interrupt, ask which round, resume with the chosen id."""
    rounds = state["research_rounds"]
    payload = {
        "type": "round_clarification",
        "message": "Which research round are you asking about?",
        "options": [{"round_id": r["round_id"], "topic": r["topic"]} for r in rounds],
    }
    user_response = interrupt(payload)
    round_id = user_response.get("round_id") if isinstance(user_response, dict) else None
    return {"resolved_round_id": round_id}


FOLLOW_UP_SYSTEM = (
    "You answer a follow-up question about a SINGLE already-completed "
    "research round. Use ONLY the data given below — do not perform new "
    "search or analysis, and do not reference any other round. Handle "
    "aggregation questions (e.g. most common limitations), comparison "
    "questions between specific papers in this round, or requests to "
    "revisit a research direction."
)


def follow_up_answer_node(state: GraphState) -> dict:
    """Step 37. One shared node parameterized by the question; split later if unwieldy."""
    round_id = state["resolved_round_id"]
    round_data = next(r for r in state["research_rounds"] if r["round_id"] == round_id)
    last_msg = state["messages"][-1].content

    context = (
        f"Round topic: {round_data['topic']}\n"
        f"Papers: {[p['title'] for p in round_data['papers']]}\n"
        f"Synthesis: {round_data['synthesis']}\n"
        f"Gaps: {round_data['gaps']}\n"
        f"Directions: {round_data['directions']}"
    )

    try:
        answer = call_llm(FOLLOW_UP_SYSTEM, f"{context}\n\nQuestion: {last_msg}")
    except LLMCallError:
        answer = (
            "Sorry — I couldn't reach the language model to answer that just "
            "now. Please try again in a moment."
        )

    return {"final_output": answer}