"""
Wires every node from every phase into the StateGraph shown in the
architecture diagram. Checkpointer gives persistence (Phase 3) and is what
makes `interrupt(...)` (Phase 5 / Phase 10) actually pausable/resumable.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import GraphState
from nodes.orchestrator import orchestrator_node, route_by_intent
from nodes.search import (
    generate_search_queries_node,
    academic_search_node,
    prefilter_node,
    relevance_ranking_node,
    check_search_sufficiency,
    bump_retry_node,
)
from nodes.hitl_selection import paper_selection_node
from nodes.analysis import fan_out_analysis, analyze_single_paper_node
from nodes.synthesis import synthesis_node
from nodes.gaps import gaps_and_directions_node
from nodes.citations import citation_check_node
from nodes.finalize import finalize_round_node
from nodes.follow_up import (
    resolve_round_node,
    check_round_resolved,
    clarify_round_node,
    follow_up_answer_node,
)
from nodes.render import render_node


def build_graph():
    graph = StateGraph(GraphState)

    # --- nodes ---
    graph.add_node("orchestrator", orchestrator_node)

    graph.add_node("generate_search_queries", generate_search_queries_node)
    graph.add_node("academic_search", academic_search_node)
    graph.add_node("prefilter", prefilter_node)
    graph.add_node("relevance_ranking", relevance_ranking_node)
    graph.add_node("bump_retry", bump_retry_node)

    graph.add_node("paper_selection", paper_selection_node)  # HITL interrupt

    graph.add_node("analyze_single_paper", analyze_single_paper_node)  # fan-out target
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("gaps_and_directions", gaps_and_directions_node)
    graph.add_node("citation_check", citation_check_node)
    graph.add_node("finalize_round", finalize_round_node)

    graph.add_node("resolve_round", resolve_round_node)
    graph.add_node("clarify_round", clarify_round_node)  # HITL interrupt
    graph.add_node("follow_up_answer", follow_up_answer_node)

    graph.add_node("render", render_node)

    # --- edges ---
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_by_intent,
        {"generate_search_queries": "generate_search_queries", "resolve_round": "resolve_round"},
    )

    # New-research branch
    graph.add_edge("generate_search_queries", "academic_search")
    graph.add_edge("academic_search", "prefilter")
    graph.add_edge("prefilter", "relevance_ranking")
    graph.add_conditional_edges(
        "relevance_ranking",
        check_search_sufficiency,
        {"retry": "bump_retry", "proceed": "paper_selection"},
    )
    graph.add_edge("bump_retry", "generate_search_queries")

    graph.add_conditional_edges("paper_selection", fan_out_analysis, ["analyze_single_paper", "synthesis"])
    graph.add_edge("analyze_single_paper", "synthesis")  # fan-in: reducer collects all branches
    graph.add_edge("synthesis", "gaps_and_directions")
    graph.add_edge("gaps_and_directions", "citation_check")
    graph.add_edge("citation_check", "finalize_round")
    graph.add_edge("finalize_round", "render")

    # Follow-up branch
    graph.add_conditional_edges(
        "resolve_round",
        check_round_resolved,
        {"resolved": "follow_up_answer", "ambiguous": "clarify_round"},
    )
    graph.add_edge("clarify_round", "follow_up_answer")
    graph.add_edge("follow_up_answer", "render")

    graph.add_edge("render", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


compiled_graph = build_graph()