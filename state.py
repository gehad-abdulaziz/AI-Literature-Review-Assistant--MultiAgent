"""
Phase 1 — State Design (Steps 6-8).

Design notes (paper-first, then code):
- research_rounds: grows only, never overwritten -> needs an accumulating reducer.
- messages: conversation history -> accumulating reducer (add_messages).
- intent / resolved_round_id: written by exactly one node per turn -> plain
  overwritable fields, no reducer needed.
- current_round: scratch space for an in-progress pipeline run. Overwritable;
  it gets folded into research_rounds at the very end (Step 38) and is not
  itself accumulated.
- paper_analyses: written in parallel by fan-out branches (Phase 6) -> needs
  an accumulating reducer or concurrent writes will clobber each other.
"""
import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Small structured shapes. Kept as plain dicts/TypedDicts (not pydantic) so
# they drop straight into LangGraph state without extra serialization work.
# ---------------------------------------------------------------------------

class Paper(TypedDict, total=False):
    id: str                # external id (e.g. arXiv id) — used by citation check
    title: str
    authors: list[str]
    year: Optional[int]
    abstract: str
    url: str


class PaperAnalysis(TypedDict, total=False):
    paper_id: str
    title: str
    authors: list[str]
    year: Optional[int]
    problem: str
    methodology: str
    dataset: str
    findings: str
    limitations: str
    failed: bool            # True if this branch errored (Step 25)
    failure_reason: str


class Gap(TypedDict, total=False):
    id: str
    observed_evidence: str
    existing_approaches: str
    common_limitation: str
    whats_missing: str
    potential_direction: str
    epistemic_label: Literal[
        "directly_supported", "reasonable_synthesis", "hypothesis", "speculative"
    ]
    supporting_paper_ids: list[str]


class ResearchDirection(TypedDict, total=False):
    gap_id: str
    why_it_matters: str
    existing_related_work: str
    what_would_be_different: str
    possible_question: str
    possible_methodology: str
    possible_evaluation: str
    risks: str


class CitationCheck(TypedDict, total=False):
    claim: str
    paper_id: str
    exists: bool
    support_label: Literal["supported", "weakly_supported", "unsupported"]


class ResearchRound(TypedDict, total=False):
    round_id: str
    topic: str
    papers: list[Paper]
    analyses: list[PaperAnalysis]
    synthesis: str
    gaps: list[Gap]
    directions: list[ResearchDirection]
    citation_checks: list[CitationCheck]
    timestamp: str


def keep_last(existing: Any, new: Any) -> Any:
    """Plain overwrite reducer (explicit, for readability at call sites)."""
    return new


class CurrentRound(TypedDict, total=False):
    """Scratch area for the in-progress new-research pipeline run (Step 6)."""
    round_id: str
    topic: str
    search_queries: list[str]
    raw_results: list[Paper]
    filtered_results: list[Paper]
    ranked_papers: list[Paper]      # after LLM relevance ranking
    approved_papers: list[Paper]    # after HITL selection (Phase 5)
    retry_count: int


class GraphState(TypedDict):
    # Accumulating fields
    messages: Annotated[list, add_messages]
    research_rounds: Annotated[list[ResearchRound], operator.add]
    paper_analyses: Annotated[list[PaperAnalysis], operator.add]

    # Overwritable, single-writer-per-turn fields
    intent: Literal["new_research", "follow_up"]
    resolved_round_id: Optional[str]
    current_round: CurrentRound
    final_output: str  # the rendered Markdown for this turn
