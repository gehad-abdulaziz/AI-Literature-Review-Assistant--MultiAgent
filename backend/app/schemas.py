"""
Pydantic models for the Literature Review API. Kept separate from graph.py
so the API contract can evolve independently of the internal graph state
shape.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PaperCandidate(BaseModel):
    id: str
    title: str
    authors: list[str]
    year: Optional[int] = None
    abstract: str
    url: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class StartRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Research topic to investigate.")


class ReviewGap(BaseModel):
    id: str
    title: str
    description: str
    potential_direction: str


class ReviewReport(BaseModel):
    topic: str
    synthesis: str
    comparison: str
    gaps: list[ReviewGap]
    papers_used: list[PaperCandidate]


class GraphStepStatus(BaseModel):
    """Describes where the graph currently is, for the UI stepper."""

    step: Literal[
        "query_expansion",
        "paper_fetching",
        "human_selection",
        "review_synthesis",
        "done",
    ]
    paused: bool = False


class StartResponse(BaseModel):
    thread_id: str
    status: GraphStepStatus
    search_queries: list[str]
    candidates: list[PaperCandidate]


class ResumeRequest(BaseModel):
    thread_id: str
    approved_ids: Optional[list[str]] = None
    approve_all: bool = False


class ResumeResponse(BaseModel):
    thread_id: str
    status: GraphStepStatus
    report: Optional[ReviewReport] = None
