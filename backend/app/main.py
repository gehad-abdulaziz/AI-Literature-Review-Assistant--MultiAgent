"""
FastAPI service exposing the Literature Review LangGraph as two endpoints:

    POST /api/start    -> kicks off a new thread, runs to the human-selection
                           interrupt, returns candidate papers.
    POST /api/resume   -> resumes a paused thread with the user's paper
                           selection, runs to completion, returns the report.
    GET  /api/health   -> liveness check.

Run with:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command

from app.graph import compiled_graph
from app.schemas import (
    GraphStepStatus,
    PaperCandidate,
    ResumeRequest,
    ResumeResponse,
    ReviewGap,
    ReviewReport,
    StartRequest,
    StartResponse,
)

logger = logging.getLogger("literature_review_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Literature Review Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_candidate(p: dict) -> PaperCandidate:
    return PaperCandidate(
        id=p["id"],
        title=p["title"],
        authors=p["authors"],
        year=p.get("year"),
        abstract=p["abstract"],
        url=p["url"],
        relevance_score=p["relevance_score"],
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/start", response_model=StartResponse)
def start_review(req: StartRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = compiled_graph.invoke({"topic": req.topic}, config=config)
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Graph invocation failed on /api/start")
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {e}") from e

    if "__interrupt__" not in result:
        # Shouldn't happen given the fixed graph shape, but guard anyway
        # rather than silently returning an incomplete response.
        raise HTTPException(
            status_code=500,
            detail="Graph completed without pausing for human selection.",
        )

    interrupt_payload = result["__interrupt__"][0].value
    candidates = [_to_candidate(p) for p in interrupt_payload["candidates"]]

    return StartResponse(
        thread_id=thread_id,
        status=GraphStepStatus(step="human_selection", paused=True),
        search_queries=result.get("search_queries", []),
        candidates=candidates,
    )


@app.post("/api/resume", response_model=ResumeResponse)
def resume_review(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    resume_value = (
        {"approve_all": True}
        if req.approve_all or not req.approved_ids
        else {"approved_ids": req.approved_ids}
    )

    try:
        result = compiled_graph.invoke(Command(resume=resume_value), config=config)
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Graph invocation failed on /api/resume")
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {e}") from e

    if "__interrupt__" in result:
        # Not expected in this simplified single-pause graph, but handle
        # gracefully rather than crashing if the graph shape changes later.
        return ResumeResponse(
            thread_id=req.thread_id,
            status=GraphStepStatus(step="human_selection", paused=True),
            report=None,
        )

    report = ReviewReport(
        topic=result.get("topic", ""),
        synthesis=result.get("synthesis", ""),
        comparison=result.get("comparison", ""),
        gaps=[ReviewGap(**g) for g in result.get("gaps", [])],
        papers_used=[_to_candidate(p) for p in result.get("approved_papers", [])],
    )

    return ResumeResponse(
        thread_id=req.thread_id,
        status=GraphStepStatus(step="done", paused=False),
        report=report,
    )
