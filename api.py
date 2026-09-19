"""
Minimal FastAPI backend over the existing compiled LangGraph, so you can
drive it from a browser instead of main.py's CLI. Does not replace main.py
— both use the same compiled_graph / same checkpointed threads.

Run:
    uvicorn api:app --reload --port 8000

Then open frontend/index.html in a browser (it talks to localhost:8000).
"""
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command

from graph import compiled_graph

app = FastAPI(title="Literature Review Assistant API")

# Local-dev CORS: wide open on purpose since this only runs on your machine
# for testing against a local frontend file. Tighten this to a specific
# origin (no "*") before this ever runs anywhere reachable by anyone else.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    resume: dict


def _format_result(result: dict) -> dict:
    """
    Shapes a graph.invoke() result into what the frontend needs: either
    "this turn paused on a HITL interrupt, here's what to show the user
    and what shape of value to resume with", or "this turn finished, here's
    the final markdown/answer text".
    """
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return {"status": "interrupted", "interrupt": payload}
    return {"status": "done", "final_output": result.get("final_output", "")}


@app.post("/chat")
def chat(req: ChatRequest):
    """Start a new thread (if thread_id omitted) or continue one with a new message."""
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = compiled_graph.invoke({"messages": [("human", req.message)]}, config=config)
    return {"thread_id": thread_id, **_format_result(result)}


@app.post("/resume")
def resume(req: ResumeRequest):
    """
    Resume a thread paused on a HITL interrupt. The shape of `resume` depends
    on the interrupt type the /chat or /resume call before it returned:
      - paper_selection    -> {"approve_all": true}  or  {"approved_ids": ["..."]}
      - round_clarification -> {"round_id": "..."}
    """
    config = {"configurable": {"thread_id": req.thread_id}}
    result = compiled_graph.invoke(Command(resume=req.resume), config=config)
    return {"thread_id": req.thread_id, **_format_result(result)}


@app.get("/health")
def health():
    return {"status": "ok"}