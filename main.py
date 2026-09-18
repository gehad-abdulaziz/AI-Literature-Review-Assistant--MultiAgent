"""
Minimal CLI to drive the graph and demonstrate the two interrupt points.

Usage:
    python main.py

This uses a fixed thread_id per run so state persists across your turns in
one session (Phase 3). Start a new thread_id for a different user/session.
"""
import uuid

from langgraph.types import Command

from graph import compiled_graph


def run():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("Literature Review Assistant. Type your research topic (or 'quit').\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            # Don't send an empty message into the graph — it has no
            # question to answer and would just re-summarize the last
            # round again. Just re-prompt instead.
            continue

        result = compiled_graph.invoke({"messages": [("human", user_input)]}, config=config)

        # If the graph paused on an interrupt, result contains "__interrupt__".
        while "__interrupt__" in result:
            interrupt_payload = result["__interrupt__"][0].value
            print(f"\n[PAUSED] {interrupt_payload['message']}")

            if interrupt_payload["type"] == "paper_selection":
                for c in interrupt_payload["candidates"]:
                    print(f"  - {c['id']}: {c['title']} ({c.get('year')})")
                raw = input("Enter comma-separated ids to keep, or blank for all: ").strip()
                resume_value = (
                    {"approve_all": True} if not raw
                    else {"approved_ids": [x.strip() for x in raw.split(",")]}
                )

            elif interrupt_payload["type"] == "round_clarification":
                for o in interrupt_payload["options"]:
                    print(f"  - {o['round_id']}: {o['topic']}")
                raw = input("Enter the round_id you mean: ").strip()
                resume_value = {"round_id": raw}

            else:
                resume_value = {}

            result = compiled_graph.invoke(Command(resume=resume_value), config=config)

        print(f"\nAssistant:\n{result.get('final_output', '(no output)')}\n")


if __name__ == "__main__":
    run()